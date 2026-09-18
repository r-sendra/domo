"""Eureka machinery — pure parts, no physics/subprocess/network."""

import pytest
import torch

from domo.eureka import TASK_REGISTRY, CandidateResult, IterationResult
from domo.eureka.prompts import dr_prompt, reflection_block, reward_prompt
from domo.eureka.rewards import extract_reward_code, load_reward_fn, validate_reward_code
from domo.eureka.spec import EurekaConfig, SkillLearningRequest
from domo.llm.client import ScriptedClient, extract_code_block, extract_json_block
from domo.robot.randomization import DomainRandomization

N = 3

GOOD_CODE = """\
def compute_reward(task):
    state = task.robot.state
    up = torch.clamp(-state.projected_gravity[:, 2], 0.0, 1.0)
    energy = -0.001 * task.actions.pow(2).sum(-1)
    return up + energy, {"up": up, "energy": energy}
"""


class FakeTask:
    def __init__(self):
        self.n_envs = N

        class S:
            projected_gravity = torch.tensor([[0.0, 0.0, -1.0]] * N)

        class R:
            state = S()

        self.robot = R()
        self.actions = torch.zeros(N, 12)


# ---------------------------------------------------------------------------
# Code extraction / validation / loading
# ---------------------------------------------------------------------------

def test_extract_code_and_json():
    text = f"Sure!\n```python\n{GOOD_CODE}```\ndone"
    assert "compute_reward" in extract_code_block(text)
    assert extract_reward_code(text) is not None
    assert extract_reward_code("no code here") is None
    assert extract_json_block('bla ```json\n{"a": 1}``` ') == '{"a": 1}'
    assert extract_json_block('inline {"b": [1,2]} tail') == '{"b": [1,2]}'


def test_extract_tolerates_gemini_variants():
    # capitalised / short language tags
    assert extract_reward_code(f"```py\n{GOOD_CODE}```") is not None
    assert extract_reward_code(f"```Python\n{GOOD_CODE}```") is not None
    # TRUNCATED block: opening fence, no closing fence (thinking-model cutoff)
    truncated = f"Here you go:\n```python\n{GOOD_CODE}"
    assert extract_reward_code(truncated) is not None
    # empty response → None, not a crash
    assert extract_reward_code("") is None
    assert extract_reward_code(None) is None


def test_validate_rejects_dangerous_and_broken():
    assert validate_reward_code(GOOD_CODE) is None
    # benign imports the LLM writes by habit are now allowed
    assert validate_reward_code("import torch\nimport math\n" + GOOD_CODE) is None
    # other imports are still rejected
    assert "disallowed import" in validate_reward_code("import os\n" + GOOD_CODE)
    assert "disallowed import" in validate_reward_code("import numpy as np\n" + GOOD_CODE)
    # subprocess is caught by the substring guard (rejected, either message)
    assert validate_reward_code("from subprocess import run\n" + GOOD_CODE) is not None
    assert "forbidden" in validate_reward_code(
        "def compute_reward(task):\n    return task.__dict__, {}")
    assert "syntax" in validate_reward_code("def compute_reward(task:")
    assert "does not define" in validate_reward_code("x = 1")


def test_load_reward_fn_runs_and_checks_output():
    fn = load_reward_fn(GOOD_CODE)
    rew, comps = fn(FakeTask())
    assert rew.shape == (N,) and torch.allclose(rew, torch.ones(N), atol=0.01)
    assert set(comps) == {"up", "energy"}

    bad = "def compute_reward(task):\n    return torch.zeros(1), {}"
    with pytest.raises(ValueError, match="shape"):
        load_reward_fn(bad)(FakeTask())
    bad2 = "def compute_reward(task):\n    return torch.zeros(3)"
    with pytest.raises(TypeError):
        load_reward_fn(bad2)(FakeTask())


# ---------------------------------------------------------------------------
# Prompts / reflection
# ---------------------------------------------------------------------------

def _request():
    return SkillLearningRequest(skill_name="getup", description="Stand up.",
                                eureka=EurekaConfig(iterations=1, samples=2))


def test_reward_prompt_contains_contract_and_interface():
    text = reward_prompt(_request(), TASK_REGISTRY["go2_getup"])
    for token in ("compute_reward", "projected_gravity", "0.26",
                  "Stand up.", "FIXED"):
        assert token in text, token


def test_reflection_includes_stats_and_failures():
    good = CandidateResult(index=0, code=GOOD_CODE, success_rate=0.55,
                           mean_ep_len=120.0, snapshots=[
                               {"frac": 0.5, "components": {"up": 0.4},
                                "success_rate": 0.2, "mean_ep_len": 200},
                               {"frac": 1.0, "components": {"up": 0.8},
                                "success_rate": 0.55, "mean_ep_len": 120}])
    bad = CandidateResult(index=1, code="x", error="Traceback...\nKeyError: 'y'")
    text = reflection_block([good, bad])
    assert "55%" in text and "FAILED" in text and "KeyError" in text
    assert "0.4" in text.replace("+0.4000", "0.4")   # component trajectory
    it = IterationResult(index=0, candidates=[good, bad])
    assert it.best is good


def test_dr_prompt_and_config_roundtrip():
    text = dr_prompt("getup", 0.6, "friction: feasible [0.5, 2.0]",
                     "  friction=0.5: success 40%  [FEASIBLE ]")
    assert "domain randomization" in text and "40%" in text
    assert "feasible" in text
    dr = DomainRandomization.from_dict(
        {"friction_range": [0.5, 1.5], "com_shift_range": [-0.05, 0.05],
         "obs_noise_std": 0.02, "unknown_key": 1})
    assert dr.friction_range == (0.5, 1.5)
    assert dr.com_shift_range == (-0.05, 0.05)
    assert dr.obs_noise_std == 0.02
    assert "unknown_key" not in dr.to_dict()
    assert DomainRandomization.from_dict(dr.to_dict()).com_shift_range == (-0.05, 0.05)


def test_safety_instruction_in_reward_prompt():
    from domo.eureka.prompts import reward_prompt
    from domo.eureka.spec import TASK_REGISTRY, EurekaConfig, SkillLearningRequest
    req = SkillLearningRequest(skill_name="g", description="Stand up.",
                               eureka=EurekaConfig(iterations=1, samples=1))
    ts = TASK_REGISTRY["go2_getup"]
    with_safety = reward_prompt(req, ts, safety=True)
    without = reward_prompt(req, ts, safety=False)
    assert "sim-to-real" in with_safety.lower() and "smoothness" in with_safety.lower()
    assert "sim-to-real" not in without.lower()


def test_rapp_feasible_bounds():
    from domo.eureka.dr import feasible_bounds
    from domo.eureka.spec import DrEurekaConfig
    cfg = DrEurekaConfig(feasible_ratio=0.5, feasible_floor=0.1)
    # nominal 0.6 → threshold 0.30. friction feasible at 0.5 and 1.5, not 4.0.
    sweeps = [
        {"label": "nominal", "param": None, "value": 0.0, "success_rate": 0.6},
        {"label": "friction=0.5", "param": "friction", "value": 0.5, "success_rate": 0.5},
        {"label": "friction=1.5", "param": "friction", "value": 1.5, "success_rate": 0.4},
        {"label": "friction=4.0", "param": "friction", "value": 4.0, "success_rate": 0.05},
        {"label": "com_shift=0.1", "param": "com_shift", "value": 0.1, "success_rate": 0.45},
    ]
    bounds, text = feasible_bounds(sweeps, cfg, nominal=0.6)
    # friction feasible from default 1.0 down to 0.5 and up to 1.5 (not 4.0)
    assert bounds["friction_range"] == (0.5, 1.5)
    # com symmetric: feasible magnitude 0.1 → [-0.1, 0.1]
    assert bounds["com_shift_range"] == (-0.1, 0.1)
    assert "feasible" in text


def test_dr_config_clamped_to_bounds():
    from domo.eureka.dr import _clamp_to_bounds
    bounds = {"friction_range": (0.5, 1.5), "obs_noise_std": (0.0, 0.05)}
    # LLM proposes wider than feasible → clamped inward
    out = _clamp_to_bounds({"friction_range": [0.2, 3.0], "obs_noise_std": 0.2}, bounds)
    assert out["friction_range"] == [0.5, 1.5]
    assert out["obs_noise_std"] == 0.05


def test_make_llm_dispatch():
    from domo.llm import make_llm
    from domo.llm.client import ScriptedClient
    assert isinstance(make_llm("scripted", responses=["a"]), ScriptedClient)
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_llm("nonesuch")


def test_langgraph_pipeline_structure():
    from domo.eureka.graph import build_graph
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    assert {"iterate", "select", "dr", "finish"} <= nodes


def test_dense_fitness_breaks_zero_success_ties():
    # Two candidates, both 0% success, different dense fitness → the one that
    # made more progress must rank first (the fix for the flat-search bug).
    a = CandidateResult(index=0, code="a", success_rate=0.0, fitness=0.12)
    b = CandidateResult(index=1, code="b", success_rate=0.0, fitness=0.47)
    it = IterationResult(index=0, candidates=[a, b])
    assert it.best is b
    # success still dominates fitness when present
    c = CandidateResult(index=2, code="c", success_rate=0.3, fitness=0.05)
    it2 = IterationResult(index=1, candidates=[a, b, c])
    assert it2.best is c


def test_record_stats_dicts_and_legacy():
    from domo.eureka.worker import _record_stats
    recs = [{"success": True, "fitness": 1.0, "peak_height": 0.30, "ever_upright": True},
            {"success": False, "fitness": 0.4, "peak_height": 0.15, "ever_upright": False}]
    s = _record_stats(recs)
    assert s["success_rate"] == 0.5
    assert abs(s["fitness"] - 0.7) < 1e-6
    assert s["ever_upright_rate"] == 0.5
    # legacy list-of-bools still parses (fitness mirrors success)
    s2 = _record_stats([True, False, False])
    assert abs(s2["success_rate"] - 1 / 3) < 1e-6 and s2["fitness"] == s2["success_rate"]


def test_reflection_uses_fitness_when_no_success():
    # Progress but never reaches the goal → generic "make more progress" note.
    good = CandidateResult(index=0, code=GOOD_CODE, success_rate=0.0,
                           fitness=0.42, peak_height=0.22, ever_upright_rate=0.0,
                           max_hold=0.0)
    worse = CandidateResult(index=1, code="x", success_rate=0.0, fitness=0.05)
    text = reflection_block([good, worse])
    assert "fitness 0.42" in text and "no candidate reached" in text
    assert "Best candidate was #0" in text          # chosen by fitness, not success


def test_reflection_diagnoses_ballistic_hold_failure():
    # Reaches the goal often but holds ~1 step → targeted "settle" diagnosis.
    c = CandidateResult(index=0, code=GOOD_CODE, success_rate=0.0,
                        fitness=0.5, peak_height=0.31, ever_upright_rate=0.28,
                        max_hold=1.0)
    text = reflection_block([c])
    assert "REACHES the goal pose" in text and "does NOT HOLD" in text
    assert "velocity" in text and "settl" in text.lower()


# ---------------------------------------------------------------------------
# Reward override on a VecTask (no physics: exercise base-class machinery)
# ---------------------------------------------------------------------------

def test_vectask_reward_override_components():
    from domo.tasks.base import VecTask

    class Dummy(VecTask):
        def step(self, a): ...
        def reset(self): ...
        def reset_idx(self, e): ...

    task = Dummy(n_envs=N, num_obs=4, num_actions=2,
                 device=torch.device("cpu"), dt=0.02, max_episode_length=10)
    task.set_reward_override(
        lambda t: (torch.full((N,), 2.0), {"a": torch.ones(N)}))
    rew = task.compute_rewards()
    assert torch.allclose(rew, torch.full((N,), 2.0))
    assert torch.allclose(task.episode_sums["a"], torch.ones(N))
    task.compute_rewards()
    assert torch.allclose(task.episode_sums["a"], 2 * torch.ones(N))
    with pytest.raises(NotImplementedError):
        task.compute_success()


def test_scripted_client_records_prompts():
    llm = ScriptedClient(["one", "two"])
    assert llm.generate("p1") == "one"
    assert llm.generate("p2") == "two"
    assert llm.generate("p3") == "one"          # cycles
    assert llm.calls == ["p1", "p2", "p3"]
