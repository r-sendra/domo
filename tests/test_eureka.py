"""Eureka machinery — pure parts, no physics/subprocess/network."""

import pytest
import torch

from domo.eureka import TASK_REGISTRY, CandidateResult, IterationResult
from domo.eureka.prompts import dr_prompt, reflection_block, reward_prompt
from domo.eureka.rewards import (extract_reward_code, load_reward_fn,
                                 validate_reward_code)
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
    text = dr_prompt("getup", 0.6, "  friction=0.5: success 40%  [FEASIBLE ]")
    assert "domain randomization" in text and "40%" in text
    dr = DomainRandomization.from_dict(
        {"friction_range": [0.5, 1.5], "obs_noise_std": 0.02,
         "unknown_key": 1})
    assert dr.friction_range == (0.5, 1.5)
    assert dr.obs_noise_std == 0.02
    assert "unknown_key" not in dr.to_dict()
    assert DomainRandomization.from_dict(dr.to_dict()).friction_range == (0.5, 1.5)


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
