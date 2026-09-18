"""
Eureka / DrEureka drivers end to end WITHOUT physics, LLM, or subprocesses.

The worker subprocess is replaced by a fake ``subprocess.run`` that reads the
spec.json the routine wrote and writes a canned results.json — so the real
``run_worker`` protocol, run-dir layout, ranking, reflection wiring, DR
clamping/selection and the imperative-vs-LangGraph invariant are all
exercised exactly as in production.
"""

from __future__ import annotations

import json
import os
import subprocess
import types

import pytest

from domo.eureka import DrEurekaConfig, EurekaConfig, SkillLearningRequest, learn_skill, routine
from domo.eureka import dr as dr_mod
from domo.eureka.spec import LearnedSkill
from domo.llm.client import ScriptedClient

GOOD_REPLY = """```python
def compute_reward(task):
    r = torch.zeros(task.n_envs)
    return r, {"r": r}
```"""
BAD_IMPORT_REPLY = "```python\nimport os\ndef compute_reward(task):\n    return 0, {}\n```"
NO_CODE_REPLY = "I would rather not."
# Wider than any feasible bound → exercises clamping; kd_scale is not swept by
# RAPP so it must pass through untouched.
DR_REPLY = ('```json\n{"friction_range": [0.2, 3.0], "obs_noise_std": 0.5, '
            '"kd_scale_range": [0.9, 1.1]}\n```')


def _train_results(run_dir: str, success: float, fitness: float = 0.0) -> dict:
    return {"error": None, "success_rate": success, "fitness": fitness,
            "peak_height": 0.2, "ever_upright_rate": 0.0, "max_hold": 0.0,
            "mean_ep_len": 400.0,
            "snapshots": [{"frac": 1.0, "components": {"r": 0.0},
                           "success_rate": success, "fitness": fitness,
                           "peak_height": 0.2, "ever_upright_rate": 0.0,
                           "mean_ep_len": 400.0}],
            "checkpoint": os.path.join(run_dir, "checkpoint_final.pt")}


class FakeWorker:
    """Stands in for ``python -m domo.eureka.worker`` behind ``subprocess.run``.

    ``train`` maps ``(iteration_dir_name, train_dir_name)`` to a results dict
    (or ``None`` to simulate a crash that leaves no results.json);
    ``sweep_success(label)`` gives the dr_eval success per sweep.
    """

    def __init__(self, train, sweep_success=lambda label: 0.0):
        self.train = train
        self.sweep_success = sweep_success
        self.specs: list[dict] = []

    def __call__(self, cmd, timeout, capture_output, text, check):
        assert cmd[1:3] == ["-m", "domo.eureka.worker"] and check is False
        with open(cmd[-1]) as f:
            spec = json.load(f)
        self.specs.append(spec)
        run_dir = spec["run_dir"]
        if spec["mode"] == "train":
            key = (os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir))
            results = self.train(key, run_dir)
        else:
            results = {"error": None, "sweeps": [
                {"label": s["label"], "dr": s["dr"],
                 "success_rate": self.sweep_success(s["label"]),
                 "fitness": 0.5, "mean_ep_len": 400.0} for s in spec["sweeps"]]}
        if results is not None:
            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(results, f)
        return types.SimpleNamespace(stdout="worker stdout", stderr="", returncode=0)


def _install(monkeypatch, worker: FakeWorker) -> FakeWorker:
    monkeypatch.setattr(subprocess, "run", worker)
    return worker


def _request(tmp_path, **kw) -> SkillLearningRequest:
    base = dict(skill_name="getup", description="Stand up.",
                eureka=EurekaConfig(iterations=2, samples=2, n_envs=8,
                                    train_steps=100, eval_episodes=3),
                run_root=str(tmp_path), llm="scripted", use_graph=False)
    base.update(kw)
    return SkillLearningRequest(**base)


def _tree(root: str) -> set[str]:
    out = set()
    for d, _, files in os.walk(root):
        for fn in files:
            out.add(os.path.relpath(os.path.join(d, fn), root))
    return out


# ---------------------------------------------------------------------------
# run_worker protocol
# ---------------------------------------------------------------------------

def test_run_worker_writes_spec_and_reads_results(tmp_path, monkeypatch):
    worker = _install(monkeypatch, FakeWorker(
        lambda key, run_dir: _train_results(run_dir, 0.4)))
    spec = {"mode": "train", "run_dir": str(tmp_path / "t")}
    out = routine.run_worker(spec, timeout_s=10)
    assert out["success_rate"] == 0.4
    assert worker.specs == [spec]                       # spec.json round-trips
    assert (tmp_path / "t" / "spec.json").exists()


def test_run_worker_failures_never_raise(tmp_path, monkeypatch):
    # Crash without results.json → error carries the output tail.
    _install(monkeypatch, FakeWorker(lambda key, run_dir: None))
    out = routine.run_worker({"mode": "train", "run_dir": str(tmp_path / "a")}, 10)
    assert "no results.json" in out["error"] and "worker stdout" in out["error"]

    # Timeout → error mentions the budget.
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="w", timeout=7)
    monkeypatch.setattr(subprocess, "run", boom)
    out = routine.run_worker({"mode": "train", "run_dir": str(tmp_path / "b")}, 7)
    assert "timeout after 7s" in out["error"]


def test_build_train_spec_shared_by_eureka_and_dr(tmp_path):
    req = _request(tmp_path, eureka=EurekaConfig(n_envs=4, rollout_steps=24,
                                                 train_steps=999, eval_episodes=5,
                                                 headless=False, hidden_size=64))
    cand = routine.build_train_spec(req, reward_code_file="c.py", run_dir="d",
                                    total_steps=999, eval_episodes=5, headless=False)
    assert cand["task_overrides"] == {"n_envs": 4, "device": "cuda", "headless": False}
    assert "dr" not in cand["task_overrides"]
    assert cand["ppo"] == {"total_steps": 999, "rollout_steps": 24,
                           "minibatch_size": 64,          # floor for tiny runs
                           "hidden_size": 64, "guard_nonfinite": True,
                           "lr_schedule": "linear"}
    assert cand["eval_episodes"] == 5 and cand["snapshots"] == 4
    dr = routine.build_train_spec(req, reward_code_file="r.py", run_dir="d",
                                  total_steps=5, eval_episodes=2, headless=True,
                                  dr={"friction_range": [0.5, 1.5]})
    assert dr["task_overrides"]["dr"] == {"friction_range": [0.5, 1.5]}
    assert dr["task_overrides"]["headless"] is True and dr["ppo"]["total_steps"] == 5
    assert list(cand) == list(dr) == ["mode", "task", "task_overrides", "reward_code_file",
                                      "ppo", "run_dir", "eval_episodes", "snapshots"]


# ---------------------------------------------------------------------------
# Eureka search: layout, ranking, reflection wiring, driver equivalence
# ---------------------------------------------------------------------------

# iter_0: cand 0 runs at 0%, cand 1 is a rejected import.
# iter_1: cand 0 is the global winner (50%), cand 1 crashes in the worker.
SEARCH_TABLE = {("iter_0", "train_0"): (0.0, 0.3),
                ("iter_1", "train_0"): (0.5, 0.9),
                ("iter_1", "train_1"): None}
SEARCH_REPLIES = [GOOD_REPLY, BAD_IMPORT_REPLY, GOOD_REPLY, GOOD_REPLY]


def _search_worker():
    def train(key, run_dir):
        entry = SEARCH_TABLE[key]
        return None if entry is None else _train_results(run_dir, *entry)
    return FakeWorker(train)


def _run_search(tmp_path, monkeypatch, use_graph: bool):
    worker = _install(monkeypatch, _search_worker())
    llm = ScriptedClient(SEARCH_REPLIES)
    req = _request(tmp_path / ("graph" if use_graph else "imp"), use_graph=use_graph)
    skill = learn_skill(req, llm=llm)
    return skill, worker, llm, os.path.join(req.run_root, req.skill_name)


@pytest.mark.parametrize("use_graph", [False, True])
def test_search_layout_ranking_and_reflection(tmp_path, monkeypatch, use_graph):
    skill, worker, llm, root = _run_search(tmp_path, monkeypatch, use_graph)

    # Global best is iter_1/cand 0 (success beats iter_0's fitness-only run).
    assert isinstance(skill, LearnedSkill)
    assert skill.success_rate == 0.5
    assert skill.checkpoint == os.path.join(root, "iter_1", "train_0", "checkpoint_final.pt")
    assert skill.dr_config is None and len(skill.history) == 2

    # Run-dir layout: rejected candidate 1 of iter_0 was never written/trained.
    assert _tree(root) == {
        "result.json",
        "iter_0/prompt.txt", "iter_0/reflection.txt", "iter_0/candidate_0.py",
        "iter_0/candidate_1.py",           # extracted (then rejected by validator)
        "iter_0/train_0/spec.json", "iter_0/train_0/results.json",
        "iter_1/prompt.txt", "iter_1/reflection.txt",
        "iter_1/candidate_0.py", "iter_1/candidate_1.py",
        "iter_1/train_0/spec.json", "iter_1/train_0/results.json",
        "iter_1/train_1/spec.json",        # crashed: no results.json
    }
    assert [os.path.basename(s["run_dir"]) for s in worker.specs] == [
        "train_0", "train_0", "train_1"]

    # Reflection wiring: round 2's prompt embeds round 1's reflection.
    with open(os.path.join(root, "iter_0", "reflection.txt")) as f:
        refl0 = f.read()
    assert "disallowed import" in refl0 and "Best candidate was #0" in refl0
    assert "Feedback from the previous round" not in llm.calls[0]
    assert "Feedback from the previous round" in llm.calls[2] and refl0 in llm.calls[2]

    # Candidate errors are recorded, not raised.
    it1 = skill.history[1]
    assert it1.candidates[1].error is not None and "no results.json" in it1.candidates[1].error
    assert skill.history[0].candidates[1].error.startswith("disallowed import")

    with open(os.path.join(root, "result.json")) as f:
        result = json.load(f)
    assert set(result) == {"name", "checkpoint", "success_rate", "dr_config",
                           "dr_success_rate", "reward_code"}
    assert result["checkpoint"] == skill.checkpoint and result["dr_config"] is None


def test_imperative_and_graph_drivers_are_equivalent(tmp_path, monkeypatch):
    """The LangGraph nodes must call the same stage functions as the loop."""
    imp, w_imp, llm_imp, root_imp = _run_search(tmp_path, monkeypatch, False)
    gra, w_gra, llm_gra, root_gra = _run_search(tmp_path, monkeypatch, True)

    def strip(specs, root):
        return [json.loads(json.dumps(s).replace(root, "<root>")) for s in specs]

    assert imp.success_rate == gra.success_rate and imp.reward_code == gra.reward_code
    assert _tree(root_imp) == _tree(root_gra)
    assert llm_imp.calls == llm_gra.calls                      # identical prompts
    assert strip(w_imp.specs, root_imp) == strip(w_gra.specs, root_gra)
    for rel in ("iter_0/reflection.txt", "iter_1/reflection.txt", "iter_1/prompt.txt"):
        with open(os.path.join(root_imp, rel)) as a, open(os.path.join(root_gra, rel)) as b:
            assert a.read() == b.read(), rel
    with open(os.path.join(root_imp, "result.json")) as a, \
            open(os.path.join(root_gra, "result.json")) as b:
        assert a.read().replace(root_imp, "<root>") == b.read().replace(root_gra, "<root>")


@pytest.mark.parametrize("use_graph", [False, True])
def test_no_runnable_candidate_raises(tmp_path, monkeypatch, use_graph):
    _install(monkeypatch, FakeWorker(lambda key, run_dir: {"error": "Traceback\nKeyError"}))
    req = _request(tmp_path, use_graph=use_graph,
                   eureka=EurekaConfig(iterations=1, samples=1))
    with pytest.raises(RuntimeError, match="no runnable candidate"):
        learn_skill(req, llm=ScriptedClient([GOOD_REPLY]))
    with open(os.path.join(req.run_root, "getup", "iter_0", "reflection.txt")) as f:
        assert "All candidates failed" in f.read()


def test_unknown_task_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown task"):
        learn_skill(_request(tmp_path, task="nope"), llm=ScriptedClient(["-"]))


def test_no_code_reply_is_rejected_before_training(tmp_path, monkeypatch):
    worker = _install(monkeypatch, FakeWorker(lambda key, run_dir: _train_results(run_dir, 0.1)))
    req = _request(tmp_path, eureka=EurekaConfig(iterations=1, samples=2))
    skill = learn_skill(req, llm=ScriptedClient([NO_CODE_REPLY, GOOD_REPLY]))
    cands = skill.history[0].candidates
    assert cands[0].error == "no python code block defining compute_reward"
    assert cands[0].code == NO_CODE_REPLY                 # raw reply kept for the reflection
    assert len(worker.specs) == 1 and skill.success_rate == 0.1


# ---------------------------------------------------------------------------
# DrEureka stage
# ---------------------------------------------------------------------------

def _dr_cfg(**kw) -> DrEurekaConfig:
    base = dict(friction_values=[0.5, 1.5, 4.0], base_mass_values=[0.0, 2.0],
                com_shift_values=[0.0, 0.05], kp_scale_values=[1.0, 1.3],
                obs_noise_values=[0.0, 0.05], eval_episodes=2, samples=2,
                retrain_steps=77)
    base.update(kw)
    return DrEurekaConfig(**base)


def _feasible_friction_only(label: str) -> float:
    """nominal 0.6 → threshold 0.3: friction 0.5/1.5 feasible, everything else collapses."""
    return {"nominal": 0.6, "friction=0.5": 0.5, "friction=1.5": 0.35}.get(label, 0.05)


def test_sweep_grid_skips_nominal_values():
    labels = [s["label"] for s in dr_mod._sweeps(_dr_cfg())]
    assert labels == ["nominal", "friction=0.5", "friction=1.5", "friction=4.0",
                      "base_mass=2.0", "com_shift=0.05", "kp_scale=1.3", "obs_noise=0.05"]
    by_label = {s["label"]: s for s in dr_mod._sweeps(_dr_cfg())}
    assert by_label["com_shift=0.05"]["dr"] == {"com_shift_range": [-0.05, 0.05]}
    assert by_label["obs_noise=0.05"]["dr"] == {"obs_noise_std": 0.05}
    assert by_label["nominal"]["dr"] == {}


def test_physics_prior_reattaches_param_and_value(tmp_path, monkeypatch):
    worker = _install(monkeypatch, FakeWorker(lambda k, d: None,
                                              sweep_success=_feasible_friction_only))
    req = _request(tmp_path, dr=_dr_cfg(), eureka=EurekaConfig(n_envs=64))
    skill = LearnedSkill("getup", "go2_getup", "ckpt.pt", "code", 0.6)
    sweeps = dr_mod.physics_prior(req, skill, str(tmp_path / "prior"))
    spec = worker.specs[0]
    assert spec["mode"] == "dr_eval" and spec["checkpoint"] == "ckpt.pt"
    assert spec["task_overrides"]["n_envs"] == 32          # 64 // 8 floored at 32
    assert spec["eval_episodes"] == 2
    fr = next(s for s in sweeps if s["label"] == "friction=1.5")
    assert fr["param"] == "friction" and fr["value"] == 1.5 and fr["success_rate"] == 0.35
    assert next(s for s in sweeps if s["label"] == "nominal")["param"] is None


def test_physics_prior_failure_is_fatal(tmp_path, monkeypatch):
    def broken(cmd, **kw):
        with open(cmd[-1]) as f:
            spec = json.load(f)
        with open(os.path.join(spec["run_dir"], "results.json"), "w") as f:
            json.dump({"error": "boom"}, f)
        return types.SimpleNamespace(stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", broken)
    skill = LearnedSkill("getup", "go2_getup", "ckpt.pt", "code", 0.6)
    with pytest.raises(RuntimeError, match="physics prior failed"):
        dr_mod.physics_prior(_request(tmp_path), skill, str(tmp_path / "p"))


def test_dr_stage_clamps_proposals_and_keeps_best(tmp_path, monkeypatch):
    retrain = {"retrain_0": 0.3, "retrain_1": 0.6}

    def train(key, run_dir):
        if key[0] == "iter_0":
            return _train_results(run_dir, 0.6)
        return _train_results(run_dir, retrain[key[1]])
    worker = _install(monkeypatch, FakeWorker(train, sweep_success=_feasible_friction_only))

    class DrAwareLLM(ScriptedClient):
        def generate(self, prompt, temperature=1.0):
            self.calls.append(prompt)
            return DR_REPLY if "domain randomization" in prompt else GOOD_REPLY

    req = _request(tmp_path, run_dr=True, dr=_dr_cfg(),
                   eureka=EurekaConfig(iterations=1, samples=1, n_envs=8,
                                       train_steps=100, eval_episodes=3))
    llm = DrAwareLLM(["-"])
    skill = learn_skill(req, llm=llm)
    root = os.path.join(req.run_root, "getup")

    # Proposal clamped to RAPP bounds: friction [0.2,3.0]→[0.5,1.5]; obs noise
    # 0.5→0.0 (dropped by to_dict); kd_scale (unswept) passes through.
    assert skill.dr_config == {"friction_range": [0.5, 1.5], "kd_scale_range": [0.9, 1.1]}
    # Train-all-select-best: retrain_1 (60%) wins; Eureka success unchanged.
    assert skill.dr_success_rate == 0.6 and skill.success_rate == 0.6
    assert skill.checkpoint == os.path.join(root, "dr", "retrain_1", "checkpoint_final.pt")

    dr_specs = [s for s in worker.specs if "dr" in s["task_overrides"]]
    assert len(dr_specs) == 2
    for s in dr_specs:
        assert s["task_overrides"]["dr"] == skill.dr_config
        assert s["task_overrides"]["headless"] is True
        assert s["ppo"]["total_steps"] == 77 and s["eval_episodes"] == 2
        assert s["reward_code_file"].endswith("reward.py")

    assert {"dr/prior/spec.json", "dr/prior/results.json",
            "dr/retrain_0/reward.py", "dr/retrain_0/spec.json",
            "dr/retrain_1/reward.py", "dr/retrain_1/results.json"} <= _tree(root)
    with open(os.path.join(root, "dr", "retrain_0", "reward.py")) as f:
        assert f.read() == skill.reward_code
    with open(os.path.join(root, "result.json")) as f:
        result = json.load(f)
    assert result["dr_config"] == skill.dr_config and result["dr_success_rate"] == 0.6
    assert result["checkpoint"] == skill.checkpoint

    # The DR prompt (one per sample) carried the measured bounds and table.
    dr_prompts = [c for c in llm.calls if "domain randomization" in c]
    assert len(dr_prompts) == 2
    assert "friction (ratio): feasible [+0.5, +1.5]" in dr_prompts[0]
    assert "friction=4.0: success 5%  [COLLAPSED]" in dr_prompts[0]
    assert "nominal success 60%" in dr_prompts[0]


def test_dr_stage_keeps_policy_when_all_retrains_fail(tmp_path, monkeypatch):
    def train(key, run_dir):
        return _train_results(run_dir, 0.6) if key[0] == "iter_0" else {"error": "Trace\nOOM"}
    _install(monkeypatch, FakeWorker(train, sweep_success=_feasible_friction_only))
    req = _request(tmp_path, run_dr=True, dr=_dr_cfg(samples=1),
                   eureka=EurekaConfig(iterations=1, samples=1))
    skill = learn_skill(req, llm=ScriptedClient([GOOD_REPLY, DR_REPLY]))
    assert skill.dr_config is None and skill.dr_success_rate is None
    assert skill.checkpoint.endswith(os.path.join("iter_0", "train_0", "checkpoint_final.pt"))


def test_propose_dr_falls_back_to_bounds_when_llm_unusable(capsys):
    sweeps = [{"label": "nominal", "param": None, "value": 0.0, "success_rate": 0.6},
              {"label": "friction=0.5", "param": "friction", "value": 0.5, "success_rate": 0.5},
              {"label": "obs_noise=0.05", "param": "obs_noise", "value": 0.05,
               "success_rate": 0.4}]
    skill = LearnedSkill("getup", "go2_getup", "c", "code", 0.6)
    llm = ScriptedClient(["no json here", "```json\n{not json\n```"])
    configs, text = dr_mod.propose_dr_configs(llm, skill, sweeps, _dr_cfg(), n_samples=2)
    assert len(llm.calls) == 2 and "falling back" in capsys.readouterr().out
    assert len(configs) == 1
    assert configs[0].friction_range == (0.5, 1.0) and configs[0].obs_noise_std == 0.05
    assert configs[0].base_mass_range is None                   # no room → not randomised
    assert "friction (ratio): feasible [+0.5, +1]" in text

    # Nothing feasible at all: current behaviour still yields ONE no-op config
    # (obs_noise_std is always present in the bounds, at 0.0), so Stage 3
    # retrains once without randomisation rather than skipping. Pinned here
    # so a deliberate change shows up.
    configs, _ = dr_mod.propose_dr_configs(
        ScriptedClient(["-"]), skill, [sweeps[0]], _dr_cfg(), n_samples=1)
    assert len(configs) == 1 and configs[0].to_dict() == {}
