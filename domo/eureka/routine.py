"""
The Eureka skill-learning routine (M2 + M3): LLM reward generation →
parallel-in-principle candidate training → ranking on the fixed success
metric → reward reflection → iterate. Optionally followed by the DrEureka
robustness stage (see dr.py).

Entry point: `learn_skill(request, llm=None) → LearnedSkill`.

Deliberately a plain function over plain data: the digital twin's
PlanningController calls it when it concedes a skill gap; a CLI calls it
for offline experiments; the future real-robot supervisor calls it from a
workstation. In every case training runs in worker subprocesses (one
Genesis per process) and the caller's own world/process is untouched.

How one iteration flows (``run_iteration``)
-------------------------------------------
Everything for round ``k`` lives in ``<run_root>/<skill_name>/iter_k/``:

1. ``prompts.reward_prompt`` builds the request (description + fixed metric +
   env interface [+ safety block] [+ previous reflection]) → ``prompt.txt``.
2. ``_sample_candidates`` asks the LLM ``cfg.samples`` times; each reply is
   extracted and statically validated → ``candidate_i.py`` (rejected replies
   keep their error and are never trained).
3. ``train_candidate`` trains each runnable candidate in a fresh worker
   subprocess → ``train_i/{spec.json, results.json, checkpoint_final.pt}``
   and copies the worker's metrics into the ``CandidateResult``.
4. ``prompts.reflection_block`` summarises the round → ``reflection.txt`` and
   becomes the feedback section of the next round's prompt.

After ``cfg.iterations`` rounds, ``build_learned_skill`` picks the global best
by ``CandidateResult.rank_key`` (success first, dense fitness second), the
DrEureka stage optionally hardens it, and ``finalize_skill`` writes
``<run_root>/<skill_name>/result.json``.

Worker protocol (``run_worker``)
--------------------------------
The parent writes ``<run_dir>/spec.json``, spawns
``python -m domo.eureka.worker <spec.json>`` and reads ``<run_dir>/results.json``.
The worker ALWAYS writes results.json (with an ``"error"`` traceback on
failure); a missing file means the process died or timed out and is turned
into an ``{"error": ...}`` result here, so callers never raise on a bad
candidate — it simply ranks last and shows up in the reflection.

Two drivers share the stage functions in this module: the imperative loop
below and the LangGraph state machine in ``graph.py``. Keep them in lock
step by changing the shared functions, not the drivers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from domo.llm.client import LLMClient, make_llm

from . import prompts
from .rewards import extract_reward_code, validate_reward_code
from .spec import (
    TASK_REGISTRY,
    CandidateResult,
    IterationResult,
    LearnedSkill,
    SkillLearningRequest,
    TaskSpec,
)

__all__ = ["learn_skill", "make_client", "run_worker"]

# Files exchanged with the worker subprocess (see module docstring).
SPEC_FILENAME = "spec.json"
RESULTS_FILENAME = "results.json"

# How much of a failed worker's stdout/stderr to keep in the error message.
_OUTPUT_TAIL_CHARS = 2000

# Denominator shown in the per-candidate "hold a/b" console line. Mirrors
# Go2GetUpConfig.success_hold_steps (0.5 s at 50 Hz); display only.
_HOLD_STEPS_DISPLAY = 25

# Rejected LLM replies keep this much of the raw text for the reflection.
_REJECTED_REPLY_CHARS = 2000


def make_client(request: SkillLearningRequest) -> LLMClient:
    """Build the LLM client named by request.llm (an explicit client wins)."""
    return make_llm(request.llm, **(request.llm_kwargs or {}))


def run_worker(spec: dict, timeout_s: float) -> dict:
    """Write spec, spawn `python -m domo.eureka.worker`, read results.

    Args:
        spec: worker spec (see ``worker`` module docstring); ``spec["run_dir"]``
            receives ``spec.json`` and ``results.json``.
        timeout_s: wall-clock limit for the subprocess.

    Returns:
        The worker's results dict. Failures never raise: a timeout or a
        crashed worker return ``{"error": <reason>}`` so the caller treats
        the candidate as failed and moves on.
    """
    run_dir = spec["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    spec_path = os.path.join(run_dir, SPEC_FILENAME)
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    try:
        # check=False on purpose: the exit code is irrelevant, results.json
        # is the contract (the worker writes an "error" entry on failure).
        proc = subprocess.run(
            [sys.executable, "-m", "domo.eureka.worker", spec_path],
            timeout=timeout_s, capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired:
        return {"error": f"worker timeout after {timeout_s:.0f}s"}
    results_path = os.path.join(run_dir, RESULTS_FILENAME)
    if not os.path.exists(results_path):
        tail = ((proc.stdout or "")[-_OUTPUT_TAIL_CHARS:] + "\n"
                + (proc.stderr or "")[-_OUTPUT_TAIL_CHARS:])
        return {"error": f"worker produced no results.json; output tail:\n{tail}"}
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------

def _sample_candidates(llm: LLMClient, prompt: str, k: int,
                       temperature: float, out_dir: str) -> list[CandidateResult]:
    """Ask the LLM ``k`` times; extract + validate; save runnable code to disk.

    Every reply becomes a ``CandidateResult`` so the reflection can report
    rejections too. Only code that passed extraction is written to
    ``out_dir/candidate_i.py`` (the file the worker will load).
    """
    candidates = []
    for i in range(k):
        response = llm.generate(prompt, temperature=temperature)
        code = extract_reward_code(response)
        if code is None:
            candidates.append(CandidateResult(
                index=i, code=response[:_REJECTED_REPLY_CHARS],
                error="no python code block defining compute_reward"))
            continue
        err = validate_reward_code(code)
        candidate = CandidateResult(index=i, code=code, error=err)
        path = os.path.join(out_dir, f"candidate_{i}.py")
        with open(path, "w") as f:
            f.write(code)
        candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# Reusable stage functions (shared by the imperative path and the LangGraph)
# ---------------------------------------------------------------------------

def build_train_spec(request: SkillLearningRequest, *, reward_code_file: str,
                     run_dir: str, total_steps: int, eval_episodes: int,
                     headless: bool, dr: dict | None = None) -> dict:
    """Worker spec for one "train" job (shared by Eureka and DrEureka).

    The PPO block derives from ``request.eureka`` — the DR retraining stage
    reuses the exact same trainer settings so that robust and non-robust
    policies differ only by randomisation and budget. ``minibatch_size`` is a
    quarter of the rollout (floored at 64 for tiny smoke runs).

    Args:
        reward_code_file: path to the ``compute_reward`` source to inject.
        run_dir: where the worker writes spec/results/checkpoint.
        total_steps: env-steps to train (``train_steps`` or ``retrain_steps``).
        eval_episodes: deterministic episodes evaluated after training.
        headless: task override (Eureka honours the config; DR always True).
        dr: ``DomainRandomization.to_dict()`` for the DR stage, else None.
    """
    cfg = request.eureka
    task_overrides: dict = {"n_envs": cfg.n_envs, "device": cfg.device,
                            "headless": headless}
    if dr is not None:
        task_overrides["dr"] = dr
    return {
        "mode": "train",
        "task": request.task,
        "task_overrides": task_overrides,
        "reward_code_file": reward_code_file,
        "ppo": {
            "total_steps": total_steps,
            "rollout_steps": cfg.rollout_steps,
            "minibatch_size": max(cfg.n_envs * cfg.rollout_steps // 4, 64),
            "hidden_size": cfg.hidden_size,
            "guard_nonfinite": True,
            "lr_schedule": "linear",
        },
        "run_dir": run_dir,
        "eval_episodes": eval_episodes,
        "snapshots": cfg.snapshots,
    }


def train_candidate(request: SkillLearningRequest, cand: CandidateResult,
                    it_dir: str) -> CandidateResult:
    """Train one validated reward candidate in a worker; fill its result.

    On success copies the worker metrics (success/fitness/diagnostics,
    snapshots, checkpoint path) into ``cand``; on failure sets ``cand.error``
    (the traceback or timeout reason) so it ranks last. Returns ``cand``.
    """
    cfg = request.eureka
    spec = build_train_spec(
        request,
        reward_code_file=os.path.join(it_dir, f"candidate_{cand.index}.py"),
        run_dir=os.path.join(it_dir, f"train_{cand.index}"),
        total_steps=cfg.train_steps, eval_episodes=cfg.eval_episodes,
        headless=cfg.headless)
    results = run_worker(spec, cfg.worker_timeout_s)
    cand.error = results.get("error")
    if cand.ok:
        cand.success_rate = results["success_rate"]
        cand.fitness = results.get("fitness", results["success_rate"])
        cand.peak_height = results.get("peak_height", 0.0)
        cand.ever_upright_rate = results.get("ever_upright_rate", 0.0)
        cand.max_hold = results.get("max_hold", 0.0)
        cand.mean_ep_len = results["mean_ep_len"]
        cand.snapshots = results.get("snapshots", [])
        cand.checkpoint = results.get("checkpoint")
    return cand


def run_iteration(request: SkillLearningRequest, task_spec: TaskSpec,
                  llm: LLMClient, reflection: str, it_dir: str, it_index: int
                  ) -> IterationResult:
    """One Eureka round: prompt → sample → validate → train → reflect.

    Args:
        reflection: previous round's ``reflection_block`` ("" for round 0).
        it_dir: ``<run_root>/<skill>/iter_<it_index>``; created here.
    """
    cfg = request.eureka
    os.makedirs(it_dir, exist_ok=True)
    prompt = prompts.reward_prompt(request, task_spec, reflection,
                                   safety=cfg.safety_reward)
    with open(os.path.join(it_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    candidates = _sample_candidates(llm, prompt, cfg.samples,
                                    cfg.temperature, it_dir)
    for cand in candidates:
        tag = f"  iter {it_index + 1} cand {cand.index}:"
        if not cand.ok:
            print(f"{tag} rejected ({cand.error})")
            continue
        print(f"{tag} training ({cfg.train_steps:,} steps)...")
        train_candidate(request, cand, it_dir)
        if cand.ok:
            print(f"{tag} success {cand.success_rate:.0%} | fitness {cand.fitness:.2f} "
                  f"| ever-upright {cand.ever_upright_rate:.0%} "
                  f"| peak-h {cand.peak_height:.2f}m "
                  f"| hold {cand.max_hold:.0f}/{_HOLD_STEPS_DISPLAY}")
        else:
            print(f"{tag} FAILED")

    with open(os.path.join(it_dir, "reflection.txt"), "w") as f:
        f.write(prompts.reflection_block(candidates))
    return IterationResult(index=it_index, candidates=candidates)


def select_global_best(history: list[IterationResult]) -> CandidateResult | None:
    """Best runnable candidate across all rounds (by ``rank_key``), or None."""
    best = None
    for it in history:
        b = it.best
        if b and (best is None or b.rank_key < best.rank_key):
            best = b
    return best


def build_learned_skill(request: SkillLearningRequest,
                        history: list[IterationResult]) -> LearnedSkill:
    """Wrap the global best candidate as a ``LearnedSkill``.

    Raises:
        RuntimeError: if no candidate in any round ran successfully.
    """
    best = select_global_best(history)
    if best is None:
        raise RuntimeError(
            "Eureka produced no runnable candidate — see iter_*/reflection.txt")
    return LearnedSkill(
        name=request.skill_name, task=request.task,
        checkpoint=best.checkpoint, reward_code=best.code,
        success_rate=best.success_rate, history=history)


def finalize_skill(request: SkillLearningRequest, skill: LearnedSkill) -> LearnedSkill:
    """Persist ``<run_root>/<skill_name>/result.json`` and print the summary."""
    root = os.path.join(request.run_root, request.skill_name)
    with open(os.path.join(root, "result.json"), "w") as f:
        json.dump({"name": skill.name, "checkpoint": skill.checkpoint,
                   "success_rate": skill.success_rate,
                   "dr_config": skill.dr_config,
                   "dr_success_rate": skill.dr_success_rate,
                   "reward_code": skill.reward_code}, f, indent=2)
    print(f"\n{skill.summary()}\n")
    return skill


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_skill(request: SkillLearningRequest,
                llm: LLMClient | None = None) -> LearnedSkill:
    """Learn a skill end to end: Eureka search [+ DrEureka robustness].

    Args:
        request: what to learn and with what budget (see ``spec``).
        llm: an explicit client; when None one is built from ``request.llm``
            and ``request.llm_kwargs`` via ``domo.llm.make_llm``.

    Returns:
        The winning policy and its provenance; also written to
        ``<run_root>/<skill_name>/result.json``.

    Raises:
        ValueError: unknown ``request.task``.
        RuntimeError: no candidate ran successfully in any round.
    """
    if request.task not in TASK_REGISTRY:
        raise ValueError(f"unknown task '{request.task}' "
                         f"(registry: {sorted(TASK_REGISTRY)})")
    llm = llm or make_client(request)

    # Prefer the LangGraph orchestration when requested and available.
    if request.use_graph:
        try:
            from .graph import run_graph
        except ImportError:
            print("  [eureka] langgraph not installed — using imperative path")
        else:
            return run_graph(request, llm)

    return _learn_skill_imperative(request, llm)


def _learn_skill_imperative(request: SkillLearningRequest,
                            llm: LLMClient) -> LearnedSkill:
    """Plain-loop driver; mirrors the node sequence of ``graph.py`` exactly."""
    task_spec = TASK_REGISTRY[request.task]
    cfg = request.eureka
    root = os.path.join(request.run_root, request.skill_name)
    os.makedirs(root, exist_ok=True)

    print(f"\n{'=' * 60}\n  EUREKA — learning '{request.skill_name}' "
          f"({cfg.iterations} iters × {cfg.samples} candidates)\n{'=' * 60}")

    history: list[IterationResult] = []
    reflection = ""
    for it in range(cfg.iterations):
        iteration = run_iteration(request, task_spec, llm, reflection,
                                  os.path.join(root, f"iter_{it}"), it)
        history.append(iteration)
        reflection = prompts.reflection_block(iteration.candidates)

    skill = build_learned_skill(request, history)

    if request.run_dr:
        from .dr import run_dr_eureka
        skill = run_dr_eureka(request, skill, llm)

    return finalize_skill(request, skill)
