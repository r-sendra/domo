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
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from typing import List, Optional

from domo.llm.client import LLMClient, make_llm

from . import prompts
from .rewards import extract_reward_code, validate_reward_code
from .spec import (TASK_REGISTRY, CandidateResult, IterationResult,
                   LearnedSkill, SkillLearningRequest)

__all__ = ["learn_skill", "run_worker", "make_client"]


def make_client(request: SkillLearningRequest) -> LLMClient:
    """Build the LLM client named by request.llm (an explicit client wins)."""
    return make_llm(request.llm, **(request.llm_kwargs or {}))


def run_worker(spec: dict, timeout_s: float) -> dict:
    """Write spec, spawn `python -m domo.eureka.worker`, read results."""
    run_dir = spec["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    spec_path = os.path.join(run_dir, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "domo.eureka.worker", spec_path],
            timeout=timeout_s, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"error": f"worker timeout after {timeout_s:.0f}s"}
    results_path = os.path.join(run_dir, "results.json")
    if not os.path.exists(results_path):
        tail = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-2000:]
        return {"error": f"worker produced no results.json; output tail:\n{tail}"}
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------

def _sample_candidates(llm: LLMClient, prompt: str, k: int,
                       temperature: float, out_dir: str) -> List[CandidateResult]:
    candidates = []
    for i in range(k):
        response = llm.generate(prompt, temperature=temperature)
        code = extract_reward_code(response)
        if code is None:
            candidates.append(CandidateResult(
                index=i, code=response[:2000],
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

def train_candidate(request: SkillLearningRequest, cand: CandidateResult,
                    it_dir: str) -> CandidateResult:
    """Train one validated reward candidate in a worker; fill its result."""
    cfg = request.eureka
    run_dir = os.path.join(it_dir, f"train_{cand.index}")
    spec = {
        "mode": "train",
        "task": request.task,
        "task_overrides": {"n_envs": cfg.n_envs, "device": cfg.device,
                           "headless": cfg.headless},
        "reward_code_file": os.path.join(it_dir, f"candidate_{cand.index}.py"),
        "ppo": {
            "total_steps": cfg.train_steps,
            "rollout_steps": cfg.rollout_steps,
            "minibatch_size": max(cfg.n_envs * cfg.rollout_steps // 4, 64),
            "hidden_size": cfg.hidden_size,
            "guard_nonfinite": True,
            "lr_schedule": "linear",
        },
        "run_dir": run_dir,
        "eval_episodes": cfg.eval_episodes,
        "snapshots": cfg.snapshots,
    }
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


def run_iteration(request: SkillLearningRequest, task_spec, llm,
                  reflection: str, it_dir: str, it_index: int
                  ) -> IterationResult:
    """One Eureka round: prompt → sample → validate → train → reflect."""
    cfg = request.eureka
    os.makedirs(it_dir, exist_ok=True)
    prompt = prompts.reward_prompt(request, task_spec, reflection,
                                   safety=cfg.safety_reward)
    with open(os.path.join(it_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    candidates = _sample_candidates(llm, prompt, cfg.samples,
                                    cfg.temperature, it_dir)
    for cand in candidates:
        if not cand.ok:
            print(f"  iter {it_index + 1} cand {cand.index}: rejected "
                  f"({cand.error})")
            continue
        print(f"  iter {it_index + 1} cand {cand.index}: training "
              f"({cfg.train_steps:,} steps)...")
        train_candidate(request, cand, it_dir)
        if cand.ok:
            print(f"  iter {it_index + 1} cand {cand.index}: "
                  f"success {cand.success_rate:.0%} | fitness {cand.fitness:.2f} "
                  f"| ever-upright {cand.ever_upright_rate:.0%} "
                  f"| peak-h {cand.peak_height:.2f}m "
                  f"| hold {cand.max_hold:.0f}/{25}")
        else:
            print(f"  iter {it_index + 1} cand {cand.index}: FAILED")

    with open(os.path.join(it_dir, "reflection.txt"), "w") as f:
        f.write(prompts.reflection_block(candidates))
    return IterationResult(index=it_index, candidates=candidates)


def select_global_best(history: List[IterationResult]) -> Optional[CandidateResult]:
    best = None
    for it in history:
        b = it.best
        if b and (best is None or b.rank_key < best.rank_key):
            best = b
    return best


def finalize_skill(request: SkillLearningRequest, skill: LearnedSkill) -> LearnedSkill:
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
                llm: Optional[LLMClient] = None) -> LearnedSkill:
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
    task_spec = TASK_REGISTRY[request.task]
    cfg = request.eureka
    root = os.path.join(request.run_root, request.skill_name)
    os.makedirs(root, exist_ok=True)

    print(f"\n{'=' * 60}\n  EUREKA — learning '{request.skill_name}' "
          f"({cfg.iterations} iters × {cfg.samples} candidates)\n{'=' * 60}")

    history: List[IterationResult] = []
    reflection = ""
    for it in range(cfg.iterations):
        iteration = run_iteration(request, task_spec, llm, reflection,
                                  os.path.join(root, f"iter_{it}"), it)
        history.append(iteration)
        reflection = prompts.reflection_block(iteration.candidates)

    best = select_global_best(history)
    if best is None:
        raise RuntimeError(
            "Eureka produced no runnable candidate — see iter_*/reflection.txt")

    skill = LearnedSkill(
        name=request.skill_name, task=request.task,
        checkpoint=best.checkpoint, reward_code=best.code,
        success_rate=best.success_rate, history=history)

    if request.run_dr:
        from .dr import run_dr_eureka
        skill = run_dr_eureka(request, skill, llm)

    return finalize_skill(request, skill)
