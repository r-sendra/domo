"""
DrEureka: reward-aware physics prior + LLM-proposed domain randomization.

Given a policy trained by the Eureka stage:
  1. PHYSICS PRIOR — evaluate the frozen policy under single-parameter
     perturbations (friction, payload mass, PD-gain scale, obs noise) to
     measure where it keeps working. This grounds the LLM: it proposes
     ranges over a table of measured feasibility, not guesses.
  2. PROPOSAL — the LLM turns the prior into a DomainRandomization config
     (as wide as feasible, avoiding collapsed regions).
  3. RETRAIN — the best reward + the proposed DR train a robust policy,
     which is what would go to the real robot (M4).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from domo.llm.client import LLMClient, extract_json_block
from domo.robot.randomization import DomainRandomization

from . import prompts
from .routine import run_worker
from .spec import LearnedSkill, SkillLearningRequest

__all__ = ["physics_prior", "propose_dr", "run_dr_eureka"]


def _sweeps(dr_cfg) -> List[dict]:
    sweeps = [{"label": "nominal", "dr": {}}]
    for v in dr_cfg.friction_values:
        sweeps.append({"label": f"friction={v}",
                       "dr": {"friction_range": [v, v]}})
    for v in dr_cfg.base_mass_values:
        if v != 0.0:
            sweeps.append({"label": f"base_mass={v:+}kg",
                           "dr": {"base_mass_range": [v, v]}})
    for v in dr_cfg.kp_scale_values:
        if v != 1.0:
            sweeps.append({"label": f"kp_scale={v}",
                           "dr": {"kp_scale_range": [v, v]}})
    for v in dr_cfg.obs_noise_values:
        if v != 0.0:
            sweeps.append({"label": f"obs_noise={v}",
                           "dr": {"obs_noise_std": v}})
    return sweeps


def physics_prior(request: SkillLearningRequest, skill: LearnedSkill,
                  run_dir: str) -> List[dict]:
    cfg = request.eureka
    spec = {
        "mode": "dr_eval",
        "task": request.task,
        "task_overrides": {"n_envs": max(cfg.n_envs // 8, 32),
                           "device": cfg.device, "headless": True},
        "checkpoint": skill.checkpoint,
        "sweeps": _sweeps(request.dr),
        "eval_episodes": request.dr.eval_episodes,
        "run_dir": run_dir,
    }
    results = run_worker(spec, cfg.worker_timeout_s)
    if results.get("error"):
        raise RuntimeError(f"physics prior failed: {results['error']}")
    return results["sweeps"]


def prior_table(sweeps: List[dict], dr_cfg, nominal: float) -> str:
    threshold = max(dr_cfg.feasible_floor, dr_cfg.feasible_ratio * nominal)
    lines = [f"(feasibility threshold: success ≥ {threshold:.0%})"]
    for s in sweeps:
        if s["label"] == "nominal":
            continue
        mark = "FEASIBLE " if s["success_rate"] >= threshold else "COLLAPSED"
        lines.append(f"  {s['label']:>18s}: success {s['success_rate']:.0%}  [{mark}]")
    return "\n".join(lines)


def propose_dr(llm: LLMClient, skill: LearnedSkill, sweeps: List[dict],
               dr_cfg) -> Optional[DomainRandomization]:
    nominal = next((s["success_rate"] for s in sweeps
                    if s["label"] == "nominal"), skill.success_rate)
    prompt = prompts.dr_prompt(skill.name, nominal,
                               prior_table(sweeps, dr_cfg, nominal))
    response = llm.generate(prompt, temperature=0.3)
    raw = extract_json_block(response)
    if raw is None:
        print("  [dr] LLM returned no JSON config — skipping DR")
        return None
    try:
        return DomainRandomization.from_dict(json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"  [dr] unusable DR config ({e}) — skipping DR")
        return None


def run_dr_eureka(request: SkillLearningRequest, skill: LearnedSkill,
                  llm: LLMClient) -> LearnedSkill:
    cfg = request.eureka
    root = os.path.join(request.run_root, request.skill_name, "dr")
    os.makedirs(root, exist_ok=True)
    print(f"\n{'=' * 60}\n  DrEUREKA — robustness stage for "
          f"'{request.skill_name}'\n{'=' * 60}")

    # 1. Reward-aware physics prior
    print("  measuring physics prior (single-parameter sweeps)...")
    sweeps = physics_prior(request, skill, os.path.join(root, "prior"))

    # 2. LLM proposes randomization ranges
    dr = propose_dr(llm, skill, sweeps, request.dr)
    if dr is None:
        return skill
    dr_dict = dr.to_dict()
    print(f"  proposed DR: {dr_dict}")

    # 3. Retrain with the best reward under DR
    reward_path = os.path.join(root, "reward.py")
    with open(reward_path, "w") as f:
        f.write(skill.reward_code)
    spec = {
        "mode": "train",
        "task": request.task,
        "task_overrides": {"n_envs": cfg.n_envs, "device": cfg.device,
                           "headless": True, "dr": dr_dict},
        "reward_code_file": reward_path,
        "ppo": {
            "total_steps": request.dr.retrain_steps,
            "rollout_steps": cfg.rollout_steps,
            "minibatch_size": max(cfg.n_envs * cfg.rollout_steps // 4, 64),
            "hidden_size": cfg.hidden_size,
            "guard_nonfinite": True,
            "lr_schedule": "linear",
        },
        "run_dir": os.path.join(root, "retrain"),
        "eval_episodes": request.dr.eval_episodes,
        "snapshots": cfg.snapshots,
    }
    print(f"  retraining under DR ({request.dr.retrain_steps:,} steps)...")
    results = run_worker(spec, cfg.worker_timeout_s)
    if results.get("error"):
        print(f"  [dr] retrain failed — keeping non-robust policy\n"
              f"       {results['error'][:300]}")
        return skill

    skill.dr_config = dr_dict
    skill.dr_success_rate = results["success_rate"]
    skill.checkpoint = results["checkpoint"]
    print(f"  robust policy: success {skill.dr_success_rate:.0%} under DR")
    return skill
