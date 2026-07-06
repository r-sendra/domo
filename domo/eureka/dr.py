"""
DrEureka: reward-aware physics prior + LLM-proposed domain randomization.

Faithful to Yu et al. 2024 (arxiv 2406.01967). Given a policy trained by the
(safety-regularized) Eureka stage:

  1. REWARD-AWARE PHYSICS PRIOR (RAPP) — evaluate the frozen policy under
     single-parameter perturbations (friction, payload mass, COM shift, PD
     gains, obs noise). For each parameter derive the FEASIBLE BOUNDS: the
     widest [min, max] over which the policy still succeeds. This grounds the
     LLM in measured limits rather than guesses.
  2. LLM-GUIDED DR — the LLM zero-shot generates `samples` INDEPENDENT DR
     configurations that stay inside the feasible bounds.
  3. TRAIN-ALL-SELECT-BEST — every proposed config is trained (best Eureka
     reward reused) and the policy with the highest success is kept. This is
     the policy that would transfer to the real robot (M4).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from domo.llm.client import LLMClient, extract_json_block
from domo.robot.randomization import DomainRandomization

from . import prompts
from .routine import run_worker
from .spec import LearnedSkill, SkillLearningRequest

__all__ = ["physics_prior", "feasible_bounds", "propose_dr_configs",
           "run_dr_eureka"]

# param key → (DR range field, default/nominal value, human label)
_PARAMS = {
    "friction":  ("friction_range",  1.0, "friction (ratio)"),
    "base_mass": ("base_mass_range",  0.0, "added mass (kg)"),
    "com_shift": ("com_shift_range",  0.0, "COM shift (m, per axis)"),
    "kp_scale":  ("kp_scale_range",   1.0, "Kp scale"),
    "obs_noise": ("obs_noise_std",    0.0, "obs noise (std)"),
}


def _sweeps(dr_cfg) -> List[dict]:
    """Single-parameter perturbations; each carries its param name + value."""
    sweeps = [{"label": "nominal", "param": None, "value": 0.0, "dr": {}}]

    def add(param, value, dr):
        sweeps.append({"label": f"{param}={value}", "param": param,
                       "value": value, "dr": dr})

    for v in dr_cfg.friction_values:
        add("friction", v, {"friction_range": [v, v]})
    for v in dr_cfg.base_mass_values:
        if v != 0.0:
            add("base_mass", v, {"base_mass_range": [v, v]})
    for v in dr_cfg.com_shift_values:
        if v != 0.0:
            add("com_shift", v, {"com_shift_range": [-v, v]})
    for v in dr_cfg.kp_scale_values:
        if v != 1.0:
            add("kp_scale", v, {"kp_scale_range": [v, v]})
    for v in dr_cfg.obs_noise_values:
        if v != 0.0:
            add("obs_noise", v, {"obs_noise_std": v})
    return sweeps


def physics_prior(request: SkillLearningRequest, skill: LearnedSkill,
                  run_dir: str) -> List[dict]:
    """Run all single-parameter sweeps in one worker; return per-value success."""
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
    # worker echoes label/dr but not param/value; re-attach from our sweeps
    meta = {s["label"]: s for s in _sweeps(request.dr)}
    for r in results["sweeps"]:
        src = meta.get(r["label"], {})
        r.setdefault("param", src.get("param"))
        r.setdefault("value", src.get("value", 0.0))
    return results["sweeps"]


def feasible_bounds(sweeps: List[dict], dr_cfg, nominal: float
                    ) -> Tuple[Dict[str, Tuple[float, float]], str]:
    """
    RAPP: per parameter, the [min, max] over which the policy stays feasible
    (success ≥ threshold), always including the nominal/default value.
    Returns (bounds dict keyed by DR field, human-readable block).
    """
    threshold = max(dr_cfg.feasible_floor, dr_cfg.feasible_ratio * nominal)
    bounds: Dict[str, Tuple[float, float]] = {}
    lines = [f"(a value is feasible if success ≥ {threshold:.0%}; "
             f"nominal success {nominal:.0%})"]

    for param, (field_name, default, label) in _PARAMS.items():
        feasible_vals = [default]
        for s in sweeps:
            if s.get("param") == param and s["success_rate"] >= threshold:
                feasible_vals.append(s["value"])
        lo, hi = min(feasible_vals), max(feasible_vals)
        # COM shift is symmetric: feasible magnitude → symmetric range.
        if param == "com_shift":
            lo = -hi
        bounds[field_name] = (lo, hi)
        width = "no room" if hi <= lo + 1e-9 else "randomisable"
        lines.append(f"  {label:>22s}: feasible [{lo:+.3g}, {hi:+.3g}]  ({width})")
    return bounds, "\n".join(lines)


def prior_table(sweeps: List[dict], dr_cfg, nominal: float) -> str:
    threshold = max(dr_cfg.feasible_floor, dr_cfg.feasible_ratio * nominal)
    lines = []
    for s in sweeps:
        if s["label"] == "nominal":
            continue
        mark = "FEASIBLE " if s["success_rate"] >= threshold else "COLLAPSED"
        lines.append(f"  {s['label']:>18s}: success {s['success_rate']:.0%}  [{mark}]")
    return "\n".join(lines)


def _clamp_to_bounds(dr_dict: dict, bounds: Dict[str, Tuple[float, float]]) -> dict:
    """Keep the LLM honest: no proposed range may exceed the feasible bounds."""
    out = {}
    for k, v in dr_dict.items():
        if k in bounds and isinstance(v, (list, tuple)) and len(v) == 2:
            lo, hi = bounds[k]
            out[k] = [max(float(v[0]), lo), min(float(v[1]), hi)]
        elif k == "obs_noise_std" and "obs_noise_std" in bounds:
            out[k] = min(float(v), bounds["obs_noise_std"][1])
        else:
            out[k] = v
    return out


def propose_dr_configs(llm: LLMClient, skill: LearnedSkill, sweeps: List[dict],
                       dr_cfg, n_samples: int
                       ) -> Tuple[List[DomainRandomization], str]:
    """
    DrEureka Stage 2: sample `n_samples` INDEPENDENT DR configs from the LLM,
    each clamped to the RAPP feasible bounds. Falls back to the feasible
    bounds themselves if the LLM yields nothing usable.
    """
    nominal = next((s["success_rate"] for s in sweeps
                    if s["label"] == "nominal"), skill.success_rate)
    bounds, bounds_text = feasible_bounds(sweeps, dr_cfg, nominal)
    prompt = prompts.dr_prompt(skill.name, nominal, bounds_text,
                               prior_table(sweeps, dr_cfg, nominal))

    configs: List[DomainRandomization] = []
    for i in range(n_samples):
        response = llm.generate(prompt, temperature=0.8)   # diverse samples
        raw = extract_json_block(response)
        if raw is None:
            continue
        try:
            d = _clamp_to_bounds(json.loads(raw), bounds)
            configs.append(DomainRandomization.from_dict(d))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if not configs:
        # Fallback: the feasible bounds are themselves a valid wide DR config.
        fallback = {k: list(v) for k, v in bounds.items()
                    if isinstance(v, tuple) and v[1] > v[0] + 1e-9}
        if "obs_noise_std" in bounds:
            fallback["obs_noise_std"] = bounds["obs_noise_std"][1]
        if fallback:
            print("  [dr] LLM produced no usable config — falling back to "
                  "RAPP feasible bounds")
            configs.append(DomainRandomization.from_dict(fallback))
    return configs, bounds_text


def _train_dr(request, skill, dr_dict, run_dir) -> dict:
    cfg = request.eureka
    reward_path = os.path.join(run_dir, "reward.py")
    os.makedirs(run_dir, exist_ok=True)
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
        "run_dir": run_dir,
        "eval_episodes": request.dr.eval_episodes,
        "snapshots": cfg.snapshots,
    }
    return run_worker(spec, cfg.worker_timeout_s)


def run_dr_eureka(request: SkillLearningRequest, skill: LearnedSkill,
                  llm: LLMClient) -> LearnedSkill:
    cfg = request.eureka
    root = os.path.join(request.run_root, request.skill_name, "dr")
    os.makedirs(root, exist_ok=True)
    print(f"\n{'=' * 60}\n  DrEUREKA — robustness stage for "
          f"'{request.skill_name}'\n{'=' * 60}")

    # 1. Reward-aware physics prior → feasible bounds
    print("  [1/3] measuring reward-aware physics prior...")
    sweeps = physics_prior(request, skill, os.path.join(root, "prior"))
    nominal = next((s["success_rate"] for s in sweeps
                    if s["label"] == "nominal"), skill.success_rate)
    _, bounds_text = feasible_bounds(sweeps, request.dr, nominal)
    print("  feasible bounds:\n" + "\n".join("    " + l
          for l in bounds_text.splitlines()))

    # 2. LLM proposes `samples` independent DR configs
    print(f"  [2/3] sampling {request.dr.samples} DR configs from the LLM...")
    configs, _ = propose_dr_configs(llm, skill, sweeps, request.dr,
                                    request.dr.samples)
    if not configs:
        print("  [dr] no DR config available — keeping non-robust policy")
        return skill

    # 3. Train every config; keep the best.
    print(f"  [3/3] training {len(configs)} DR policies "
          f"({request.dr.retrain_steps:,} steps each)...")
    best = None
    for i, dr in enumerate(configs):
        dr_dict = dr.to_dict()
        print(f"    config {i}: {dr_dict}")
        results = _train_dr(request, skill, dr_dict,
                            os.path.join(root, f"retrain_{i}"))
        if results.get("error"):
            print(f"    config {i}: FAILED ({results['error'].splitlines()[-1][:80]})")
            continue
        rate = results["success_rate"]
        print(f"    config {i}: robust success {rate:.0%}")
        if best is None or rate > best[1]:
            best = (dr_dict, rate, results["checkpoint"])

    if best is None:
        print("  [dr] all DR trainings failed — keeping non-robust policy")
        return skill

    skill.dr_config, skill.dr_success_rate, skill.checkpoint = best
    print(f"  best robust policy: success {skill.dr_success_rate:.0%} "
          f"under DR {skill.dr_config}")
    return skill
