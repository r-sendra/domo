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

Run layout, under ``<run_root>/<skill_name>/dr/``:
    prior/{spec.json, results.json}          one "dr_eval" worker, all sweeps
    retrain_i/{reward.py, spec.json, results.json, checkpoint_final.pt}
                                             one "train" worker per DR config

Entry point: ``run_dr_eureka(request, skill, llm) → LearnedSkill`` (called by
both drivers; mutates and returns ``skill`` with ``dr_config``,
``dr_success_rate`` and the robust ``checkpoint`` when a DR policy trained).
"""

from __future__ import annotations

import json
import os

from domo.llm.client import LLMClient, extract_json_block
from domo.robot.randomization import DomainRandomization

from . import prompts
from .routine import build_train_spec, run_worker
from .spec import DrEurekaConfig, LearnedSkill, SkillLearningRequest

__all__ = [
    "feasible_bounds",
    "physics_prior",
    "propose_dr_configs",
    "run_dr_eureka",
]

# param key → (DR range field, default/nominal value, human label)
_PARAMS = {
    "friction":  ("friction_range",  1.0, "friction (ratio)"),
    "base_mass": ("base_mass_range",  0.0, "added mass (kg)"),
    "com_shift": ("com_shift_range",  0.0, "COM shift (m, per axis)"),
    "kp_scale":  ("kp_scale_range",   1.0, "Kp scale"),
    "obs_noise": ("obs_noise_std",    0.0, "obs noise (std)"),
}

# Label of the unperturbed sweep entry (the RAPP reference point).
_NOMINAL_LABEL = "nominal"

# The prior only needs an estimate, so it evaluates on a fraction of the
# training envs (never fewer than a small batch).
_PRIOR_ENV_DIVISOR = 8
_PRIOR_MIN_ENVS = 32

# Sampling temperature for DR proposals: below the reward stage's default so
# proposals stay well-formed JSON, above 0 so the m samples differ.
_DR_SAMPLE_TEMPERATURE = 0.8

# Tolerance below which a feasible interval counts as empty ("no room").
_EPS = 1e-9


def _sweeps(dr_cfg: DrEurekaConfig) -> list[dict]:
    """Single-parameter perturbations; each carries its param name + value.

    Mass/COM/Kp/noise skip their unperturbed value (the "nominal" entry
    already measures it); friction is swept at every listed value, 1.0
    included.
    """
    sweeps = [{"label": _NOMINAL_LABEL, "param": None, "value": 0.0, "dr": {}}]

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


def _feasible_threshold(dr_cfg: DrEurekaConfig, nominal: float) -> float:
    """Success rate a perturbed setting must keep to count as feasible."""
    return max(dr_cfg.feasible_floor, dr_cfg.feasible_ratio * nominal)


def _nominal_success(sweeps: list[dict], fallback: float) -> float:
    """Unperturbed success from the sweep results (fallback if absent)."""
    return next((s["success_rate"] for s in sweeps
                 if s["label"] == _NOMINAL_LABEL), fallback)


def physics_prior(request: SkillLearningRequest, skill: LearnedSkill,
                  run_dir: str) -> list[dict]:
    """Run all single-parameter sweeps in one worker; return per-value success.

    Each returned entry is ``{"label", "param", "value", "dr", "success_rate",
    "fitness", "mean_ep_len"}``.

    Raises:
        RuntimeError: if the evaluation worker failed (unlike candidate
            training, there is nothing to rank against, so this is fatal).
    """
    cfg = request.eureka
    spec = {
        "mode": "dr_eval",
        "task": request.task,
        "task_overrides": {"n_envs": max(cfg.n_envs // _PRIOR_ENV_DIVISOR,
                                         _PRIOR_MIN_ENVS),
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
    meta = {s["label"]: s for s in spec["sweeps"]}
    for r in results["sweeps"]:
        src = meta.get(r["label"], {})
        r.setdefault("param", src.get("param"))
        r.setdefault("value", src.get("value", 0.0))
    return results["sweeps"]


def feasible_bounds(sweeps: list[dict], dr_cfg: DrEurekaConfig, nominal: float
                    ) -> tuple[dict[str, tuple[float, float]], str]:
    """
    RAPP: per parameter, the [min, max] over which the policy stays feasible
    (success ≥ threshold), always including the nominal/default value.
    Returns (bounds dict keyed by DR field, human-readable block).
    """
    threshold = _feasible_threshold(dr_cfg, nominal)
    bounds: dict[str, tuple[float, float]] = {}
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
        width = "no room" if hi <= lo + _EPS else "randomisable"
        lines.append(f"  {label:>22s}: feasible [{lo:+.3g}, {hi:+.3g}]  ({width})")
    return bounds, "\n".join(lines)


def prior_table(sweeps: list[dict], dr_cfg: DrEurekaConfig, nominal: float) -> str:
    """Per-value success table (FEASIBLE / COLLAPSED) shown to the LLM."""
    threshold = _feasible_threshold(dr_cfg, nominal)
    lines = []
    for s in sweeps:
        if s["label"] == _NOMINAL_LABEL:
            continue
        mark = "FEASIBLE " if s["success_rate"] >= threshold else "COLLAPSED"
        lines.append(f"  {s['label']:>18s}: success {s['success_rate']:.0%}  [{mark}]")
    return "\n".join(lines)


def _clamp_to_bounds(dr_dict: dict, bounds: dict[str, tuple[float, float]]) -> dict:
    """Keep the LLM honest: no proposed range may exceed the feasible bounds.

    Ranges are clipped inward; ``obs_noise_std`` (a scalar) is capped at its
    feasible maximum; keys without a bound (e.g. ``kd_scale_range``, which
    RAPP does not sweep) pass through untouched.
    """
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


def propose_dr_configs(llm: LLMClient, skill: LearnedSkill, sweeps: list[dict],
                       dr_cfg: DrEurekaConfig, n_samples: int
                       ) -> tuple[list[DomainRandomization], str]:
    """
    DrEureka Stage 2: sample `n_samples` INDEPENDENT DR configs from the LLM,
    each clamped to the RAPP feasible bounds. Falls back to the feasible
    bounds themselves if the LLM yields nothing usable.

    Returns:
        (configs, feasible-bounds text). Replies without a parseable JSON
        object are skipped silently; ``configs`` may therefore be shorter
        than ``n_samples`` (or empty when even the fallback has no room).
    """
    nominal = _nominal_success(sweeps, skill.success_rate)
    bounds, bounds_text = feasible_bounds(sweeps, dr_cfg, nominal)
    prompt = prompts.dr_prompt(skill.name, nominal, bounds_text,
                               prior_table(sweeps, dr_cfg, nominal))

    configs: list[DomainRandomization] = []
    for _ in range(n_samples):
        response = llm.generate(prompt, temperature=_DR_SAMPLE_TEMPERATURE)
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
                    if isinstance(v, tuple) and v[1] > v[0] + _EPS}
        if "obs_noise_std" in bounds:
            fallback["obs_noise_std"] = bounds["obs_noise_std"][1]
        if fallback:
            print("  [dr] LLM produced no usable config — falling back to "
                  "RAPP feasible bounds")
            configs.append(DomainRandomization.from_dict(fallback))
    return configs, bounds_text


def _train_dr(request: SkillLearningRequest, skill: LearnedSkill,
              dr_dict: dict, run_dir: str) -> dict:
    """Retrain the winning reward under one DR config (Stage 3, one sample)."""
    reward_path = os.path.join(run_dir, "reward.py")
    os.makedirs(run_dir, exist_ok=True)
    with open(reward_path, "w") as f:
        f.write(skill.reward_code)
    spec = build_train_spec(
        request, reward_code_file=reward_path, run_dir=run_dir,
        total_steps=request.dr.retrain_steps,
        eval_episodes=request.dr.eval_episodes,
        headless=True, dr=dr_dict)
    return run_worker(spec, request.eureka.worker_timeout_s)


def run_dr_eureka(request: SkillLearningRequest, skill: LearnedSkill,
                  llm: LLMClient) -> LearnedSkill:
    """Stages 1–3 on the Eureka winner; returns ``skill`` (mutated in place).

    If no DR config can be proposed or every DR training fails, the
    non-robust skill is returned unchanged (with a console note) rather than
    raising — a fragile policy is still a result.
    """
    root = os.path.join(request.run_root, request.skill_name, "dr")
    os.makedirs(root, exist_ok=True)
    print(f"\n{'=' * 60}\n  DrEUREKA — robustness stage for "
          f"'{request.skill_name}'\n{'=' * 60}")

    # 1. Reward-aware physics prior → feasible bounds
    print("  [1/3] measuring reward-aware physics prior...")
    sweeps = physics_prior(request, skill, os.path.join(root, "prior"))
    nominal = _nominal_success(sweeps, skill.success_rate)
    _, bounds_text = feasible_bounds(sweeps, request.dr, nominal)
    print("  feasible bounds:\n" + "\n".join("    " + line
          for line in bounds_text.splitlines()))

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
