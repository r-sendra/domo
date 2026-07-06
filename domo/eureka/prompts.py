"""
Prompt construction for Eureka (reward generation + reflection) and
DrEureka (domain-randomization proposal). Kept in one place so the prompt
surface is reviewable and versionable like any other interface.
"""

from __future__ import annotations

from typing import List

from .spec import CandidateResult, SkillLearningRequest, TaskSpec

__all__ = ["reward_prompt", "reflection_block", "dr_prompt"]


_REWARD_SYSTEM = """\
You are a reward engineer for legged-robot reinforcement learning (PPO,
massively parallel simulation). Write a dense, well-shaped reward function
that trains the requested behaviour.

Rules:
- Output ONE python code block and nothing else.
- The block must define exactly:
      def compute_reward(task) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
  returning (total_reward [N], components) where `components` maps a short
  name to each term's per-env contribution (already scaled). The components
  are logged for your own future feedback — name them meaningfully.
- Use only `torch` and `math` (already imported) and the documented `task`
  fields. No file/network access, no imports, no loops over envs.
- All operations must be batched over N envs and return float tensors.
- Prefer smooth shaping (exp / tanh of errors) over sparse indicators;
  penalise energy/action-rate lightly so the motion stays feasible.
- Policy success is measured by a FIXED external metric you cannot change,
  described below. Your reward is good iff that metric improves."""


def reward_prompt(request: SkillLearningRequest, task_spec: TaskSpec,
                  reflection: str = "") -> str:
    parts = [
        _REWARD_SYSTEM,
        f"\n## Skill to train\n{request.description}",
        f"\n## Fixed success metric\n{task_spec.success_description}",
        f"\n## Environment interface\n{task_spec.env_interface}",
    ]
    if reflection:
        parts.append("\n## Feedback from the previous round\n" + reflection)
        parts.append(
            "\nWrite an IMPROVED reward function. Change what the feedback "
            "shows to be broken: rescale terms that dominate or vanish, "
            "replace components that stayed flat (they carry no learning "
            "signal), keep components that correlated with success.")
    else:
        parts.append("\nWrite the reward function now.")
    return "\n".join(parts)


def reflection_block(candidates: List[CandidateResult]) -> str:
    """Eureka-style reward reflection: outcomes + component trajectories."""
    ok = [c for c in candidates if c.ok]
    best = max(ok, key=lambda c: c.success_rate) if ok else None
    lines = ["Candidate outcomes (fixed success metric):"]
    for c in candidates:
        if c.ok:
            lines.append(f"  #{c.index}: success {c.success_rate:.0%}, "
                         f"mean episode length {c.mean_ep_len:.0f} steps")
        else:
            first = (c.error or "").strip().splitlines()
            lines.append(f"  #{c.index}: FAILED to run — {first[-1] if first else 'error'}")
    if best is None:
        lines.append("\nAll candidates failed. Common causes: undefined "
                     "fields, wrong tensor shapes, python-level loops.")
        return "\n".join(lines)

    lines.append(f"\nBest candidate was #{best.index}. Its code:")
    lines.append(f"```python\n{best.code}\n```")
    if best.snapshots:
        lines.append("Its reward components during training "
                     "(mean per env-step at ~equal intervals):")
        names = sorted(best.snapshots[-1].get("components", {}))
        for name in names:
            series = [f"{s['components'].get(name, 0.0):+.4f}"
                      for s in best.snapshots]
            lines.append(f"  {name:>20s}: {' → '.join(series)}")
        succ = [f"{s.get('success_rate', 0.0):.0%}" for s in best.snapshots]
        lens = [f"{s.get('mean_ep_len', 0.0):.0f}" for s in best.snapshots]
        lines.append(f"  {'success rate':>20s}: {' → '.join(succ)}")
        lines.append(f"  {'episode length':>20s}: {' → '.join(lens)}")
    return "\n".join(lines)


_DR_SYSTEM = """\
You are configuring domain randomization to make a trained legged-robot
policy robust for sim-to-real transfer (DrEureka-style). Below is the
policy's measured success under single-parameter perturbations — its
reward-aware physics prior. Choose training randomization RANGES that are
as wide as possible while staying inside regions where the policy still
substantially works (avoid ranges where success collapsed: training there
wastes samples and destabilises learning).

Output ONE json code block, exactly this schema (omit keys to leave a
parameter unrandomised):
```json
{
  "friction_range": [low, high],
  "base_mass_range": [low, high],
  "kp_scale_range": [low, high],
  "obs_noise_std": value
}
```"""


def dr_prompt(skill_name: str, nominal_success: float,
              prior_table: str) -> str:
    return (f"{_DR_SYSTEM}\n\n## Policy\n'{skill_name}', nominal success "
            f"{nominal_success:.0%} (no perturbation).\n\n"
            f"## Measured physics prior\n{prior_table}\n\n"
            f"Choose the randomization config now.")
