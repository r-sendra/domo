"""
Prompt construction for Eureka (reward generation + reflection) and
DrEureka (domain-randomization proposal). Kept in one place so the prompt
surface is reviewable and versionable like any other interface.

Three prompts, one per LLM call site:

* ``reward_prompt``    — asks for a ``compute_reward(task)`` candidate. Built
  from the request description, the task's fixed success metric and env
  interface (``TaskSpec``), the optional safety block (DrEureka Stage 1) and,
  from the second round on, the previous round's ``reflection_block``.
* ``reflection_block`` — Eureka's "reward reflection": a text summary of the
  last round (ranking, diagnostics, best code + its component trajectories)
  that the next ``reward_prompt`` feeds back. Also written to
  ``iter_k/reflection.txt`` for humans.
* ``dr_prompt``        — DrEureka Stage 2: asks for a DR config inside the
  RAPP feasible bounds.

The wording is part of the method: change it deliberately, with a run.
"""

from __future__ import annotations

from .spec import CandidateResult, SkillLearningRequest, TaskSpec

__all__ = ["SAFETY_INSTRUCTION", "dr_prompt", "reflection_block", "reward_prompt"]


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

# DrEureka Stage 1: safety-regularized reward design. Appended to the task
# specification (l_task + l_safety) so the LLM shapes rewards that produce
# behaviour robust enough to transfer — without hand-tuned safety term scales.
SAFETY_INSTRUCTION = """\
This policy is intended for SIM-TO-REAL transfer to physical hardware, so the
learned behaviour must be safe and feasible on a real robot. Encourage:
- stability: keep the torso steady, penalise large roll/pitch and violent
  base motion;
- smoothness: penalise action rate, jerk, and rapid oscillation so motors are
  not slammed;
- feasibility: discourage extreme joint velocities/torques, self-collision,
  and postures near joint limits.
Fold these into the reward as shaping terms (do NOT hand-pick large fixed
penalty scales — keep them proportionate to the task terms). A jittery or
violent policy that maximises the metric in simulation will not transfer, so
smoothness genuinely matters."""


def reward_prompt(request: SkillLearningRequest, task_spec: TaskSpec,
                  reflection: str = "", safety: bool = True) -> str:
    """Assemble the reward-generation prompt for one Eureka round.

    Args:
        request: supplies the natural-language skill description.
        task_spec: supplies the fixed success metric and env interface.
        reflection: ``reflection_block`` of the previous round ("" on the
            first round → the prompt asks for a fresh reward instead of an
            improved one).
        safety: append ``SAFETY_INSTRUCTION`` (``EurekaConfig.safety_reward``).
    """
    task_block = f"\n## Skill to train\n{request.description}"
    if safety:
        task_block += "\n\n### Safety & transfer requirements\n" + SAFETY_INSTRUCTION
    parts = [
        _REWARD_SYSTEM,
        task_block,
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


# Thresholds for the "reaches but does not hold" diagnosis in
# `reflection_block`: the best candidate reached the goal pose in more than
# this fraction of episodes, yet its longest upright streak stayed below this
# many steps (the fixed metric needs ~25). Chosen from the ballistic-overshoot
# runs where ever-upright was ~28% with a 1-step hold.
_REACHED_RATE_MIN = 0.05
_HOLD_STEPS_MAX = 8

# Training-snapshot scalars echoed in the reflection, in display order.
_SNAPSHOT_SERIES = [
    ("fitness", "fitness (0-1)", "{:.2f}"),
    ("peak_height", "peak height (m)", "{:.2f}"),
    ("ever_upright_rate", "ever-reached-goal", "{:.0%}"),
    ("success_rate", "success rate", "{:.0%}"),
    ("mean_ep_len", "episode length", "{:.0f}"),
]


def _candidate_line(c: CandidateResult) -> str:
    """One-line outcome for a candidate (last traceback line if it failed)."""
    if c.ok:
        return (f"  #{c.index}: fitness {c.fitness:.2f} | success {c.success_rate:.0%} "
                f"| ever-reached-goal {c.ever_upright_rate:.0%} "
                f"| peak base height {c.peak_height:.2f} m")
    first = (c.error or "").strip().splitlines()
    return f"  #{c.index}: FAILED to run — {first[-1] if first else 'error'}"


def _zero_success_note(best: CandidateResult) -> str:
    """Targeted guidance when even the best candidate never succeeded.

    Distinguishes the two failure regimes from the diagnostics so the reward
    fix is targeted, not generic: ballistic overshoot (reaches the pose but
    cannot hold it) vs. plain lack of progress.
    """
    if best.ever_upright_rate > _REACHED_RATE_MIN and best.max_hold < _HOLD_STEPS_MAX:
        return (
            f"\nDIAGNOSIS: the robot REACHES the goal pose "
            f"({best.ever_upright_rate:.0%} of episodes) but does NOT HOLD it "
            f"(longest upright streak only {best.max_hold:.0f} steps, needs "
            f"~25). It is passing THROUGH the upright pose ballistically and "
            f"toppling. Fix: add a strong term rewarding STAYING upright at "
            f"LOW base linear AND angular velocity — i.e. settling to rest in "
            f"the standing stance — gated on being upright and at height. "
            f"Reward sustained low-velocity uprightness, not just reaching it. "
            f"A settled-standing bonus (upright × tall × still) weighted "
            f"heavily will convert reaches into holds.")
    return (
        "\nNOTE: no candidate reached the success threshold yet. Judge "
        "progress by fitness / peak height / ever-reached-goal, and "
        "reshape the reward so the robot makes MORE progress toward the "
        "upright, raised posture — stronger shaping of uprightness and "
        "base height, and terms that reward intermediate progress "
        "(pushing the base up, tucking legs under the body).")


def _snapshot_lines(snapshots: list[dict]) -> list[str]:
    """Component and metric trajectories of the best candidate's training."""
    lines = ["Its signals during training "
             "(mean per env-step at ~equal intervals):"]
    names = sorted(snapshots[-1].get("components", {}))
    for name in names:
        series = [f"{s['components'].get(name, 0.0):+.4f}" for s in snapshots]
        lines.append(f"  {name:>20s}: {' → '.join(series)}")
    for key, label, fmt in _SNAPSHOT_SERIES:
        series = [fmt.format(s.get(key, 0.0)) for s in snapshots]
        lines.append(f"  {label:>20s}: {' → '.join(series)}")
    return lines


def reflection_block(candidates: list[CandidateResult]) -> str:
    """
    Eureka reward reflection: outcomes + component trajectories. Candidates
    are compared on the DENSE fitness (0–1 progress toward the goal), not
    only the binary success — so the feedback carries a gradient even when no
    candidate has fully succeeded yet, plus diagnostics (did the robot ever
    reach the goal pose, how high did the base get).

    Returns the text that (a) becomes the "Feedback from the previous round"
    section of the next ``reward_prompt`` and (b) is saved as
    ``iter_k/reflection.txt``.
    """
    ok = [c for c in candidates if c.ok]
    best = min(ok, key=lambda c: c.rank_key) if ok else None
    lines = ["Candidate outcomes — ranked by dense fitness (0=no progress, "
             "1=goal), with the binary success metric alongside:"]
    lines.extend(_candidate_line(c) for c in candidates)
    if best is None:
        lines.append("\nAll candidates failed. Common causes: undefined "
                     "fields, wrong tensor shapes, python-level loops.")
        return "\n".join(lines)

    if best.success_rate == 0.0:
        lines.append(_zero_success_note(best))

    lines.append(f"\nBest candidate was #{best.index} (fitness {best.fitness:.2f}). "
                 f"Its code:")
    lines.append(f"```python\n{best.code}\n```")
    if best.snapshots:
        lines.extend(_snapshot_lines(best.snapshots))
    return "\n".join(lines)


_DR_SYSTEM = """\
You are configuring domain randomization (DR) to make a trained legged-robot
policy robust for sim-to-real transfer (DrEureka-style).

You are given a Reward-Aware Physics Prior (RAPP): for each physics
parameter, the FEASIBLE bounds — the widest range over which the trained
policy still succeeds when that parameter alone is perturbed. Also shown is
the per-value success under each perturbation.

Choose training randomization RANGES that are as WIDE AS POSSIBLE while
staying INSIDE the feasible bounds (training outside them wastes samples and
destabilises learning). Do not exceed the feasible min/max. It is fine to be
slightly narrower than the bounds for safety, but prefer wide ranges — wide
DR is what makes transfer work.

Output ONE json code block, exactly this schema (omit a key to leave that
parameter unrandomised):
```json
{
  "friction_range": [low, high],
  "base_mass_range": [low, high],
  "com_shift_range": [low, high],
  "kp_scale_range": [low, high],
  "kd_scale_range": [low, high],
  "obs_noise_std": value
}
```"""


def dr_prompt(skill_name: str, nominal_success: float,
              feasible_bounds: str, prior_table: str) -> str:
    """Assemble the DrEureka DR-proposal prompt.

    Args:
        skill_name: for the policy header only.
        nominal_success: success with no perturbation (RAPP reference).
        feasible_bounds: text block from ``dr.feasible_bounds``.
        prior_table: per-value success table from ``dr.prior_table``.
    """
    return (f"{_DR_SYSTEM}\n\n## Policy\n'{skill_name}', nominal success "
            f"{nominal_success:.0%} (no perturbation).\n\n"
            f"## RAPP feasible bounds (stay inside these)\n{feasible_bounds}\n\n"
            f"## Per-value success detail\n{prior_table}\n\n"
            f"Choose the randomization config now.")
