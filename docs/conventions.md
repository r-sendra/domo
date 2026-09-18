# Conventions

These conventions hold everywhere in `domo/`, `examples/` and `tests/`.
Breaking one of them is a bug even when the code runs.

## Geometry and state

* **Quaternions are `wxyz`, scalar first.** Every quaternion tensor is
  `[N, 4]` ordered `(w, x, y, z)`. This matches Genesis. Conversion helpers
  live in `domo/utils/rotations.py`; do not hand-roll them.
* **Euler angles** produced by `quat_to_euler_xyz` are *intrinsic X-Y-Z*
  (the convention the original scripts used through Genesis'
  `quat_to_xyz(rpy=False)`), not aerospace roll-pitch-yaw. Cross terms differ
  in sign; treat them as the observation feature they are, not as physical
  roll/pitch/yaw, unless the docstring says otherwise. Yaw is index 2.
* **Frames**: quantities are world-frame unless the name says `body`,
  `local` or `base`. Velocity commands to skills are body-frame
  `(vx, vy, vyaw)`.
* **Go2 joint order** is `[FR, FL, RR, RL] × [hip, thigh, calf]`, twelve
  DOFs. Observation layouts and checkpoints depend on it.
* **Units** are SI: metres, seconds, radians, newton-metres. Angles in
  configs are radians unless the field name ends in `_deg`.

## Tensors

* Everything batched is a `torch.Tensor` of shape `[n_envs, ...]` living on
  the simulation device. There are no per-env Python loops on the hot path.
* Shapes are written in docstrings as `[N, 3]`, `[N, 12]`, `[N, n_beams, 3]`.
* `n_envs` is the number of parallel environments; the RL layer also sees it
  as `num_envs` for compatibility with external trainers.

## Layering

```
utils  ←  sim  ←  robot  ←  {control, scenes}  ←  tasks  ←  rl
                                      ↑
                        skills  ←  world  ←  examples
```

* `domo/sim/`: the only package that imports a physics engine.
* `domo/robot/`: the only classes that call physics handles (sensors and
  actuators). Real-robot drivers reimplement these same classes.
* `domo/control/` and `domo/skills/`: pure torch, engine-free, importable on
  any machine.
* `domo/rl/`: consumes only the `VecTask` API; never imports physics.
* `scripts/`: frozen, never imported.

## Control hierarchy naming

| Level | Rate | Name | Never call it |
|-------|------|------|---------------|
| primitive | 50 Hz | `Skill` | a controller |
| orchestration | 1–10 Hz | `Controller` / `PlanningController` | a skill or a policy |
| metronome | sim step | `ControlLoop` | an environment |

## Code style

* Python 3.10+ syntax: `X | None`, `list[int]`, `from __future__ import annotations`.
* Every module has a docstring stating its purpose and place in the
  architecture. Public classes and functions have Google-style docstrings
  with tensor shapes.
* Comments explain *why*: design decisions, units, engine quirks. Knowledge
  about Genesis behaviour lives next to the code that depends on it and in
  [troubleshooting.md](troubleshooting.md).
* Configuration is a `dataclass` with defaults that reproduce the reference
  runs. Never change a default silently; it changes what the checkpoints
  mean.
* Lint with `ruff check domo examples tests main.py`; the rule set is in
  `pyproject.toml`. Run `pytest tests/` before every commit.
* Never `except Exception: pass`. Narrow the exception or log why it is
  tolerated.
* Skill programs are never CLI parameters.
