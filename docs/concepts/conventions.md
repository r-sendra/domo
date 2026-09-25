# Conventions

These conventions hold everywhere in `domo/`, `examples/` and `tests/`.
Breaking one of them is a bug even when the code runs — most of them encode
a fact about the engine, the hardware or a trained checkpoint that the type
system cannot enforce for you.

## Geometry and state

Two of these cause real bugs, and both are silent: the code keeps running
and the numbers are wrong.

!!! danger "Quaternions are `wxyz`, scalar first"

    Every quaternion tensor is `[N, 4]` ordered `(w, x, y, z)`. This matches
    Genesis. Conversion helpers live in `domo/utils/rotations.py` — do not
    hand-roll them, and do not assume a library you import agrees. An
    `xyzw` quaternion read as `wxyz` is a plausible-looking rotation, which
    is exactly what makes it expensive: `GO2.base_init_quat` was `(0,0,0,1)`
    in the original scripts on the belief that Genesis was `xyzw`; in `wxyz`
    that is a 180° yaw flip.

!!! danger "Euler angles are intrinsic X-Y-Z, not roll-pitch-yaw"

    `quat_to_euler_xyz` returns the convention the original scripts got from
    Genesis' `quat_to_xyz(rpy=False)`. Cross terms differ in sign from
    aerospace roll-pitch-yaw. Treat `base_euler` as the observation feature
    it is, not as physical roll/pitch/yaw, unless a docstring says
    otherwise. Yaw is index 2 and is the one component you can rely on.

**Frames.** Quantities are world-frame unless the name says `body`, `local`
or `base`. Velocity commands to skills are body-frame `(vx, vy, vyaw)`.
`RobotState` marks the exceptions explicitly: `base_lin_vel_world` is world
frame, `base_lin_vel` and `base_ang_vel` are body frame.

**Units are SI**: metres, seconds, radians, newton-metres. A field is in
another unit only when its name says so.

| Field | Unit | Why it is exempt |
|-------|------|------------------|
| `WorldConfig.dt = 0.02` | s | SI, the rule |
| `NavGains.tol_pos = 0.2` | m | SI, the rule |
| `WorldConfig.base_init_yaw_deg` | degrees | name ends in `_deg` |
| `CPGConfig.omega_range_hz` | Hz | name says `_hz` |
| `ViewerConfig.camera_fov` | degrees | camera convention, documented on the field |

### Go2 joint order

`[FR, FL, RR, RL] × [hip, thigh, calf]`, twelve DOFs. Observation layouts
and every trained checkpoint depend on this order, so it is fixed in
`domo/robot/go2.py` and never sorted, filtered or re-derived from a URDF.

| Index | Joint | Default (rad) | Index | Joint | Default (rad) |
|-------|-------|---------------|-------|-------|---------------|
| 0 | `FR_hip` | 0.0 | 6 | `RR_hip` | 0.0 |
| 1 | `FR_thigh` | 0.8 | 7 | `RR_thigh` | 1.0 |
| 2 | `FR_calf` | −1.5 | 8 | `RR_calf` | −1.5 |
| 3 | `FL_hip` | 0.0 | 9 | `RL_hip` | 0.0 |
| 4 | `FL_thigh` | 0.8 | 10 | `RL_thigh` | 1.0 |
| 5 | `FL_calf` | −1.5 | 11 | `RL_calf` | −1.5 |

The stride-3 slices are idiomatic and appear throughout the CPG code:
`targets[:, 0::3]` are hips, `1::3` thighs, `2::3` calves. Legs are indexed
`[FR, FL, RR, RL]` in every per-leg tensor too, including `foot_contacts`
and the oscillator bank. Rear thighs sit slightly higher than front ones in
the nominal stance; that asymmetry is the spec, not a typo.

## Tensors

* Everything batched is a `torch.Tensor` of shape `[n_envs, ...]` living on
  the simulation device. There are no per-env Python loops on the hot path.
* Shapes are written in docstrings as `[N, 3]`, `[N, 12]`, `[N, n_beams, 3]`.
* `n_envs` is the number of parallel environments; the RL layer also exposes
  it as `num_envs` for compatibility with external trainers.
* `RobotState` fields are updated **in place** (`tensor[:] = ...`), so a
  consumer may keep a reference across steps — and must not assume it owns
  the memory.

## Layering

The full stack and the reasoning behind it are in
[architecture](architecture.md#the-layer-stack). The rules a reviewer
checks:

| Package | Rule |
|---------|------|
| `domo/sim/` | the only package that imports a physics engine |
| `domo/robot/` | the only classes that touch physics handles; real-robot drivers reimplement exactly these |
| `domo/control/`, `domo/skills/` | pure torch, engine-free, importable on any machine |
| `domo/rl/` | consumes only the `VecTask` API; never imports physics |
| `domo/tasks/` | may use `sim`, `robot`, `scenes`, `control`; never `rl` |
| `examples/` | imports the library; the library never imports an example |
| `scripts/` | frozen: never edited, never imported |

## Naming

The control hierarchy is the vocabulary most easily got wrong, because
DOMO's is inverted relative to most stacks — see
[control hierarchy](control-hierarchy.md).

| Level | Rate | Name | Never call it |
|-------|------|------|---------------|
| primitive | 50 Hz | `Skill` | a controller |
| orchestration | 1–10 Hz | `Controller` / `PlanningController` | a skill or a policy |
| metronome | sim step | `ControlLoop` | an environment |

A *policy* is the network inside a skill. A *program* is grammar text. A
*card* is a skill's description, never the skill itself. A *task* is a
`VecTask`, and only training has tasks.

## Code style

* Python 3.10+ syntax: `X | None`, `list[int]`,
  `from __future__ import annotations`.
* Every module has a docstring stating its purpose and its place in the
  architecture. Public classes and functions have Google-style docstrings
  with tensor shapes.
* Comments explain *why*: design decisions, units, engine quirks. Knowledge
  about Genesis behaviour lives next to the code that depends on it, and in
  [troubleshooting](../guides/troubleshooting.md).
* Configuration is a `dataclass` whose defaults reproduce the reference
  runs. Never change a default silently — it changes what the checkpoints
  mean.
* Never `except Exception: pass`. Narrow the exception, or log why it is
  tolerated.
* Skill programs are never CLI parameters.

## Checklist before you commit

- [ ] `pytest tests/` passes — the suite is engine-free and runs in seconds,
      so there is no excuse for skipping it.
- [ ] `ruff check domo examples tests main.py` is clean; the rule set is in
      `pyproject.toml`.
- [ ] No physics-engine import outside `domo/sim/`
      (`grep -rn "import genesis" domo/ | grep -v domo/sim` returns nothing).
- [ ] No new CLI flag that takes a skill program, a route or a mission —
      that logic belongs in `PlanningController.plan()`.
- [ ] Nothing added to or edited in `scripts/`.
- [ ] New batched tensors documented with their shape, and new config fields
      with their unit.
