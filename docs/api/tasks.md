# `domo.tasks` — vectorised, reward-bearing environments

A task is a temporary lens over the simulation: it builds a scene and a
robot with the same builders the goal-free `World` uses, adds rewards,
resets, termination and (optionally) a fixed success metric, and steps
thousands of environments in lockstep behind a gym-like tensor API. It is
the only thing `domo.rl` and `domo.eureka` ever see; it never imports a
physics engine (only `domo.sim` interfaces through `domo.robot` and
`domo.world`).

```
domo.sim (engine handles) → domo.robot (sensors/actuators/state)
    → domo.tasks (THIS: VecTask environments)
        → domo.rl (PPO) / domo.eureka (reward injection) / examples
```

Every task follows the **legged-gym order** inside `step`: apply the action,
step physics, refresh the state, decide which envs are finished, reset
those envs *before* computing rewards and observations, then return. A
finished env is therefore scored on its fresh spawn state and the policy's
next observation already belongs to the new episode. The trainer sees the
reset through `reset_flags` and the truncation-vs-failure distinction
through `extras["time_outs"]`.

Four tasks exist today. Two of them build their scene directly with
`domo.sim.create_engine`; the other two sit on top of a `domo.world.World`
so the twin can be spawned in the exact environment a checkpoint was
trained in.

| Task | Learns | Obs / act | Scene | Entry point |
|------|--------|-----------|-------|-------------|
| `Go2WalkTask` | velocity tracking, joint-position actions | 45 / 12 | `create_engine`, flat or rough terrain | `main.py` |
| `Go2CPGWalkTask` | velocity tracking, CPG-modulating actions (the gait everything else uses) | 76 / 12 | `create_engine`, flat ground | `examples/locomotion/go2_cpg_rl.py` |
| `Go2AvoidTask` | lidar velocity corrections over a frozen CPG policy | 36 / 3 | `World`, arena or ReplicaCAD house | `examples/avoidance/go2_cpg_rl_lidar.py`, `go2_cpg_rl_avoid_house.py` |
| `Go2GetUpTask` | recovery from a fallen pose; ships with **no** reward | 42 / 12 | `World`, flat ground | `examples/eureka/eureka_getup.py` (via `domo.eureka`) |

## Module map

| Module | Contents |
|--------|----------|
| `domo/tasks/base.py` | `VecTask` (ABC), `RewardFn`, `rand_uniform` |
| `domo/tasks/common.py` | `sample_velocity_commands`, `envs_due_for_resample`, `fall_termination` — pure tensor helpers shared by the Go2 tasks (not re-exported; import from `domo.tasks.common`) |
| `domo/tasks/go2_walk.py` | `Go2WalkConfig`, `Go2WalkTask` |
| `domo/tasks/go2_cpg_walk.py` | `Go2CPGWalkConfig`, `Go2CPGWalkTask`, `CPG_OBS_DIM = 76`, `CPG_ACT_DIM = 12`; re-exports `build_cpg_observation` and `CPG_OBS_SCALES` from `domo.control` |
| `domo/tasks/go2_avoid.py` | `Go2AvoidConfig`, `Go2AvoidTask`, `AVOID_OBS_DIM = 36`, `AVOID_ACT_DIM = 3`; re-exports `ObstacleArenaConfig` from `domo.scenes` |
| `domo/tasks/go2_getup.py` | `Go2GetUpConfig`, `Go2GetUpTask`, `GETUP_OBS_DIM = 42`, `GETUP_ACT_DIM = 12`, `build_getup_observation` |

`domo.tasks.__all__` lists every public name above; `tests/test_public_api.py`
imports each of them, so a new task must be re-exported from
`domo/tasks/__init__.py` to count as public.

## `VecTask`

```python
class VecTask(ABC):
    def __init__(self, n_envs: int, num_obs: int, num_actions: int,
                 device: torch.device, dt: float, max_episode_length: int)
```

The base class owns the shared buffers, the reward registry and the
episode bookkeeping; subclasses build the scene and robot in `__init__`
and implement `step`, `reset` and `reset_idx`.

### Buffers and attributes

| Attribute | Shape / type | Meaning |
|-----------|--------------|---------|
| `n_envs`, `num_envs` | int | parallel envs; `num_envs` is the alias external trainers (rsl_rl) expect |
| `num_obs`, `num_actions` | int | observation / action width |
| `device` | `torch.device` | where every buffer lives |
| `dt` | float | control period [s]; reward weights are multiplied by it at registration |
| `max_episode_length` | int | episode budget in control steps |
| `obs_buf` | `[N, num_obs]` float32 | last observation, already normalised for the policy |
| `rew_buf` | `[N]` float32 | sum of the scaled reward terms |
| `reset_buf` | `[N]` bool | True where the env was reset this step (initialised to all-True) |
| `episode_length_buf` | `[N]` int32 | steps since reset |
| `extras` | dict | returned by `step`; carries `"time_outs"` and `"episode"` (below) |
| `reward_scales` | `{name: float}` | weight per term, already × `dt` when registered so |
| `episode_sums` | `{name: [N]}` | running per-env sum of each scaled term |
| `reward_components` | `{name: [N]}` | last components dict of an injected reward; empty while the registry is in use |

### The step API

```python
obs, privileged_obs, reward, reset_flags, extras = task.step(actions)   # actions [N, num_actions]
obs, privileged_obs = task.reset()
task.reset_idx(envs_idx)                                                # int64 [K]; no-op when empty
```

`privileged_obs` is `None` for every current task (reserved for an
asymmetric critic; `get_privileged_observations()` returns `None`).
`get_observations()` returns `obs_buf`.

`extras` carries two keys:

* `extras["time_outs"]`: float `[N]`, 1.0 where the budget ran out
  (truncation, not failure). Written by `mark_time_outs()`.
* `extras["episode"]`: `{"rew_<term>": float}`, the mean over the envs that
  just finished of each term's accumulated sum, divided by the nominal
  episode length in seconds (reward per second, comparable across episode
  lengths). Refreshed by `log_episode_sums()` whenever envs finish.

### Reward registry

```python
def register_rewards(self, scales: dict[str, float], scale_by_dt: bool = True) -> None
def compute_rewards(self) -> torch.Tensor          # fills rew_buf, returns it
def set_reward_override(self, fn: RewardFn | None) -> None
```

`register_rewards` binds each `name` to the method `_reward_<name>()`
(`AttributeError` if missing), stores `scale × dt` and allocates the
episode sum. A zero weight still registers the term: it is logged and
contributes nothing. `compute_rewards` evaluates the terms in registration
order, multiplies by the stored scale, accumulates `episode_sums` and sums
into `rew_buf`.

`RewardFn = Callable[[VecTask], tuple[Tensor, dict[str, Tensor]]]`. With an
override installed, `compute_rewards` calls `fn(task)` and uses the returned
`reward [N]` verbatim (**no dt scaling**), stores the components in
`reward_components` and accumulates them in `episode_sums` under their own
names; the registered terms are untouched. `None` restores the registry.
This is the hook `domo.eureka` uses to inject an LLM-generated reward
([eureka.md](eureka.md)).

### Success metric and bookkeeping

```python
def compute_success(self) -> torch.Tensor          # bool [N]; NotImplementedError by default
def log_episode_sums(self, envs_idx: torch.Tensor, episode_length_s: float) -> None
def mark_time_outs(self) -> torch.Tensor           # bool [N]; also sets extras["time_outs"]
```

`compute_success` is deliberately separate from the reward so that
generated rewards are ranked against a signal they cannot game. The
locomotion tasks do not define it and cannot be Eureka targets.

`mark_time_outs` flags `episode_length_buf > max_episode_length` (strictly
greater: with a 1000-step budget the time-out fires on the 1001st step) and
returns the mask so the caller can OR it into `reset_buf`.

### Helpers

```python
def rand_uniform(lower: float, upper: float, shape, device) -> torch.Tensor   # U[lower, upper)
```

In `domo.tasks.common` (pure tensor functions, no task state):

```python
def sample_velocity_commands(commands, envs_idx, lin_vel_x_range, lin_vel_y_range,
                             ang_vel_range, device) -> None      # in place into commands [N, 3]
def envs_due_for_resample(episode_length_buf, resampling_time_s, dt) -> torch.Tensor   # [K]
def fall_termination(state, pitch_limit, roll_limit, height_limit=None) -> torch.Tensor  # bool [N]
```

`sample_velocity_commands` draws x, then y, then yaw with one `torch.rand`
each; the draw order is part of seeded reproducibility and is pinned by
`tests/test_tasks_common.py`. `envs_due_for_resample` returns the envs whose
step count is a multiple of `int(resampling_time_s / dt)` (step 0 included;
`reset_idx` resamples those itself). `fall_termination` tests
`|euler[:, 1]| > pitch_limit`, `|euler[:, 0]| > roll_limit` and, when given,
`base_pos[:, 2] < height_limit`; the angles are the intrinsic X-Y-Z Euler
features of `RobotState` ([conventions.md](../conventions.md)).

---

## `Go2WalkTask`

Velocity-tracking locomotion with direct joint-position actions, the
official Genesis six-term reward set. This is what `main.py` trains. It is
a separate task from the CPG variant on purpose: different action space,
different gains, different observation.

```python
class Go2WalkTask(VecTask):
    OBS_DIM = 45
    ACT_DIM = 12
    def __init__(self, cfg: Go2WalkConfig, spec: RobotSpec = GO2)
```

### `Go2WalkConfig`

Serialised with `dataclasses.asdict` into every checkpoint
(`extra["task_config"]`); field names and defaults are the checkpoint
contract.

| Field | Default | Meaning | Unit |
|-------|---------|---------|------|
| `n_envs` | `4096` | parallel envs | — |
| `dt` | `0.02` | control period (50 Hz) | s |
| `max_episode_steps` | `1000` | episode budget (20 s) | steps |
| `device` | `"cuda"` | torch / engine device | — |
| `headless` | `True` | no viewer | — |
| `engine` | `"genesis"` | backend name for `create_engine` | — |
| `terrain` | `"flat"` | `"flat"` or `"rough"` | — |
| `action_scale` | `0.25` | joint offset per unit action | rad |
| `clip_actions` | `100.0` | symmetric clip on raw actions | — |
| `kp`, `kd` | `20.0`, `0.5` | joint PD gains | N·m/rad, N·m·s/rad |
| `simulate_action_latency` | `True` | execute the previous action (the real Go2's ~1-step delay) | — |
| `resampling_time_s` | `4.0` | command resampling period | s |
| `lin_vel_x_range` | `(0.5, 0.5)` | forward command range (low, high) | m/s |
| `lin_vel_y_range` | `(0.0, 0.0)` | lateral command range | m/s |
| `ang_vel_range` | `(0.0, 0.0)` | yaw-rate command range | rad/s |
| `termination_pitch`, `termination_roll` | `1.0`, `1.0` | tilt limits | rad |
| `termination_height` | `0.20` | minimum base height, flat terrain only | m |
| `tracking_sigma` | `0.25` | width of the `exp(−err²/σ)` tracking terms | — |
| `base_height_target` | `0.34` | nominal standing height | m |
| `reward_scales` | see below | term → weight (× dt at registration) | — |
| `obs_scales` | `{"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}` | observation scaling; `lin_vel` is only used for the command | — |
| `rough_terrain` | `TerrainConfig()` | subterrain grid for `terrain="rough"` ([sim.md](sim.md#terrainconfig)) | — |
| `rough_spawn_height` | `0.5` | drop height above the tiles | m |

### Observation (45)

| Index | Content | Scale |
|-------|---------|-------|
| `[0:3]` | base angular velocity, body frame | × 0.25 |
| `[3:6]` | projected gravity, body frame (`[0, 0, −1]` upright) | 1 |
| `[6:9]` | command `(vx, vy, vyaw)` | × `(2, 2, 0.25)` |
| `[9:21]` | joint positions − default stance | × 1 |
| `[21:33]` | joint velocities | × 0.05 |
| `[33:45]` | current action (the one just applied, after clipping) | 1 |

### Action (12)

Joint-position offsets in `RobotSpec.joint_names` order:
`target = clip(a, ±clip_actions) × action_scale + default_dof_pos`. With
`simulate_action_latency` the target is computed from `last_actions` (the
previous step's action), while `obs[33:45]` shows the current one.

### Rewards

| Name | Weight | Formula | Intent |
|------|--------|---------|--------|
| `tracking_lin_vel` | `1.0` | `exp(−‖v_cmd,xy − v_xy‖² / σ)` | follow the planar command |
| `tracking_ang_vel` | `0.2` | `exp(−(ω_cmd − ω_z)² / σ)` | follow the yaw rate |
| `lin_vel_z` | `−1.0` | `v_z²` | no bouncing |
| `base_height` | `−50.0` | `(h − 0.34)²` | keep the nominal height |
| `action_rate` | `−0.005` | `‖a_t − a_{t−1}‖²` | smooth targets |
| `similar_to_default` | `−0.1` | `Σ |q − q_default|` | stay near the stance |

### Termination

`|pitch| > 1`, `|roll| > 1`, `h < 0.20` (flat terrain only: absolute height
is meaningless over tiles), or the budget. No success metric.

### Terrain

`terrain="rough"` calls `scene.add_terrain(cfg.rough_terrain)` and spawns
one env at the centre of each subterrain tile, cycling through the grid
when `n_envs` exceeds the tile count; the spawn height is
`rough_spawn_height`. Flat terrain uses the robot's default spawn.

### Build and step

```python
import torch
from domo.tasks import Go2WalkConfig, Go2WalkTask

task = Go2WalkTask(Go2WalkConfig(n_envs=4, device="cpu"))
obs, _ = task.reset()                                    # stale obs_buf, see limitations
obs, _, rew, reset, extras = task.step(torch.zeros(4, 12))
```

---

## `Go2CPGWalkTask`

Velocity tracking where the policy modulates per-leg oscillators
(Bellegarda & Ijspeert, *CPG-RL*, RA-L 2022). The oscillators and the
closed-form IK live in `domo.control` ([control.md](control.md#cpg)), so
the same stack drives the frozen policy at deployment (`CPGLocomotionSkill`)
and inside `Go2AvoidTask`. The 76-dim observation is the canonical
interface of every CPG checkpoint (`policies/walk.pt`, `runs/go2_cpg/*`).

```python
class Go2CPGWalkTask(VecTask):
    OBS_DIM = CPG_OBS_DIM   # 76
    ACT_DIM = CPG_ACT_DIM   # 12
    def __init__(self, cfg: Go2CPGWalkConfig, spec: RobotSpec = GO2)
```

Construction prints a banner with the env count, the foot-contact source
(`force` when the backend exposes contact forces, otherwise `phase-proxy`)
and the IK self-test round-trip error (`OK` below 1e-4 rad).

### `Go2CPGWalkConfig`

| Field | Default | Meaning | Unit |
|-------|---------|---------|------|
| `n_envs` | `4096` | parallel envs | — |
| `dt` | `0.02` | control period | s |
| `max_episode_steps` | `1000` | episode budget | steps |
| `device` | `"cuda"` | | — |
| `headless` | `True` | | — |
| `engine` | `"genesis"` | | — |
| `kp`, `kd` | `100.0`, `2.0` | paper PD gains | N·m/rad, N·m·s/rad |
| `base_init_pos` | `(0.0, 0.0, 0.35)` | spawn position | m |
| `cpg` | `CPGConfig()` | oscillator ranges and gait parameters ([control.md](control.md#cpgconfig)) | — |
| `resampling_time_s` | `4.0` | command resampling period | s |
| `lin_vel_x_range` | `(0.3, 3.0)` | forward command range | m/s |
| `lin_vel_y_range` | `(−0.5, 0.5)` | lateral command range | m/s |
| `ang_vel_range` | `(−1.0, 1.0)` | yaw-rate command range | rad/s |
| `termination_pitch`, `termination_roll` | `1.0`, `1.0` | tilt limits | rad |
| `termination_height` | `0.18` | minimum base height | m |
| `tracking_sigma` | `0.5` | tracking width | — |
| `reward_clamp` | `10.0` | symmetric clamp on the summed reward | — |
| `reward_scales` | see below | | — |

### Observation (76), `build_cpg_observation`

| Index | Content | Scale |
|-------|---------|-------|
| `[0:3]` | base linear velocity, body frame | × 2.0 |
| `[3:6]` | base angular velocity, body frame | × 0.25 |
| `[6:9]` | projected gravity | 1 |
| `[9:12]` | command `(vx, vy, vyaw)` | × `(2, 2, 0.25)` |
| `[12:24]` | joint positions − default | × 1 |
| `[24:36]` | joint velocities | × 0.05 |
| `[36:48]` | previous action (the one applied at the last step) | 1 |
| `[48:52]` | foot contacts `FR, FL, RR, RL` (force sensor, or oscillator stance phase when the asset has no foot links) | 0 / 1 |
| `[52:76]` | CPG state: `r`, `ṙ`, `cos θ`, `sin θ`, `cos φ`, `sin φ` for the four legs | 1 |

`build_cpg_observation(state, commands, commands_scale, default_dof_pos,
last_actions, foot_contacts, oscillators)` is defined in `domo.control.cpg`
and re-exported here; tasks and skills that drive a CPG policy share it
exactly.

### Action (12)

Raw `(μ ×4, ω ×4, ψ ×4)`. `CPGLegController.joint_targets(actions, dt)`
tanh-squashes them into the `CPGConfig` ranges, integrates the oscillators,
maps to foot targets and solves the IK; the task does no clipping.

### Rewards

| Name | Weight | Formula | Intent |
|------|--------|---------|--------|
| `tracking_lin_vel_x` | `0.75` | `exp(−(vx_cmd − vx)² / σ)` | forward speed |
| `tracking_lin_vel_y` | `0.75` | `exp(−(vy_cmd − vy)² / σ)` | lateral speed |
| `tracking_ang_vel` | `0.75` | `exp(−(ω_cmd − ω_z)² / σ)` | yaw rate |
| `lin_vel_z` | `−2.0` | `v_z²` | no bouncing |
| `ang_vel_xy` | `−0.05` | `ω_x² + ω_y²` | level base |
| `work` | `−0.001` | `|Σ τ_PD · Δq̇|` with `τ_PD = kp (q_target − q) − kd q̇` | actuator effort |

The summed reward is clamped to `±reward_clamp` after `compute_rewards`.

### Termination

`|pitch| > 1`, `|roll| > 1`, `h < 0.18`, or the budget. No success metric.

### Build and step

```python
import torch
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask

task = Go2CPGWalkTask(Go2CPGWalkConfig(n_envs=4, device="cpu"))
obs, _ = task.reset()                                    # fresh observation of the spawn state
obs, _, rew, reset, extras = task.step(torch.zeros(4, 12))
```

---

## `Go2AvoidTask`

Learned velocity corrections around a frozen CPG locomotion policy. Two
decoupled layers:

```
lidar (36 sectors) → avoidance policy → (Δvx, Δvy, Δvyaw)
base command + correction → FROZEN CPG policy (CPGLocomotionSkill) → joint targets
```

The locomotion policy is a plain callable injected at construction; the
task never loads checkpoints (that wiring is `domo.checkpoints` /
`domo.policies`). The task does not own the simulation either: it is a
reward-bearing lens over a `domo.world.World`, so
`world_from_avoid_config(cfg)` spawns the twin in the identical environment
([world-and-services.md](world-and-services.md#domocheckpoints)).

```python
class Go2AvoidTask(VecTask):
    OBS_DIM = AVOID_OBS_DIM   # 36
    ACT_DIM = AVOID_ACT_DIM   # 3
    def __init__(self, cfg: Go2AvoidConfig,
                 locomotion_policy: Callable[[torch.Tensor], torch.Tensor],   # obs [N, 76] → action [N, 12]
                 spec: RobotSpec = GO2)
```

After construction `task.world`, `task.scene`, `task.robot`, `task.lidar`
(a `SimulatedLidar`) and `task.arena` (arena mode only) are exposed.

### `Go2AvoidConfig`

| Field | Default | Meaning | Unit |
|-------|---------|---------|------|
| `n_envs` | `4096` | parallel envs | — |
| `dt` | `0.02` | control period | s |
| `max_episode_steps` | `1000` | episode budget | steps |
| `device` | `"cuda"` | | — |
| `headless` | `True` | | — |
| `engine` | `"genesis"` | | — |
| `scene_kind` | `"arena"` | `"arena"` (walls + random boxes) or `"replica"` (ReplicaCAD house) | — |
| `arena` | `ObstacleArenaConfig()` | arena geometry and obstacle count ([world-and-services.md](world-and-services.md#obstaclearenaconfig)) | — |
| `replica_scene_json` | `scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json` | house layout | path |
| `replica_asset_root` | `scripts/house_scene/data/replica_cad/` | house assets | path |
| `ground_height` | `0.0` | ground plane height | m |
| `base_init_pos` | `(0.0, 0.0, 0.42)` | spawn position | m |
| `base_init_yaw_deg` | `0.0` | spawn heading | deg |
| `kp`, `kd` | `100.0`, `2.0` | inner-loop PD gains | — |
| `cpg` | `CPGConfig()` | must match the frozen policy's training config | — |
| `lidar_model` | `generic_sector_lidar()` | device model; `hesai_xt16()` simulates the real sensor ([robot.md](robot.md#lidar-device-models)) | — |
| `obs_max_range` | `4.0` | policy-facing normalisation range, independent of the device's physical range | m |
| `d_collision`, `d_danger`, `d_caution`, `d_anticipate` | `0.25`, `0.60`, `0.90`, `1.40` | nested reward zones on the closest return | m |
| `delta_vx_max`, `delta_vy_max`, `delta_vyaw_max` | `0.8`, `0.5`, `1.5` | correction bounds | m/s, m/s, rad/s |
| `base_command` | `(0.6, 0.0, 0.0)` | `(vx, vy, vyaw)` the correction is added to | m/s, m/s, rad/s |
| `vx_clamp`, `vy_clamp`, `vyaw_clamp` | `(−1, 2)`, `(−0.5, 0.5)`, `(−1.5, 1.5)` | final command clamps | |
| `termination_pitch`, `termination_roll`, `termination_height` | `1.0`, `1.0`, `0.18` | fall limits | rad, rad, m |
| `reward_scales` | see below | | — |

### Observation (36)

`clamp(lidar.read() / obs_max_range, 0, 1)`: one normalised minimum
distance per sector, `1` meaning nothing within range. Sector 0 starts at
the robot's +x and sectors run counter-clockwise
([sim.md](sim.md#lidar-layout)). The scan is re-read after `reset_idx`, so
envs that just respawned observe max range.

### Action (3)

`t = tanh(a)`; `correction = (0.8 t₀, 0.5 t₁, 1.5 t₂)`;
`command = clamp(base_command + correction, clamps)`. The command is written
into `CPGLocomotionSkill.command` and the skill's `update(state, dt)`
produces the joint targets.

### Rewards

| Name | Weight | Formula | Intent |
|------|--------|---------|--------|
| `survival` | `1.0` | `1` | make early termination costly |
| `avoidance` | `5.0` | `−(anticipate + caution + danger + collision)` with, on the closest return `d`: anticipate `0.05 (1.4 − d)` for `0.9 ≤ d < 1.4`; caution `0.5 (0.9 − d)` for `0.6 ≤ d < 0.9`; danger `3 (0.6 − d)²` for `0.25 ≤ d < 0.6`; collision `2.0` for `d < 0.25` | keep clear |
| `smoothness` | `−0.05` | `‖correction_t − correction_{t−1}‖²` | no jitter |
| `command_tracking` | `3.0` | `clamp((d − 0.9) / 0.5, 0, 1) · exp(−2 ‖correction‖²)` | prefer the base command when clear; gated so it never fights a needed turn |

### Termination

Fall (tilt / height as above); in arena mode `|x|` or `|y|` beyond
`arena.termination_distance` (`half_size` plus a margin, an attribute of
the built `ObstacleArena`); closest lidar return below `d_collision` once
at least one scan exists (`lidar.has_scan`); or the budget. No success
metric.

### Resets

`reset_idx` resets the robot, the locomotion skill's oscillators and the
lidar cache, restores `_min_dist` to the device's max range and, in arena
mode, re-scatters the obstacles of those envs (`arena.randomise`).

### Build and step

```python
import torch
from domo.policies import load_stable_locomotion
from domo.tasks import Go2AvoidConfig, Go2AvoidTask

walk_fn, _ = load_stable_locomotion("cpu")               # or: lambda obs: torch.zeros(obs.shape[0], 12)
task = Go2AvoidTask(Go2AvoidConfig(n_envs=4, device="cpu"), walk_fn)
obs, _ = task.reset()                                    # [4, 36], all ones (no scan yet)
obs, _, rew, reset, extras = task.step(torch.zeros(4, 3))
```

---

## `Go2GetUpTask`

The reward-injection target for Eureka. The robot spawns fallen (random
side, random yaw, scrambled joints) and must right itself and hold a
standing pose. The task ships with **no reward**: `rew_buf` stays zero until
`set_reward_override` installs one. What it does fix, outside the generated
code's reach, is the success metric and a dense fitness proxy.

```python
class Go2GetUpTask(VecTask):
    OBS_DIM = GETUP_OBS_DIM   # 42
    ACT_DIM = GETUP_ACT_DIM   # 12
    def __init__(self, cfg: Go2GetUpConfig, spec: RobotSpec = GO2)
```

### `Go2GetUpConfig`

Eureka workers rebuild the task from `task_overrides` dicts keyed by these
names ([eureka.md](eureka.md#the-worker-protocol)).

| Field | Default | Meaning | Unit |
|-------|---------|---------|------|
| `n_envs` | `4096` | parallel envs | — |
| `dt` | `0.02` | control period | s |
| `max_episode_steps` | `400` | budget (8 s) | steps |
| `device` | `"cuda"` | | — |
| `headless` | `True` | | — |
| `engine` | `"genesis"` | | — |
| `action_scale` | `0.5` | joint offset per unit action (wider than walking, for the flip) | rad |
| `clip_actions` | `100.0` | symmetric clip | — |
| `kp`, `kd` | `100.0`, `2.0` | PD gains | — |
| `spawn_height` | `0.18` | spawn base height | m |
| `spawn_roll_range` | `(π/3, 5π/6)` | `|roll|` at spawn (60°–150°), sign drawn separately | rad |
| `spawn_joint_noise` | `0.3` | joints = default ± `0.3 · U(−1, 1)` | rad |
| `success_height` | `0.26` | base height threshold (standing is ~0.32) | m |
| `success_tilt` | `0.4` | `|roll|` and `|pitch|` threshold | rad |
| `success_hold_steps` | `25` | consecutive upright steps for episode success (0.5 s) | steps |
| `dr` | `None` | `DomainRandomization.to_dict()` or `None` ([robot.md](robot.md#domainrandomization)) | — |

### Observation (42), `build_getup_observation`

```python
def build_getup_observation(state: RobotState, default_dof_pos: torch.Tensor,
                            last_actions: torch.Tensor) -> torch.Tensor   # [N, 42]
```

| Index | Content | Scale |
|-------|---------|-------|
| `[0:3]` | base angular velocity, body frame | × 0.25 |
| `[3:6]` | projected gravity | 1 |
| `[6:18]` | joint positions − default | × 1 |
| `[18:30]` | joint velocities | × 0.05 |
| `[30:42]` | previous action | 1 |

With DR enabled and `obs_noise_std > 0`, zero-mean Gaussian noise is added
in `step` but not in `reset` (the first observation is clean). The same
builder is used by the deployed `LearnedJointSkill`, so a trained policy
sees the training layout at run time.

### Action (12)

`target = clip(a, ±clip_actions) × 0.5 + default_dof_pos`. No latency.

### Reward

None unless injected. `step` calls `compute_rewards()` only when an override
is installed and zeroes `rew_buf` otherwise, so a stray reward can never
leak into candidate training.

### Success and fitness

```python
def compute_success(self) -> torch.Tensor   # bool [N], instantaneous
def compute_fitness(self) -> torch.Tensor   # float [N] in [0, 1]
```

* `compute_success`: `h > 0.26 ∧ |roll| < 0.4 ∧ |pitch| < 0.4`, evaluated
  every step. Episode success is that condition held for
  `success_hold_steps` consecutive steps (`_hold` resets on any miss).
* `compute_fitness`: `clamp(−g_z, 0, 1) · clamp(h / 0.26, 0, 1)`, where
  `g_z` is the projected-gravity z-component (uprightness × height
  fraction). Eureka's dense tie-breaker.

At every episode end (success or time-out) one record per finished env is
appended to `task.episode_outcomes`:

```python
{"success":      bool,    # hold reached this episode
 "fitness":      float,   # TIME-AVERAGE of compute_fitness over the episode (dwell), not the peak
 "peak_height":  float,   # max base height [m]
 "ever_upright": bool,    # compute_success was ever True
 "max_hold":     int}     # longest consecutive-upright streak
```

`extras["success"]` is a float `[N]` (1.0 where the hold was reached this
step) alongside `extras["time_outs"]`.

### Termination

Success or budget only. There is no fall termination: the robot starts
fallen, and thrashing simply wastes the budget.

### Spawn

`reset_idx` first performs the robot's default reset, then overwrites the
pose: `|roll| ~ U(spawn_roll_range)` with a random sign, `yaw ~ U(−π, π)`,
quaternion `R = Rz(yaw) · Rx(roll)` written as `wxyz`, joints scrambled
around the default stance, velocities zeroed. The RNG draw order (roll,
side, yaw, joints) is part of seeded reproducibility. DR, when configured,
is applied to the reset envs afterwards.

### Build and step

```python
import torch
from domo.tasks import Go2GetUpConfig, Go2GetUpTask

task = Go2GetUpTask(Go2GetUpConfig(n_envs=4, device="cpu"))
task.set_reward_override(
    lambda t: (t.compute_fitness(), {"fitness": t.compute_fitness()}))
obs, _ = task.reset()
obs, _, rew, reset, extras = task.step(torch.zeros(4, 12))
print(task.compute_success(), extras["success"], task.episode_outcomes[-3:])
```

---

## Known limitations

* **`Go2WalkTask.reset()` returns the stale `obs_buf`** (legged-gym
  behaviour): the first observation after a full reset is whatever the last
  `step` produced (zeros on a fresh task). The CPG, avoid and get-up tasks
  return a fresh observation of the spawn state.
* **`Go2GetUpTask.episode_outcomes` grows without bound.** Workers slice the
  tail; a long evaluation loop should clear it periodically.
* **Rough terrain and the arena draw from numpy's global RNG** (Genesis'
  terrain generator and `ObstacleArena.randomise` use `np.random`), so
  `torch.manual_seed` alone does not make those runs reproducible; seed
  `np.random.seed` too.
* **Time-outs are terminal for the trainer.** `extras["time_outs"]` is
  produced but `domo.rl`'s GAE does not bootstrap on it
  ([rl.md](rl.md#known-limitations)).
* **Foot contacts are a phase proxy on the bundled Go2.** The URDF has no
  foot links, so `robot.contact_sensor` is `None` and `obs[48:52]` of the
  CPG observation is the oscillator stance mask
  ([troubleshooting.md](../troubleshooting.md#genesis)).
* **The locomotion tasks define no `compute_success`** and cannot be Eureka
  targets as they stand.
