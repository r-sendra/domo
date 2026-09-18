# `domo.robot` — robot spec, state, sensors, actuators

`domo.robot` turns a physics handle into a robot. It sits above `domo.sim`
and below `domo.control` / `domo.scenes` (see
[architecture.md](../architecture.md)). A `RobotSpec` says what the robot
is, a `Robot` binds that spec to a scene, and sensors and actuators move
data between the physics handles and a `RobotState` of `[N, ...]` tensors.

The rule it enforces: **sensors and actuators are the only classes that
call physics handles.** Tasks, controllers and skills see `RobotState`
tensors and call `Robot.set_joint_targets` / `Robot.reset_idx`; nothing
above this package touches an `Articulation`. Real-robot drivers
reimplement the same sensor and actuator classes on top of DDS/ROS topics,
which is why the layers above cannot tell sim from hardware.

Conventions: quaternions `wxyz`; joint tensors follow `RobotSpec.joint_names`
(for Go2, `[FR, FL, RR, RL] × [hip, thigh, calf]`); `base_euler` is
intrinsic X-Y-Z, not aerospace roll-pitch-yaw
([conventions.md](../conventions.md)).

## Module map

| Module | Contents |
|--------|----------|
| `domo/robot/spec.py` | `RobotSpec`, `QuadrupedGeometry` (frozen dataclasses) |
| `domo/robot/go2.py` | `GO2`, `GO2_GEOMETRY` constants |
| `domo/robot/state.py` | `RobotState` |
| `domo/robot/robot.py` | `Robot` facade (bind, refresh, set_joint_targets, reset_idx) |
| `domo/robot/sensors.py` | `StateSensor`, `ExteroceptiveSensor`, `SimIMU`, `SimJointEncoders`, `SimBaseStateSensor`, `SimContactSensor`, `SectorLidar` |
| `domo/robot/actuators.py` | `Actuator`, `PDJointPositionActuator` |
| `domo/robot/lidar_models.py` | `LidarModelConfig`, `hesai_xt16`, `generic_sector_lidar`, `XT16_FULL_AZIMUTH`, `SimulatedLidar` |
| `domo/robot/randomization.py` | `DomainRandomization` (imported from the submodule; not re-exported by `domo.robot`) |

Everything except `DomainRandomization` is re-exported by `domo.robot`.

## Specs

### `QuadrupedGeometry`

Frozen dataclass: per-leg kinematics for a 3-DOF-per-leg quadruped, used by
the analytic IK/FK in `domo.control.kinematics`. Lengths in metres.

| Field | Default | Meaning |
|-------|---------|---------|
| `l_hip` | — | hip → thigh lateral offset |
| `l_thigh` | — | thigh link length |
| `l_calf` | — | calf link length |
| `side_sign` | `(-1.0, +1.0, -1.0, +1.0)` | lateral sign per leg in the order of the leg blocks in `joint_names` (+1 left, −1 right) |

### `RobotSpec`

Frozen dataclass: a static, engine-independent robot description. Per-run
overrides (gains, spawn pose) are passed to `Robot`, never written into the
spec.

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | — | short identifier (checkpoint metadata, logging) |
| `urdf_path` | — | asset path as understood by the backend (Genesis resolves relative paths against its bundled assets) |
| `joint_names` | — | actuated joints in canonical order; every DOF-indexed tensor follows it |
| `default_joint_angles` | — | nominal standing pose (rad) keyed by joint name |
| `foot_link_names` | `()` | links probed for force-based contact; may be empty or absent from the asset |
| `base_init_pos` | `(0.0, 0.0, 0.4)` | spawn position (m) |
| `base_init_quat` | `(1.0, 0.0, 0.0, 0.0)` | spawn orientation, `wxyz` |
| `kp` / `kd` | `20.0` / `0.5` | nominal joint PD gains |
| `geometry` | `None` | `QuadrupedGeometry` for analytic controllers |

Properties: `num_dofs` (`len(joint_names)`) and `default_dof_angles`
(`tuple[float, ...]` in `joint_names` order).

### `GO2` and `GO2_GEOMETRY`

The Unitree Go2 spec, `urdf_path="urdf/go2/urdf/go2.urdf"` (bundled with
Genesis). Twelve joints in the order

```
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

Default pose: hips `0.0`, calves `-1.5`, **front thighs `0.8` and rear
thighs `1.0` rad** (the rear sits slightly higher). `foot_link_names` are
`FR_foot, FL_foot, RR_foot, RL_foot`; `base_init_pos=(0, 0, 0.42)`,
`base_init_quat=(1, 0, 0, 0)` — the true `wxyz` identity. (The legacy
scripts used `(0, 0, 0, 1)` believing Genesis was `xyzw`; in `wxyz` that is
a 180° yaw, harmless because all observations are body-frame, but corrected
here.) Gains `kp=20`, `kd=0.5`.

`GO2_GEOMETRY = QuadrupedGeometry(l_hip=0.0955, l_thigh=0.213, l_calf=0.213)`
from the `go2_description` URDF.

## `RobotState`

Dataclass of `[N, ...]` tensors filled by the state sensors on
`Robot.refresh()` and read by controllers, observation builders and rewards.
All fields are updated **in place** (`tensor[:] = ...`), so a consumer may
keep a reference to a field across steps. On the real robot the same
structure is filled from the state-estimation stack.

| Field | Shape | Frame | Source |
|-------|-------|-------|--------|
| `base_pos` | `[N, 3]` | world (m) | privileged (`SimBaseStateSensor`) |
| `base_quat` | `[N, 4]` | `wxyz` | IMU |
| `base_lin_vel_world` | `[N, 3]` | world (m/s) | privileged |
| `base_lin_vel` | `[N, 3]` | body (m/s) | privileged (rotated by IMU quaternion) |
| `base_ang_vel` | `[N, 3]` | body (rad/s) | IMU |
| `projected_gravity` | `[N, 3]` | body, unit vector | IMU |
| `base_euler` | `[N, 3]` | intrinsic X-Y-Z (rad), yaw at index 2 | IMU |
| `dof_pos` | `[N, D]` | canonical joint order (rad) | encoders |
| `dof_vel` | `[N, D]` | canonical joint order (rad/s) | encoders |
| `foot_contacts` | `[N, n_feet]` | float 0/1 | contact sensor, or a task-provided proxy |

```python
@classmethod
def zeros(cls, n_envs: int, n_dofs: int, n_feet: int,
          device: torch.device, dtype=torch.float32) -> RobotState
def zero_idx(self, envs_idx: torch.Tensor) -> None
```

`zeros` allocates an all-zero state with identity quaternions; `zero_idx`
zeroes every field for the given envs and restores `w = 1`.

## `Robot`

```python
class Robot:
    def __init__(self, spec: RobotSpec, scene: Scene, device: torch.device,
                 kp: float | None = None, kd: float | None = None,
                 base_init_pos: tuple | None = None,
                 base_init_quat: tuple | None = None)
```

The simulated robot facade. The constructor adds the articulation to the
(un-built) scene at the spawn pose; `kp`/`kd` default to the spec's nominal
gains and `base_init_pos`/`base_init_quat` to the spec's spawn pose (they
are also what `reset_idx` returns to). Lifecycle:

```python
robot = Robot(spec, scene, device)     # adds the articulation to the scene
scene.build(n_envs)
robot.bind(n_envs)                     # resolve DOFs, create sensors/actuators, allocate state
...
state = robot.refresh()                # once per control step
robot.set_joint_targets(targets)       # [N, D]
```

### `bind`

```python
def bind(self, n_envs: int) -> None
```

Call after `scene.build()`. Populates:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `n_envs` | `int` | |
| `dof_idx` | `list[int]` | engine-local DOF indices in canonical joint order |
| `default_dof_pos` | `Tensor [D]` | nominal pose from the spec |
| `state` | `RobotState` | allocated with `max(n_feet, 1)` contact columns so `foot_contacts` is always `[N, n_feet]` |
| `actuator` | `PDJointPositionActuator` | writes the gains to the engine once |
| `contact_sensor` | `SimContactSensor \| None` | `None` when the asset has no foot links or the backend exposes no contact forces |

State sensors are created in a **fixed order**: `SimIMU`, then
`SimJointEncoders`, then `SimBaseStateSensor`, then the contact sensor if
available. The IMU must run first because the others read
`state.base_quat`.

Missing foot links are tolerated: `link_indices` may raise an
engine-specific error (Genesis' bundled go2 merges them), which is logged at
`INFO` (`"no foot links for contact sensing"`) and results in
`contact_sensor = None`. Tasks then fall back to a stance-phase proxy.

### `refresh`, `set_joint_targets`, `reset_idx`

```python
def refresh(self) -> RobotState
def set_joint_targets(self, targets: torch.Tensor) -> None
def reset_idx(self, envs_idx: torch.Tensor, base_pos: torch.Tensor | None = None) -> None
```

* `refresh` runs every state sensor in order and returns `self.state`.
* `set_joint_targets` forwards `[N, D]` desired joint angles (canonical
  order) to the actuator.
* `reset_idx` resets the selected envs to the default configuration:
  joints to `default_dof_pos` with velocities zeroed, base to
  `base_init_pos` (or the optional `[len(envs_idx), 3]` `base_pos`, e.g. a
  terrain grid) and `base_init_quat`, then `zero_all_velocities`. The
  `RobotState` snapshot is written alongside the engine, so a controller
  reading `state` before the next `refresh()` already sees the reset pose.
  An empty `envs_idx` is a no-op.

```python
state = robot.refresh()
assert torch.allclose(quat_apply(state.base_quat, state.base_lin_vel),
                      state.base_lin_vel_world, atol=1e-6)   # body → world
robot.reset_idx(torch.tensor([0, 2]))
```

## Sensors

Two families, both ABCs in `domo/robot/sensors.py`:

| ABC | Method | Contract |
|-----|--------|----------|
| `StateSensor` | `update(state: RobotState) -> None` | writes its measurements into the shared `RobotState` in place; `Robot.refresh` calls them in a fixed order |
| `ExteroceptiveSensor` | `read() -> torch.Tensor` | owns its own measurement tensor (lidar, camera) |

Exteroceptive sensors handed to a control loop are duck-typed: anything
with `tick()` is advanced after time moves, anything with `reset_idx()` is
reset with the robot. `SectorLidar` and `SimulatedLidar` expose both.

### Simulated proprioception

| Class | Constructor | Writes |
|-------|-------------|--------|
| `SimIMU` | `(articulation, device)` | `base_quat`, body-frame `base_ang_vel`, `projected_gravity` (world `(0, 0, -1)` rotated into the body), `base_euler` via `quat_to_euler_xyz` |
| `SimJointEncoders` | `(articulation, dof_idx)` | `dof_pos`, `dof_vel` in canonical order |
| `SimBaseStateSensor` | `(articulation)` | `base_pos`, `base_lin_vel_world`, and `base_lin_vel = quat_apply_inverse(state.base_quat, world vel)`. **Privileged**: on the real robot this comes from a state estimator and is less reliable — tasks meant for transfer should prefer IMU/encoder observations. Must run after `SimIMU` |
| `SimContactSensor` | `(articulation, foot_link_idx, force_threshold: float = 1.0)` | `foot_contacts = (‖net contact force‖ > threshold)` as float. Probes `get_link_contact_forces()` at construction; any exception sets `available = False`, after which `update` writes nothing |

`foot_link_idx` are engine-local link indices in the order of
`RobotSpec.foot_link_names`; the threshold is in newtons.

### `SectorLidar`

```python
class SectorLidar(ExteroceptiveSensor):
    def __init__(self, handle: LidarSensorHandle, n_envs: int,
                 device: torch.device, update_interval: int = 1)
```

The plain 2-D sector lidar: `read()` returns `[N, n_horizontal]` minimum
distances straight from `handle.read_sector_distances()`, no device model.
`update_interval` emulates a sensor slower than the control loop — `tick()`
increments a step counter and refreshes the cache only every
`update_interval` steps; reads in between return the cached scan, as on
real hardware. `has_scan` is `True` once at least one scan has been taken;
`reset_idx(envs_idx)` resets those envs' cache to `max_range`.

Prefer `SimulatedLidar` for anything new; `SectorLidar` remains for
replicating the original scripts exactly.

## Actuators

```python
class Actuator(ABC):
    def apply(self, command: torch.Tensor) -> None      # [N, D], one control step

class PDJointPositionActuator(Actuator):
    def __init__(self, articulation: Articulation, dof_idx: Sequence[int],
                 kp: float, kd: float)
```

Joint-space PD position control with the gains living in the engine (or, on
hardware, in the firmware). The gains are written once at construction with
`Articulation.set_pd_gains` (shared by all envs) and stay readable as
`.kp` / `.kd` so `DomainRandomization` can scale the nominal values.
`apply(targets)` calls `set_joint_position_targets`. `Robot.set_joint_targets`
is its only caller.

## Lidar device models

`domo/robot/lidar_models.py` separates the *device* (beam geometry, scan
rate, range limits, noise) from the *backend geometry* (`LidarConfig`).
Policies consume sector minima, so the number of azimuth samples raycast in
simulation is a fidelity knob, not an interface change: train cheap (180
rays), validate dense (`XT16_FULL_AZIMUTH`).

### `LidarModelConfig`

| Field | Default | Meaning |
|-------|---------|---------|
| `name` | `"generic"` | |
| `n_horizontal` | `36` | azimuth samples raycast in sim (the fidelity knob) |
| `n_vertical` | `5` | elevation channels |
| `fov_deg` | `(360.0, 50.0)` | (horizontal, vertical) FOV |
| `rate_hz` | `10.0` | scan rate |
| `min_range` | `0.0` | blind zone (m); closer returns become no-returns |
| `max_range` | `4.0` | metres; no-returns are reported as this |
| `range_noise_std` | `0.0` | Gaussian range noise σ (m) |
| `dropout_prob` | `0.0` | probability a beam returns nothing |
| `pos_offset` | `(0.0, 0.0, 0.35)` | mount offset from the base link (m); measure on the real robot |
| `draw_debug` | `False` | draw rays in the viewer |

```python
def to_lidar_config(self) -> LidarConfig            # geometry-only view for Scene.add_lidar
def update_interval(self, control_dt: float) -> int  # max(1, round(1 / (rate_hz * control_dt)))
```

### Presets

```python
def hesai_xt16(n_horizontal: int = 180,
               pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.35),
               rate_hz: float = 10.0, draw_debug: bool = False) -> LidarModelConfig
def generic_sector_lidar(draw_debug: bool = False) -> LidarModelConfig

XT16_FULL_AZIMUTH = 1980
```

* `hesai_xt16` — the Hesai PandarXT-16 mounted on the real Go2: 16 channels
  over ±15° (`fov_deg=(360, 30)`, 2° spacing), 0.05–120 m, σ = 1 cm,
  0.3 % dropout, supported real frame rates 5 / 10 / 20 Hz. The real device
  scans 2000 points per ring; 2000 is not divisible by the 36 policy
  sectors, so full-fidelity simulation uses 1980 (55 rays per sector).
* `generic_sector_lidar` — the idealised 36 × 5, 4 m, noiseless sensor of
  the original avoidance scripts; the default `WorldConfig.lidar_model`.

### `SimulatedLidar`

```python
class SimulatedLidar(ExteroceptiveSensor):
    def __init__(self, handle: LidarSensorHandle, model: LidarModelConfig,
                 n_envs: int, n_sectors: int, device: torch.device,
                 control_dt: float)
```

Wraps a backend handle and applies the device imperfections on every scan.
Raises `ValueError` unless `model.n_horizontal % n_sectors == 0`, so every
sector pools the same number of rays.

| Member | Returns / effect |
|--------|------------------|
| `tick()` | advance one control step; when `step_count % update_interval == 0`, rescan: `handle.read_ranges()` → blind zone, Gaussian noise, dropout → sector pooling → (optionally) point cloud |
| `read()` | `[N, n_sectors]` sector minima of the last scan (min over channels, then min over each azimuth group); `max_range` before the first scan and for no-returns |
| `read_raw()` | `[N, n_vertical, n_horizontal]` last processed scan |
| `read_points()` | `(points [N, n_beams, 3], valid [N, n_beams] bool)` world-frame hit points and validity mask (`min_range < range < max_range`, minus an independent dropout draw) |
| `reset_idx(envs_idx)` | reset those envs' raw/sector caches to `max_range`, points to zero, validity to `False` |
| `update_interval` | property, control steps between scans |
| `has_scan` | property, `True` once a scan has been taken |
| `model`, `n_sectors`, `max_range` | attributes |

Point clouds are populated only when the handle exposes `read_points`
(`hasattr` check at construction; a handle that raises
`NotImplementedError` or `AttributeError` on first use switches the sensor
to sector-only mode permanently). Noise and dropout use torch's global RNG;
seed with `torch.manual_seed` for reproducible scans.

```python
from domo.robot import SimulatedLidar, hesai_xt16

model = hesai_xt16(n_horizontal=72)
handle = scene.add_lidar(robot.articulation, model.to_lidar_config())   # before build
scene.build(n_envs); robot.bind(n_envs)
lidar = SimulatedLidar(handle, model, n_envs, n_sectors=36, device=device, control_dt=0.02)
assert lidar.update_interval == 5              # 10 Hz at 50 Hz control
for _ in range(lidar.update_interval):
    lidar.tick()
sectors = lidar.read()                         # [N, 36]
```

## `DomainRandomization`

```python
from domo.robot.randomization import DomainRandomization
```

Per-env physics perturbations resampled on reset — the sim-side half of
sim-to-real robustness and the surface DrEureka optimises over. All
parameters are optional (`None` → untouched).

| Field | Default | Meaning |
|-------|---------|---------|
| `friction_range` | `None` | friction ratio, `~1.0`, one draw per env |
| `base_mass_range` | `None` | added payload on the base (kg), per env |
| `com_shift_range` | `None` | ± m, drawn independently on each of x, y, z, per env |
| `kp_scale_range` | `None` | × nominal `actuator.kp` — **one draw per reset group** |
| `kd_scale_range` | `None` | × nominal `actuator.kd` — one draw per reset group |
| `obs_noise_std` | `0.0` | task-level; consumed by tasks that support it |
| `action_latency_steps` | `0` | task-level; consumed by tasks that support it |

```python
@classmethod
def from_dict(cls, d: dict) -> DomainRandomization   # unknown keys ignored, lists → tuples
def to_dict(self) -> dict                             # ACTIVE parameters only (None / 0 omitted)
def apply(self, robot, envs_idx: torch.Tensor) -> None
```

`apply` takes a bound `Robot` (it uses `robot.articulation`, `robot.dof_idx`,
`robot.device` and the actuator's nominal gains) and the env indices being
reset (empty → no-op). Each physics parameter goes through the matching
`Articulation` capability; a backend that raises `NotImplementedError` for
one of them is **skipped with a single warning per capability per
instance** (`"[dr] backend does not support 'friction' — skipped"`), so the
same config runs on any backend. `from_dict` / `to_dict` are the JSON seam
used by the Eureka loop and checkpoints.

The PD-gain scale is drawn once per `apply` call because engine gains are
at most per-DOF and broadcast over envs; env diversity still emerges
because envs reset at different times.

## How to add a sensor

Decide which family it belongs to:

* It measures the robot itself and the quantity has a slot in
  `RobotState` → a `StateSensor`. Take the handle in `__init__`, write into
  the state in `update`, and append it to `Robot._state_sensors` in
  `Robot.bind` (after `SimIMU` if it needs `base_quat`).
* It measures the surroundings and returns its own tensor → an
  `ExteroceptiveSensor`. Implement `read()`; if it runs slower than the
  control loop, cache the last measurement and add `tick()` and
  `reset_idx(envs_idx)` like `SimulatedLidar`, then pass the instance to
  the control loop / task explicitly (`World.sensors` is the list the loop
  ticks).

```python
from domo.robot import StateSensor, RobotState
from domo.sim.base import Articulation

class SimBaseHeight(StateSensor):
    """Example: a state sensor that only needs the articulation handle."""

    def __init__(self, articulation: Articulation):
        self._art = articulation

    def update(self, state: RobotState) -> None:
        state.base_pos[:, 2] = self._art.get_base_position()[:, 2]
```

The real-robot version implements the same class with the same `update` /
`read` signature on top of the hardware topic; nothing above `domo.robot`
changes. If the sensor needs a new physics query, add it to the
`Articulation` (or a new handle) contract in `domo.sim.base` as an optional
capability with a `NotImplementedError` default, implement it in the
backend, and probe it as `SimContactSensor` does.

## Gotchas

* **`base_euler` is intrinsic X-Y-Z**, produced by `quat_to_euler_xyz`, the
  Genesis default; it is not aerospace roll-pitch-yaw. Cross terms differ in
  sign. Treat it as the observation feature it is; yaw is index 2.
* **`SimBaseStateSensor` is privileged.** `base_pos`, `base_lin_vel_world`
  and `base_lin_vel` exist in sim for free; on the real robot they come from
  an estimator. Observations meant to transfer should be built from IMU and
  encoder fields.
* **`contact_sensor` is `None` on Genesis' bundled go2** because the foot
  links are merged into the calves; `foot_contacts` stays zero unless a task
  writes a proxy into it.
* **Sensor order matters.** `SimIMU` fills `base_quat`; `SimBaseStateSensor`
  rotates with it. A custom sensor that depends on a state field must be
  appended after the sensor that fills it.
* **`SimulatedLidar` pools before the policy sees anything**: `read()` is
  sector minima, so changing `n_horizontal` changes fidelity, not the
  observation size. Changing `n_sectors` does change the observation size
  and therefore what a checkpoint means.
* **Point clouds and `read_raw()` are separate products**: dropout is drawn
  independently for each, and the cloud is in the backend's native beam
  order, not the `[N, n_v, n_h]` grid.
* **`DomainRandomization.to_dict()` drops inactive knobs**, so a
  round-tripped config prints as the minimal set of parameters that are on.
* Nominal gains (`kp=20`, `kd=0.5` in `GO2`) are not what the twin uses:
  `WorldConfig` defaults to the stiff `100 / 2` pair, and tasks carry their
  own. Pass gains to `Robot`, never mutate the (frozen) spec.
