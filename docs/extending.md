# Extending DOMO

Every extension point below is a class to subclass or a registry to add
to, and each has a real file in the repository that serves as its
template. The rules that make an extension fit are the layering rules of
[architecture.md](architecture.md) and the conventions of
[conventions.md](conventions.md): tensors are `[N, ...]` on the sim device,
quaternions are `wxyz`, nothing above `domo/sim/` imports an engine, and
everything engine-free gets a test under `tests/` that runs in seconds.

| Extension | Subclass / registry | Template |
|-----------|---------------------|----------|
| [a task](#add-a-task) | `domo.tasks.VecTask` + `domo/tasks/__init__.py` | `domo/tasks/go2_cpg_walk.py` |
| [a reward term](#add-a-reward-term) | `_reward_<name>` + `reward_scales` | any task |
| [a skill and its card](#add-a-skill-and-a-card) | `Skill` / `CommandSkill` + `SkillCard` + `library.register` | `domo/control/skill.py`, `domo/skills/builtin.py` |
| [a controller or planner](#add-a-controller-or-planner) | `Controller` / `PlanningController` | `examples/avoidance/go2_cpg_rl_lidar.py::AvoidanceMission` |
| [a sensor](#add-a-sensor) | `StateSensor` / `ExteroceptiveSensor` | `domo/robot/sensors.py` |
| [a scene kind](#add-a-scene-kind) | builder in `domo/scenes/` + `World._build_environment` | `domo/scenes/arena.py` |
| [a physics backend](#add-a-physics-backend) | `PhysicsEngine`, `Scene`, `Articulation`, `LidarSensorHandle` + `register_backend` | `tests/test_foundation_world.py` |
| [an LLM provider](#add-an-llm-provider) | `LLMClient` + `make_llm` | `domo/llm/langchain_client.py` |
| [an Eureka task spec](#add-an-eureka-taskspec) | `TaskSpec` in `TASK_REGISTRY` | `domo/eureka/spec.py` |

## Add a task

A task is a `VecTask` subclass in `domo/tasks/<name>.py` with a config
dataclass. The contract it must honour is in
[api/tasks.md](api/tasks.md#vectask); the full reference implementation is
`domo/tasks/go2_cpg_walk.py` (scene built directly) and
`domo/tasks/go2_avoid.py` (scene built through `World`).

1. **Constants and config.** `OBS_DIM` / `ACT_DIM` as class attributes and
   module constants; a `@dataclass` config whose fields are the checkpoint
   contract (`n_envs`, `dt`, `max_episode_steps`, `device`, `headless`,
   `engine`, then the task's own knobs, then `reward_scales`). Defaults
   must reproduce your reference run; never change one silently.
2. **Build the scene.** Either `create_engine(cfg.engine, device=cfg.device)`
   → `engine.create_scene(SimConfig(...))` → add ground / terrain →
   `Robot(spec, scene, device, kp, kd, ...)` → `scene.build(n_envs)` →
   `robot.bind(n_envs)`, or `World(WorldConfig(...), n_envs=cfg.n_envs)`
   when the twin should be able to spawn in the same environment. Import
   `domo.world` lazily inside `__init__` so `domo.tasks` stays importable
   without the scene loaders.
3. **Register rewards.** `self.register_rewards(cfg.reward_scales)` after
   the robot is bound; every key needs a `_reward_<name>` method.
4. **`step`** in the legged-gym order: apply the action →
   `scene.step()` → `episode_length_buf += 1` → `robot.refresh()` (and
   tick exteroceptive sensors) → `reset_buf = mark_time_outs() |
   fall_termination(...) | ...` → `reset_idx(reset_buf.nonzero(...))` →
   `compute_rewards()` → build `obs_buf` → update `last_*` buffers →
   return the 5-tuple.
5. **`reset`**: set `reset_buf[:] = True`, `reset_idx(all envs)`,
   `robot.refresh()`, return a fresh observation.
6. **`reset_idx`**: return early when empty; `robot.reset_idx(envs_idx)`
   plus any skill / sensor resets; zero the per-env buffers and
   `episode_length_buf`; set `reset_buf[envs_idx] = True`;
   `log_episode_sums(envs_idx, self.episode_length_s)`; resample commands.
7. **Export.** Add the task, its config and constants to
   `domo/tasks/__init__.py` and its `__all__`; `tests/test_public_api.py`
   imports every listed name.

```python
"""Go2 <name> task: one-line purpose. Observation/action/termination summary."""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from domo.robot import GO2, Robot, RobotSpec
from domo.sim import SimConfig, create_engine
from .base import VecTask
from .common import fall_termination

NAME_OBS_DIM, NAME_ACT_DIM = 42, 12

@dataclass
class Go2NameConfig:
    n_envs: int = 4096
    dt: float = 0.02
    max_episode_steps: int = 1000
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"
    kp: float = 100.0
    kd: float = 2.0
    action_scale: float = 0.25
    termination_pitch: float = 1.0
    termination_roll: float = 1.0
    reward_scales: dict[str, float] = field(default_factory=lambda: {"alive": 1.0})

class Go2NameTask(VecTask):
    OBS_DIM, ACT_DIM = NAME_OBS_DIM, NAME_ACT_DIM

    def __init__(self, cfg: Go2NameConfig, spec: RobotSpec = GO2):
        engine = create_engine(cfg.engine, device=cfg.device)
        super().__init__(cfg.n_envs, self.OBS_DIM, self.ACT_DIM, engine.device,
                         cfg.dt, max_episode_length=cfg.max_episode_steps)
        self.cfg = cfg
        self.episode_length_s = cfg.max_episode_steps * cfg.dt
        self.scene = engine.create_scene(SimConfig(dt=cfg.dt, substeps=2,
                                                   device=cfg.device, headless=cfg.headless))
        self.scene.add_ground()
        self.robot = Robot(spec, self.scene, self.device, kp=cfg.kp, kd=cfg.kd)
        self.scene.build(cfg.n_envs)
        self.robot.bind(cfg.n_envs)
        self.register_rewards(cfg.reward_scales)
        self.actions = torch.zeros((cfg.n_envs, self.ACT_DIM), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

    def _observe(self) -> torch.Tensor:
        s = self.robot.state
        return torch.cat([s.base_ang_vel * 0.25, s.projected_gravity,
                          s.dof_pos - self.robot.default_dof_pos,
                          s.dof_vel * 0.05, self.last_actions], dim=-1)

    def step(self, actions: torch.Tensor):
        cfg, state = self.cfg, self.robot.state
        self.actions = actions
        self.robot.set_joint_targets(actions * cfg.action_scale + self.robot.default_dof_pos)
        self.scene.step()
        self.episode_length_buf += 1
        self.robot.refresh()
        self.reset_buf = self.mark_time_outs()
        self.reset_buf |= fall_termination(state, cfg.termination_pitch, cfg.termination_roll)
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())   # before rewards/obs
        self.compute_rewards()
        self.obs_buf = self._observe()
        self.last_actions[:] = self.actions
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.robot.refresh()
        self.obs_buf = self._observe()
        return self.obs_buf, None

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        self.robot.reset_idx(envs_idx)
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.log_episode_sums(envs_idx, self.episode_length_s)

    def _reward_alive(self):
        return torch.ones(self.n_envs, device=self.device)
```

If the task is to be an Eureka target, add `compute_success()`,
`compute_fitness()` and `episode_outcomes` as in `domo/tasks/go2_getup.py`
and register a [TaskSpec](#add-an-eureka-taskspec).

**Tests** (engine-free, in `tests/test_tasks_common.py` or a new
`tests/test_tasks_<name>.py`): the config defaults (`_defaults(cls)`
pattern), `(OBS_DIM, ACT_DIM)` against the constants, every
`reward_scales` key resolving to a `_reward_<name>` method, and the
observation builder's index layout on a hand-filled `RobotState.zeros`.
Stepping the real task is a CPU smoke run of its entry point with
`--n-envs 2`, not a unit test.

## Add a reward term

A term is a method `_reward_<name>(self) -> Tensor[N]` returning the
**unscaled** value and an entry `"<name>": weight` in the config's
`reward_scales`. The registry multiplies by `weight × dt`, accumulates the
per-episode sum and reports `extras["episode"]["rew_<name>"]`.

```python
# in the config
reward_scales: dict[str, float] = field(default_factory=lambda: {
    ...,
    "foot_slip": -0.1,
})

# in the task
def _reward_foot_slip(self):
    """Σ over stance feet of planar foot speed² (penalty)."""
    ...
    return slip                                   # [N], unscaled
```

Conventions: penalties return a positive quantity and carry a negative
weight; docstrings state the formula; names are referenced by string in
configs, in checkpoints (`extra["task_config"]["reward_scales"]`) and in
Eureka prompts, so renaming one is a contract change. A zero weight keeps
the term registered and logged, which is the way to A/B a term without
touching code paths. The test in `tests/test_tasks_common.py::
test_reward_scale_keys_have_reward_methods` catches a key without a
method; add the new key to the task's default-check test.

## Add a skill and a card

A skill is a motor primitive (`state → joint targets [N, 12]`) or a
command skill (`state → command for the motor skill below it`). Both live
in `domo/control/`, are pure torch, and are described to the planner by a
`SkillCard` registered in the library. The class reference is in
[api/control.md](api/control.md#skills); worked examples of both kinds are
in [api/control.md — How to](api/control.md#how-to) and the card recipe in
[api/skills.md — Add a card](api/skills.md#add-a-card-to-the-library).

```python
class Skill(ABC):
    name: str = "skill"
    def setup(self, robot) -> None                          # after robot.bind(); allocate per-env state
    def reset_idx(self, envs_idx: torch.Tensor) -> None    # default no-op
    def update(self, state, dt: float) -> torch.Tensor     # abstract: [N, D] joint targets

class CommandSkill(Skill):
    channel: str = "velocity"                               # must match the base skill's accepted channel
    additive: bool = True                                   # add to (True) or author (False) the base command
    def configure(self, **params) -> None                   # grammar parameters; default rejects any
    def success_flags(self, state) -> torch.Tensor | None  # bool [N]; None = never finishes on its own
    def update_command(self, state, dt: float) -> torch.Tensor   # abstract: e.g. [N, 3] Δ(vx, vy, vyaw)
```

A learned policy needs no new class: wrap it in `LearnedJointSkill`
(joint-space observation builder + policy) or `CPGLocomotionSkill`
(velocity tracking).

Then the card and the registration, in `domo/skills/builtin.py` for a
skill that ships with the library or in your own code otherwise:

```python
from domo.skills import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard

CROUCH_CARD = SkillCard(
    name="crouch",
    description="Lower the stance by a fixed depth and hold it.",
    interface=MOTOR,                                       # or CMD_VELOCITY for a command skill
    params=[ParamSpec("depth", "how far to lower", default=0.2, range=(0.0, 0.4), unit="m")],
    preconditions=["standing on ground"],
    effects=["base lowered by depth"],
    success_when=[],                                       # bound with .for()/.until()
    fail_when=["fallen(0.15)"],
    safety_notes=["cannot walk while crouched"],
)

library.register(CROUCH_CARD, lambda: CrouchSkill())       # factory: a FRESH instance each time
```

`SkillLibrary.register(card, factory)` stores both; `library.compile(text)`
type-checks channels against `interface` / `accepts` and calls the factory
once per occurrence in the program. Conditions that need a sensor are
closures over the sensor object, registered with
`library.register_condition(name, factory)` exactly as `make_go2_library`
does for `blocked(d)` / `clear(d)`. Nothing else in the stack changes; the
planner sees the new card in `library.describe()` on the next call.

**Tests**: the skill on a hand-filled `RobotState` (`tests/test_skill_controller.py`,
`tests/test_control_nav_skill.py`), the card in a library compiling a
program that uses it and asserting the `CompositeSkill` trace
(`tests/test_skills_library.py`). No engine in either.

## Add a controller or planner

A `Controller` orchestrates skills at 1–10 Hz; a `PlanningController` does
so by writing programs in the skill grammar. Both live above `domo/control`
and `domo/skills` and stay engine-free
([api/control.md](api/control.md#controllers),
[api/skills.md](api/skills.md#planning)).

```python
class Controller(ABC):
    def __init__(self, skills: dict[str, Skill], initial: str, decision_interval: int = 5)
    def setup(self, robot) -> None
    def reset_idx(self, envs_idx: torch.Tensor) -> None
    def activate(self, name: str) -> None                   # switch; the newcomer is reset first
    def decide(self, state) -> None                         # abstract: called every decision_interval steps
    def update(self, state, dt: float) -> torch.Tensor      # decide (maybe), then the active skill's targets
    ticks: int                                              # property: steps since reset

class PlanningController(Controller):
    def __init__(self, library: SkillLibrary, decision_interval: int = 10)
    def plan(self, state, last: PlanOutcome | None) -> str | None   # override: grammar text, or None to idle
```

A hand-written controller sets skill attributes (for instance
`self.skills["walk"].command[:] = ...`) and calls `activate`; a planner
returns program text such as `"(avoid @ walk(vx=0.6)).until(moved(4)) >>
stand.for(1)"` and reads `last.succeeded` and `last.trace` to decide the
next one. Skill programs are never CLI parameters: the mission is authored
inside the controller, as `examples/avoidance/go2_cpg_rl_lidar.py::AvoidanceMission`
and `examples/twin/twin_demo.py` do. Wire it with
`controller.setup(world.robot)`, `loop = world.make_loop(controller)`,
`loop.reset()`, `loop.step()` / `loop.run(n)`.

**Tests**: `tests/test_skill_controller.py` and `tests/test_planner.py` drive
controllers with fake skills and a fake `RobotState`; a new planner gets
the same treatment, asserting the sequence of programs it emits for a
scripted sequence of outcomes.

## Add a sensor

Sensors are the only classes besides actuators that touch physics handles.
Two families ([api/robot.md](api/robot.md#sensors), with a worked example
in [How to add a sensor](api/robot.md#how-to-add-a-sensor)):

* `StateSensor.update(state)` writes into the shared `RobotState`. Append
  the instance to `self._state_sensors` in `Robot.bind`, after `SimIMU`
  when it needs `state.base_quat` (the IMU runs first for that reason).
  If the quantity has no slot, add a field to `RobotState` and its
  allocation in `RobotState.zeros`.
* `ExteroceptiveSensor.read()` returns its own tensor. If it runs slower
  than the control loop, cache the measurement and expose `tick()` and
  `reset_idx(envs_idx)` like `SectorLidar` / `SimulatedLidar`; the control
  loop ticks every sensor in `World.sensors`, and a task must tick it in
  `step` and reset it in `reset_idx` itself.

```python
class StateSensor(ABC):
    def update(self, state: RobotState) -> None: ...

class ExteroceptiveSensor(ABC):
    def read(self) -> torch.Tensor: ...
```

A sensor that needs a physics query the handles do not offer adds it to
`Articulation` (or a new handle) in `domo/sim/base.py` as an optional
capability with a `NotImplementedError` default, implements it in the
backend, and probes it at construction the way `SimContactSensor` sets
`available`. The real-robot driver reimplements the same class over a
hardware topic; nothing above `domo/robot` changes.

**Tests**: a fake handle returning fixed tensors, asserting the fields the
sensor writes (`tests/test_foundation_robot.py`), or for a device model
the noise / dropout / cadence behaviour (`tests/test_lidar_models.py`).

## Add a scene kind

A scene kind is a builder in `domo/scenes/` that uses only the `Scene`
contract, plus a branch in the World. The steps are in
[api/world-and-services.md — Add a scene kind](api/world-and-services.md#add-a-scene-kind);
in short:

1. Builder in `domo/scenes/<name>.py`: takes a `Scene` and a config
   dataclass, adds entities **before** `build()`, does per-env work
   (randomisation, teleports) in a method called **after** build, as
   `ObstacleArena.randomise` does. Export it from `domo/scenes/__init__.py`.
2. A config field on `WorldConfig`, a branch in `World._build_environment`
   returning whatever handle the World should keep (or `None`), and the
   name appended to `SCENE_KINDS`.
3. If a task should train in it, the same branch in the task (the tasks
   built over `World` get it for free through `scene_kind`).

```python
elif cfg.scene_kind == "corridor":
    self.scene.add_ground(cfg.ground_height)
    return Corridor(self.scene, cfg.corridor, spawn_xy=cfg.base_init_pos[:2])
```

**Tests**: build it against `tests/test_foundation_scenes.py::RecordingScene`
and assert the recorded `add_*` calls; a `World` with the fake backend of
`tests/test_foundation_world.py` and `scene_kind="corridor"` checks the
wiring.

## Add a physics backend

`domo/sim/genesis_backend.py` is the only module that imports `genesis`. A
second engine is `domo/sim/<name>_backend.py` implementing the ABCs of
`domo/sim/base.py` and registering a factory; the full walkthrough with
the complete fake is in [api/sim.md — Adding a backend](api/sim.md#adding-a-backend).

What to implement:

| ABC | Required | Optional (default raises `NotImplementedError`) |
|-----|----------|--------------------------------------------------|
| `PhysicsEngine` | `create_scene(cfg: SimConfig) -> Scene`, `device` property | |
| `Scene` | `add_ground`, `add_terrain`, `add_mesh`, `add_urdf_prop`, `add_box`, `add_cylinder`, `add_sphere`, `add_articulation(urdf_path, pos, quat_wxyz)`, `add_lidar(articulation, cfg)`, `build(n_envs)`, `step()` | `add_camera(...)` |
| `Articulation` | `dof_indices`, `link_indices`, the `get_base_*` / `get_joint_*` readers, `set_pd_gains`, `set_joint_position_targets`, `set_base_pose`, `set_joint_positions`, `zero_all_velocities` | `get_link_contact_forces`, `set_friction_ratio`, `set_base_mass_shift`, `set_base_com_shift`, `set_pd_gains_scaled` |
| `LidarSensorHandle` | `config`, `read_sector_distances() -> [N, n_horizontal]`, `read_ranges() -> [N, n_vertical, n_horizontal]` | `read_points()` |
| `CameraHandle` | `render()` | |

Rules: every tensor is `[N, ...]` on `engine.device`; quaternions are
`wxyz`; lidar rays are laid out `[N, n_vertical, n_horizontal]` with
azimuth 0 at +x running counter-clockwise, and an engine that returns
another layout normalises it with `lidar_ranges_to_grid`; entities are
added before `build`, moved after. Register with

```python
from domo.sim import register_backend
register_backend("<name>", lambda **kw: NameEngine(**kw))      # kwargs come from create_engine
```

and every config's `engine="<name>"` selects it. `Robot` probes the
optional capabilities (`SimContactSensor.available`) and the tasks fall
back gracefully, so a backend can start with the required set only.

**Tests**: `tests/test_foundation_world.py` is the template and the
acceptance test. It stands up a complete `World` (engine → scene → arena
→ robot → lidar → build → bind → device-model lidar) on an in-memory fake
engine with no physics at all. A new backend gets the same file pattern
plus a CPU smoke run of `examples/basic_examples/skill_demo.py` with
`engine` switched.

## Add an LLM provider

`domo.llm.LLMClient` is one method, `generate(prompt, temperature=1.0) ->
str`. Providers keep their SDK imports inside the constructor so the core
library stays dependency-free, and `make_llm(provider, **kwargs)` is the
single dispatch point Eureka and the planners use
([api/llm-hri.md](api/llm-hri.md)).

Two ways to add one:

```python
# 1. a direct client (domo/llm/client.py)
class MyClient(LLMClient):
    def __init__(self, model: str = "...", api_key: str | None = None, max_tokens: int = 16384):
        import my_sdk                                   # inside __init__, never at module level
        self._client = my_sdk.Client(api_key or os.environ["MY_API_KEY"])
        ...

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        return self._client.complete(prompt, temperature=temperature).text

# 2. a LangChain chat model (domo/llm/langchain_client.py)
def my_client(model: str, **kw) -> LangChainClient:
    from langchain_my import ChatMy
    return LangChainClient(ChatMy(model=model, **kw), supports_temperature=True)
```

Then a branch in `make_llm`:

```python
if provider == "my":
    return MyClient(**kwargs)
```

and the name in the `ValueError` message and in `SkillLearningRequest.llm`'s
docstring. Replies go through `extract_code_block` / `extract_json_block`,
which salvage truncated fences; if the provider is a thinking model, size
`max_tokens` so reasoning does not eat the output budget
([troubleshooting.md](troubleshooting.md#llm-providers)).

**Tests**: `tests/test_llm_client.py` covers the parsing helpers and
`make_llm` dispatch with the SDK absent; test the new branch the same way
(monkeypatch the SDK module, assert the constructor kwargs) and keep the
network out of the test suite. `ScriptedClient` remains the provider the
end-to-end tests run against.

## Add an Eureka TaskSpec

Making a task learnable through `domo.eureka` is a registry entry, once
the task exposes the fixed metric. The requirements and a complete
example are in [api/eureka.md — Adding a task](api/eureka.md#adding-a-task);
the reference implementation is `domo/tasks/go2_getup.py`.

```python
@dataclass(frozen=True)
class TaskSpec:
    module: str                 # "domo.tasks.go2_jump"
    task_class: str             # "Go2JumpTask"
    config_class: str           # "Go2JumpConfig"
    success_description: str    # the FIXED metric, stated exactly as the task computes it
    env_interface: str          # every `task` attribute the generated reward may read, with shapes and units
```

The worker imports `module`, builds `task_class(config_class(**task_overrides))`
(so the config must accept `n_envs`, `device`, `headless` and `dr`),
installs the generated reward with `set_reward_override`, sanity-steps it
once, trains, and ranks candidates on `episode_outcomes` (`success`,
`fitness`). The task therefore needs `compute_success() -> bool [N]`,
`compute_fitness() -> [N] in [0, 1]`, both reward-independent, and one
`episode_outcomes` record per finished episode. Add the entry to
`TASK_REGISTRY` in `domo/eureka/spec.py` (or at run time from user code).

**Tests**: `tests/test_eureka.py` and `tests/test_eureka_routine.py` run the
routine with `ScriptedClient` and a fake worker; extend the registry test
so the new spec's `module`, `task_class` and `config_class` import and the
config accepts the worker's override keys.
