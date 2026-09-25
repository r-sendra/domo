# Extending DOMO

Nine extension points, each a class to subclass or a registry to add to,
and each with a file in the repository that already does it correctly. This
page is the recipe for every one of them: the steps, a minimal skeleton
with the real signatures, and the test file to copy.

The rules that make an extension *fit* are the layering rules of
[architecture](../concepts/architecture.md) and the conventions of
[conventions](../concepts/conventions.md). In one line: tensors are
`[N, ...]` on the sim device, quaternions are `wxyz`, nothing above
`domo/sim/` imports a physics engine, and everything engine-free gets a
test under `tests/` that runs in seconds.

## Where to start

| I want to… | Extension point | Template in the repo | Test to copy |
|------------|-----------------|----------------------|--------------|
| train a new behaviour with a reward | [a task](#add-a-task) | `domo/tasks/go2_cpg_walk.py` | `tests/test_tasks_common.py` |
| change what an existing task optimises | [a reward term](#add-a-reward-term) | any task in `domo/tasks/` | `tests/test_tasks_common.py` |
| give the planner a new primitive | [a skill and its card](#add-a-skill-and-a-card) | `domo/control/skill.py`, `domo/skills/builtin.py` | `tests/test_skill_controller.py`, `tests/test_skills_library.py` |
| decide what the robot does next | [a controller or planner](#add-a-controller-or-planner) | `examples/avoidance/go2_cpg_rl_lidar.py::AvoidanceMission` | `tests/test_planner.py` |
| read a new quantity off the robot | [a sensor](#add-a-sensor) | `domo/robot/sensors.py` | `tests/test_foundation_robot.py` |
| put the robot somewhere new | [a scene kind](#add-a-scene-kind) | `domo/scenes/arena.py` | `tests/test_foundation_scenes.py` |
| swap Genesis for another engine | [a physics backend](#add-a-physics-backend) | `tests/test_foundation_world.py` | `tests/test_foundation_world.py` |
| use a different language model | [an LLM provider](#add-an-llm-provider) | `domo/llm/client.py`, `domo/llm/langchain_client.py` | `tests/test_llm_client.py` |
| let Eureka write rewards for a task | [an Eureka `TaskSpec`](#add-an-eureka-taskspec) | `domo/eureka/spec.py` | `tests/test_eureka.py` |

## Add a task

A task is a `VecTask` subclass in `domo/tasks/<name>.py` with a config
dataclass. The contract it must honour is
[`VecTask`](../api/tasks.md#vectask); the reference implementations are
`domo/tasks/go2_cpg_walk.py` (scene built directly) and
`domo/tasks/go2_avoid.py` (scene built through `World`).

1. **Constants and config.** `OBS_DIM` / `ACT_DIM` as class attributes and
   as module constants; a `@dataclass` config whose fields are the
   checkpoint contract — `n_envs`, `dt`, `max_episode_steps`, `device`,
   `headless`, `engine`, then the task's own knobs, then `reward_scales`.
   Defaults must reproduce your reference run.
2. **Build the scene.** Either `create_engine(cfg.engine, device=cfg.device)`
   → `engine.create_scene(SimConfig(...))` → ground or terrain →
   `Robot(spec, scene, device, kp, kd, ...)` → `scene.build(n_envs)` →
   `robot.bind(n_envs)`; or `World(WorldConfig(...), n_envs=cfg.n_envs)`
   when the twin should be able to spawn in the same environment. Import
   `domo.world` lazily inside `__init__` so `domo.tasks` stays importable
   without the scene loaders.
3. **Register the rewards** with `self.register_rewards(cfg.reward_scales)`
   once the robot is bound. Every key needs a `_reward_<name>` method.
4. **Write `step`** in the legged-gym order: apply the action →
   `scene.step()` → `episode_length_buf += 1` → `robot.refresh()` and tick
   any exteroceptive sensor → `reset_buf = mark_time_outs() |
   fall_termination(...)` → `reset_idx(...)` → `compute_rewards()` → build
   `obs_buf` → update the `last_*` buffers → return the 5-tuple. Resetting
   *before* rewards and observations is what keeps a reset env's first
   observation valid.
5. **Write `reset`**: set `reset_buf[:] = True`, `reset_idx` over all envs,
   `robot.refresh()`, return a fresh observation.
6. **Write `reset_idx`**: return early when empty; `robot.reset_idx(envs_idx)`
   plus any skill or sensor resets; zero the per-env buffers and
   `episode_length_buf`; set `reset_buf[envs_idx] = True`;
   `log_episode_sums(envs_idx, self.episode_length_s)`; resample commands.
7. **Export it.** Add the task, its config and its constants to
   `domo/tasks/__init__.py` and its `__all__` — `tests/test_public_api.py`
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
        self.scene.build(cfg.n_envs)          # (1)!
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
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())   # (2)!
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

1.  Entities are added before `build()` and moved after it. Anything
    added later raises inside the engine.
2.  Before rewards and observations, so a freshly reset env reports the
    pose it was reset to.

If the task is to be an Eureka target it also needs `compute_success()`,
`compute_fitness()` and `episode_outcomes`, as in `domo/tasks/go2_getup.py`,
plus a [`TaskSpec`](#add-an-eureka-taskspec).

**Tests to add** — in `tests/test_tasks_common.py` or a new
`tests/test_tasks_<name>.py`, following the `_defaults(cls)` pattern: the
config defaults, `(OBS_DIM, ACT_DIM)` against the module constants, every
`reward_scales` key resolving to a `_reward_<name>` method, and the
observation builder's index layout on a hand-filled `RobotState.zeros`.
Stepping the real task is a CPU smoke run of its entry point with
`--n-envs 2`, not a unit test.

## Add a reward term

A term is a method `_reward_<name>(self) -> Tensor[N]` returning the
**unscaled** value, plus an entry `"<name>": weight` in the config's
`reward_scales`. The registry multiplies by `weight × dt`, accumulates the
per-episode sum and reports it as `extras["episode"]["rew_<name>"]`.

1. Add the key and weight to the config's `reward_scales` default.
2. Add the matching method to the task.
3. Add the key to the task's default-check test.

```python
# in the config
reward_scales: dict[str, float] = field(default_factory=lambda: {
    ...,
    "foot_slip": -0.1,          # penalties carry the negative weight
})

# in the task
def _reward_foot_slip(self):
    """Σ over stance feet of planar foot speed² (penalty)."""
    ...
    return slip                 # [N], unscaled, non-negative
```

Penalties return a positive quantity and carry a negative weight;
docstrings state the formula. Names are referenced by string in configs, in
checkpoints (`extra["task_config"]["reward_scales"]`) and in Eureka
prompts, so renaming a term is a contract change, not a refactor. A weight
of zero keeps the term registered and logged, which is how to A/B a term
without touching any code path.

**Tests to add**:
`tests/test_tasks_common.py::test_reward_scale_keys_have_reward_methods`
already catches a key with no method; extend the task's defaults test with
the new key so the weight itself is pinned.

## Add a skill and a card

A skill is a motor primitive (`state → joint targets [N, 12]`) or a command
skill (`state → a command for the motor skill below it`). Both live in
`domo/control/`, are pure torch, and are described to the planner by a
`SkillCard` registered in a `SkillLibrary`. The class reference is
[Skills](../api/control.md#skills), worked examples of both kinds are in
[How to](../api/control.md#how-to), and the card recipe is
[Add a card](../api/skills.md#add-a-card-to-the-library).

1. Subclass `Skill` (motor) or `CommandSkill` (a layer above one).
2. Allocate per-env state in `setup(robot)`, after `robot.bind()`.
3. Clear that state in `reset_idx(envs_idx)`.
4. Write a `SkillCard` describing it in natural language.
5. Register the card with a **factory**, not an instance.

```python
class Skill(ABC):
    name: str = "skill"
    def setup(self, robot) -> None                         # after robot.bind()
    def reset_idx(self, envs_idx: torch.Tensor) -> None    # default: no-op
    def update(self, state, dt: float) -> torch.Tensor     # abstract: [N, D] joint targets

class CommandSkill(Skill):
    channel: str = "velocity"                              # must match the base skill's channel
    additive: bool = True                                  # add to (True) or author (False) the command
    def configure(self, **params) -> None                  # grammar parameters; default rejects any
    def success_flags(self, state) -> torch.Tensor | None  # bool [N]; None = never finishes on its own
    def update_command(self, state, dt: float) -> torch.Tensor   # abstract: e.g. [N, 3] Δ(vx, vy, vyaw)
```

A learned policy needs no new class at all: wrap it in `LearnedJointSkill`
(joint-space observation builder plus policy) or `CPGLocomotionSkill`
(velocity tracking).

Then the card and the registration — in `domo/skills/builtin.py` for a
skill that ships with the library, in your own code otherwise:

```python
from domo.skills import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard

CROUCH_CARD = SkillCard(
    name="crouch",
    description="Lower the stance by a fixed depth and hold it.",
    interface=MOTOR,                                    # or CMD_VELOCITY for a command skill
    params=[ParamSpec("depth", "how far to lower", default=0.2, range=(0.0, 0.4), unit="m")],
    preconditions=["standing on ground"],
    effects=["base lowered by depth"],
    success_when=[],                                    # bound by .for() / .until()
    fail_when=["fallen(0.15)"],
    safety_notes=["cannot walk while crouched"],
)

library.register(CROUCH_CARD, lambda: CrouchSkill())    # a FRESH instance per occurrence
```

`SkillLibrary.register(card, factory)` stores both; `library.compile(text)`
type-checks channels against `interface` and `accepts`, and calls the
factory once per occurrence in the program. A condition that needs a sensor
is a closure over that sensor, registered with
`library.register_condition(name, factory)` exactly as `make_go2_library`
does for `blocked(d)` and `clear(d)`. Nothing else in the stack changes:
the planner sees the new card in `library.describe()` on its next call.

**Tests to add**: the skill on a hand-filled `RobotState`
(`tests/test_skill_controller.py`, `tests/test_control_nav_skill.py`), and
the card in a library that compiles a program using it and asserts the
resulting `CompositeSkill` trace (`tests/test_skills_library.py`). No
engine in either.

## Add a controller or planner

A `Controller` orchestrates skills at 1–10 Hz; a `PlanningController` does
so by writing programs in the skill grammar. Both sit above `domo/control`
and `domo/skills` and stay engine-free
([Controllers](../api/control.md#controllers),
[Planning](../api/skills.md#planning)).

1. Subclass `Controller` (you own the switching) or `PlanningController`
   (you emit grammar text and the executor owns the switching).
2. Implement `decide(state)` or `plan(state, last)`.
3. Wire it: `controller.setup(world.robot)` → `loop =
   world.make_loop(controller)` → `loop.reset()` → `loop.step()`.

```python
class Controller(ABC):
    def __init__(self, skills: dict[str, Skill], initial: str, decision_interval: int = 5)
    def setup(self, robot) -> None
    def reset_idx(self, envs_idx: torch.Tensor) -> None
    def activate(self, name: str) -> None                  # switch; the newcomer is reset first
    def decide(self, state) -> None                        # abstract: every decision_interval steps
    def update(self, state, dt: float) -> torch.Tensor     # decide (maybe), then the active skill
    ticks: int                                             # property: steps since reset

class PlanningController(Controller):
    def __init__(self, library: SkillLibrary, decision_interval: int = 10)
    def plan(self, state, last: PlanOutcome | None) -> str | None   # grammar text, or None to idle
```

A hand-written controller sets skill attributes (`self.skills["walk"].command[:] = ...`)
and calls `activate`. A planner returns program text such as
`"(avoid @ walk(vx=0.6)).until(moved(4)) >> stand.for(1)"` and reads
`last.succeeded` and `last.trace` to decide the next one.

**Tests to add**: `tests/test_planner.py` drives a planner with fake skills
and a fake `RobotState`, asserting the sequence of programs it emits for a
scripted sequence of outcomes; `tests/test_skill_controller.py` does the
same for a plain `Controller`.

## Add a sensor

Sensors and actuators are the only classes besides the backend that touch
physics handles. There are two families
([Sensors](../api/robot.md#sensors), worked example in
[How to add a sensor](../api/robot.md#how-to-add-a-sensor)):

```python
class StateSensor(ABC):
    def update(self, state: RobotState) -> None: ...      # writes into the shared RobotState

class ExteroceptiveSensor(ABC):
    def read(self) -> torch.Tensor: ...                   # returns its own tensor
```

1. Pick the family. Proprioception writes into `RobotState`; anything that
   perceives the world returns its own tensor.
2. Take the handle it needs in `__init__` and probe optional capabilities
   there, the way `SimContactSensor` sets `available`.
3. For a `StateSensor`, append the instance to `self._state_sensors` in
   `Robot.bind` — after `SimIMU` if it needs `state.base_quat`, which is
   why the IMU runs first. If the quantity has no slot, add a field to
   `RobotState` and its allocation in `RobotState.zeros`.
4. For an `ExteroceptiveSensor` that runs slower than the control loop,
   cache the measurement and expose `tick()` and `reset_idx(envs_idx)` like
   `SectorLidar` and `SimulatedLidar`. The control loop ticks everything in
   `World.sensors`; a task must tick it in `step` and reset it in
   `reset_idx` itself.

A sensor that needs a physics query the handles do not offer adds it to
`Articulation` (or a new handle) in `domo/sim/base.py` as an optional
capability with a `NotImplementedError` default, implements it in the
backend, and probes it at construction. The real-robot driver later
reimplements the same class over a hardware topic and nothing above
`domo/robot` changes.

**Tests to add**: a fake handle returning fixed tensors, asserting the
fields the sensor writes (`tests/test_foundation_robot.py`); for a device
model, the noise, dropout and cadence behaviour
(`tests/test_lidar_models.py`).

## Add a scene kind

A scene kind is a builder in `domo/scenes/` that uses only the `Scene`
contract, plus a branch in the `World`
([Add a scene kind](../api/world-and-services.md#add-a-scene-kind)).

1. Write the builder in `domo/scenes/<name>.py`: it takes a `Scene` and a
   config dataclass, adds its entities **before** `build()`, and does all
   per-env work — randomisation, teleports — in a method called **after**
   the build, as `ObstacleArena.randomise` does. Export it from
   `domo/scenes/__init__.py`.
2. Add a config field on `WorldConfig`, a branch in
   `World._build_environment` returning whatever handle the World should
   keep (or `None`), and the name in `SCENE_KINDS`.
3. If a task should train in it, add the same branch there. Tasks built
   over `World` get it for free through `scene_kind`.

```python
elif cfg.scene_kind == "corridor":
    self.scene.add_ground(cfg.ground_height)
    return Corridor(self.scene, cfg.corridor, spawn_xy=cfg.base_init_pos[:2])
```

**Tests to add**: build the scene against
`tests/test_foundation_scenes.py::RecordingScene` and assert the recorded
`add_*` calls; then a `World` with the fake backend of
`tests/test_foundation_world.py` and `scene_kind="corridor"` to check the
wiring.

## Add a physics backend

`domo/sim/genesis_backend.py` is the only module in the library that
imports `genesis`. A second engine is `domo/sim/<name>_backend.py`
implementing the ABCs of `domo/sim/base.py` and registering a factory. The
full walkthrough is
[Adding a backend](../api/sim.md#adding-a-backend).

| ABC | Required | Optional (the default raises `NotImplementedError`) |
|-----|----------|------------------------------------------------------|
| `PhysicsEngine` | `create_scene(cfg: SimConfig) -> Scene`, the `device` property | |
| `Scene` | `add_ground`, `add_terrain`, `add_mesh`, `add_urdf_prop`, `add_box`, `add_cylinder`, `add_sphere`, `add_articulation(urdf_path, pos, quat_wxyz)`, `add_lidar(articulation, cfg)`, `build(n_envs)`, `step()` | `add_camera(...)` |
| `Articulation` | `dof_indices`, `link_indices`, the `get_base_*` and `get_joint_*` readers, `set_pd_gains`, `set_joint_position_targets`, `set_base_pose`, `set_joint_positions`, `zero_all_velocities` | `get_link_contact_forces`, `set_friction_ratio`, `set_base_mass_shift`, `set_base_com_shift`, `set_pd_gains_scaled` |
| `LidarSensorHandle` | `config`, `read_sector_distances() -> [N, n_horizontal]`, `read_ranges() -> [N, n_vertical, n_horizontal]` | `read_points()` |
| `RigidObject` | `set_position(pos, envs_idx=None)` | |
| `CameraHandle` | `render()` | |

Then register the factory, and `engine="<name>"` in any config selects it:

```python
from domo.sim import register_backend
register_backend("<name>", lambda **kw: NameEngine(**kw))   # kwargs come from create_engine
```

Four rules decide whether the rest of the stack will work unchanged:

* Every tensor is `[N, ...]` on `engine.device`, and `PhysicsEngine.device`
  reports the device after any fallback, not the one that was requested.
* Quaternions are `wxyz`. Convert inside the backend if the engine is
  `xyzw`; nothing above `domo/sim` ever sees another convention.
* Lidar rays are `[N, n_vertical, n_horizontal]`, azimuth 0 at +x running
  counter-clockwise. An engine that returns another layout normalises it
  with `lidar_ranges_to_grid(raw, n_v, n_h, azimuth_major=...)` — Genesis
  returns azimuth-major and would otherwise scramble the sectors
  ([Lidar layout](../api/sim.md#lidar-layout)).
* Entities are added before `build`, moved after it.

`Robot` probes the optional capabilities and the tasks degrade gracefully,
so a backend can ship with the required set only.

**Tests to add**: `tests/test_foundation_world.py` is both the template and
the acceptance test. It stands up a complete `World` — engine → scene →
arena → robot → lidar → build → bind → device-model lidar — on an in-memory
fake engine with no physics in it at all, in about eighty lines. Copy that
file's shape for the new backend, then add a CPU smoke run of
`examples/basic_examples/skill_demo.py` with `engine` switched.

## Add an LLM provider

`domo.llm.LLMClient` is one method, `generate(prompt, temperature=1.0) -> str`.
Providers keep their SDK imports inside the constructor so the core library
stays dependency-free, and `make_llm(provider, **kwargs)` is the single
dispatch point Eureka and the planners use
([Adding a provider](../api/llm-hri.md#adding-a-provider)).

1. Write the client — directly, or as a LangChain chat model.
2. Add a branch to `make_llm`.
3. Add the name to the `ValueError` message `make_llm` raises and to the
   docstring of `SkillLearningRequest.llm`.

```python
# 1a. a direct client (domo/llm/client.py)
class MyClient(LLMClient):
    def __init__(self, model: str = "...", api_key: str | None = None, max_tokens: int = 16384):
        import my_sdk                                  # inside __init__, never at module level
        self._client = my_sdk.Client(api_key or os.environ["MY_API_KEY"])

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        return self._client.complete(prompt, temperature=temperature).text

# 1b. or a LangChain chat model (domo/llm/langchain_client.py)
def my_client(model: str, **kw) -> LangChainClient:
    from langchain_my import ChatMy
    return LangChainClient(ChatMy(model=model, **kw), supports_temperature=True)

# 2. the dispatch branch, in make_llm
if provider == "my":
    return MyClient(**kwargs)
```

Replies go through `extract_code_block` and `extract_json_block`, which
salvage truncated fences. If the provider is a thinking model, size
`max_tokens` so the reasoning does not eat the output budget
([truncated replies](troubleshooting.md#empty-or-truncated-replies-from-gemini)).

**Tests to add**: `tests/test_llm_client.py` covers the parsing helpers and
`make_llm` dispatch with the SDK absent. Test the new branch the same way —
monkeypatch the SDK module, assert the constructor kwargs — and keep the
network out of the suite. `ScriptedClient` stays the provider the
end-to-end tests run against.

## Add an Eureka TaskSpec

Making a task learnable through `domo.eureka` is a registry entry, once the
task exposes a reward-independent success metric
([Adding a task](../api/eureka.md#adding-a-task)). The reference
implementation is `domo/tasks/go2_getup.py`.

1. Give the task `compute_success() -> bool [N]` and `compute_fitness() ->
   [N] in [0, 1]`, both independent of the injected reward, and one
   `episode_outcomes` record per finished episode.
2. Make its config accept `n_envs`, `device`, `headless` and `dr` as
   keyword arguments — that is what `task_overrides` carries.
3. Add the `TaskSpec` to `TASK_REGISTRY` in `domo/eureka/spec.py`, or at
   run time from your own code.

```python
@dataclass(frozen=True)
class TaskSpec:
    module: str                 # "domo.tasks.go2_jump"
    task_class: str             # "Go2JumpTask"
    config_class: str           # "Go2JumpConfig"
    success_description: str    # the FIXED metric, stated exactly as the task computes it
    env_interface: str          # every `task` attribute the generated reward may read
```

The worker imports `module`, builds `task_class(config_class(**task_overrides))`,
installs the generated reward with `set_reward_override`, sanity-steps it
once, trains, and ranks candidates on `episode_outcomes`. Both string
fields are pasted verbatim into the prompt, so a mismatch between them and
the task is the most common cause of rewards that fail in the sanity step.

**Tests to add**: `tests/test_eureka.py` and
`tests/test_eureka_routine.py` run the routine with `ScriptedClient` and a
fake worker. Extend the registry test so the new spec's `module`,
`task_class` and `config_class` import and the config accepts the worker's
override keys.

## House rules

!!! warning "Four ways to break the architecture without noticing"

    **One module may import a physics engine.** That module is
    `domo/sim/<name>_backend.py`. If your extension needs `import genesis`
    anywhere else, the abstraction is missing a method — add it to
    `domo/sim/base.py` as an optional capability instead.

    **Skills are not controllers.** A skill maps state to targets at
    50 Hz and never decides what to do next; a controller decides and
    never computes joint angles. A skill that branches on a mission goal
    belongs in a `PlanningController`.

    **A skill program is never a CLI parameter.** There is no `--program`
    flag and there should not be one: missions are authored inside the
    controller, as `AvoidanceMission` and `twin_demo.py` do, because the
    point is that the *planner* writes them.

    **Config dataclass fields are the checkpoint contract.** Renaming a
    field or changing a default silently changes what every existing
    checkpoint means when it is rebuilt. Add fields with defaults; never
    re-default an existing one.
