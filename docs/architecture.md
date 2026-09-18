# Architecture

DOMO is a layered library. Each layer has one job, imports only from the
layers below it, and exposes an interface the layers above code against.
The point of the discipline is twofold: the physics engine can be swapped
(or replaced by the real robot) without touching behaviour code, and the
supervisor can reason about skills without knowing how they are executed.

## The layer stack

```
┌──────────────────────────────────────────────────────────────────────┐
│ examples/            runnable missions, training scripts, demos      │
├──────────────────────────────────────────────────────────────────────┤
│ domo/world.py        World: engine + scene + robot + sensors, goal-free│
│ domo/dashboard.py    optional live telemetry (twin/eval only)        │
│ domo/policies.py     blessed checkpoints by name                     │
├──────────────────────────────────────────────────────────────────────┤
│ domo/eureka/         Eureka + DrEureka skill-learning routine (M2+M3)│
│ domo/llm/            provider-agnostic LLM clients                   │
│ domo/hri/            voice → commands (M6 precursor)                 │
├──────────────────────────────────────────────────────────────────────┤
│ domo/skills/         cards, grammar, compiler, CompositeSkill,       │
│                      PlanningController (M5)                         │
├──────────────────────────────────────────────────────────────────────┤
│ domo/rl/             PPO, actor-critic, rollout buffer (VecTask only)│
│ domo/tasks/          VecTask environments: rewards, resets, success  │
├──────────────────────────────────────────────────────────────────────┤
│ domo/control/        Skill / Controller / ControlLoop, CPG, leg IK,  │
│                      SLAM, navigation — pure torch                   │
│ domo/scenes/         arena and ReplicaCAD scene builders             │
├──────────────────────────────────────────────────────────────────────┤
│ domo/robot/          RobotSpec, RobotState, sensors, actuators,      │
│                      lidar device models, domain randomisation       │
├──────────────────────────────────────────────────────────────────────┤
│ domo/sim/            abstract engine interfaces + Genesis backend    │
├──────────────────────────────────────────────────────────────────────┤
│ domo/utils/          quaternion math (wxyz)                          │
└──────────────────────────────────────────────────────────────────────┘
```

### Dependency rules

* `domo/sim/genesis_backend.py` is the only module that imports `genesis`.
  New engines implement the ABCs in `domo/sim/base.py` and register with
  `domo.sim.register_backend(name, factory)`.
* Physics handles (`Articulation`, `LidarSensorHandle`, `CameraHandle`) are
  touched only by the sensor and actuator classes in `domo/robot/`. Tasks,
  controllers and skills see `RobotState` tensors and call `Robot` methods.
* `domo/control/` and `domo/skills/` never import an engine. They run
  unchanged on a machine without Genesis and, later, on the real robot.
* `domo/rl/` knows only the `VecTask` API (`step`, `reset`, buffers).
* `domo/eureka/` drives tasks through the same API plus two hooks every task
  offers: `set_reward_override` (inject a generated reward) and
  `compute_success` (the fixed metric generated rewards cannot game).

## Two runtimes on one robot

The same `Robot`, sensors and skills serve two different loops.

### The digital twin: World + ControlLoop

```
World(cfg) ──builds──▶ engine, scene, Robot(+sensors), optional camera
      │
      └─ make_loop(controller) ──▶ ControlLoop
                                      │  every tick:
                                      │   1. sensors.tick()        read physics
                                      │   2. state = robot.read()  RobotState [N,...]
                                      │   3. controller.update()   picks/parametrises skills
                                      │   4. skill(state) → targets joint targets [N,12]
                                      │   5. command_filter()      reserved for M7 safety
                                      │   6. actuators.apply()     PD targets to physics
                                      │   7. engine.step()
```

There is no reward and no episode. A `Controller` decides what the robot
does; a `PlanningController` does so by writing a program in the skill
grammar and letting the library compile and run it. Missions end because the
controller says so, not because an environment terminates.

### Skill learning: VecTask + PPO

```
Go2*Config ──▶ VecTask (scene + Robot + rewards + resets + success metric)
                   │
                   └─ PPO(task).learn()  ──▶ runs/<task>/checkpoint_*.pt
                                                 │
                                   ActorCritic.from_state_dict ──▶ policy_fn
                                                 │
                                    wrapped as a Skill (e.g. CPGLocomotionSkill)
                                    and registered as a card in the library
```

A task is a temporary lens: it owns a scene built with the same builders
the World uses, adds rewards and termination, and steps thousands of
environments in parallel. The trained network is turned into a `Skill` and,
from then on, only the twin runtime sees it.

## The control hierarchy

| Level | Class | Rate | Responsibility |
|-------|-------|------|----------------|
| primitive | `Skill` | 50 Hz | `state → joint targets`. Motor skills (`StandSkill`, `CPGLocomotionSkill`, learned joint skills) write targets; command skills (`AvoidSkill`, `TrajectoryTrackingSkill`, `SlamSkill`) write or modify the velocity command of the motor skill under them |
| orchestration | `Controller` | 1–10 Hz | activates skills, sets their parameters, decides when a leg of a mission is done. Researchers write these by hand; the LLM writes them through `PlanningController.plan()` |
| metronome | `ControlLoop` | sim step | ticks sensors, controller, skills, the safety filter, actuators and the engine in that order; identical code for sim and real |

Skills are lower level than controllers. This is the vocabulary of the whole
project; see [conventions.md](conventions.md).

## The skill grammar (M5)

Cards describe skills in natural language and types; programs compose cards:

| Syntax | Meaning |
|--------|---------|
| `avoid @ walk(vx=0.6)` | layering: the command skill on the left drives the motor skill on the right. Type-checked: channels must match |
| `A >> B` | sequence: B starts when A succeeds |
| `A \| B` | fallback: B runs if A fails |
| `.for(T)` `.until(cond)` `.repeat(n)` | modifiers on any node |

`library.compile(text)` type-checks and returns a `CompositeSkill`, which is
itself a `Skill` and keeps an execution `trace` for the supervisor to read.
`library.describe()` renders the catalogue and the grammar as the planning
prompt. The full reference is in [api/skills.md](api/skills.md).

## The learning loop (M2 + M3)

`domo.eureka.learn_skill(SkillLearningRequest)` runs Eureka: the LLM writes
candidate reward functions, each candidate trains in its own subprocess (one
Genesis per process), candidates are ranked on the task's fixed success
metric and a dense fitness, a reflection prompt feeds the next iteration.
DrEureka follows: single-parameter sweeps find the feasible domain-
randomisation bounds, the LLM proposes several DR configurations inside
them, all are trained and the best is kept. The result is a `LearnedSkill`
ready to wrap as a joint skill. Details in [api/eureka.md](api/eureka.md).

## Sim ↔ real equivalence

| Simulation | Real robot (planned) |
|------------|----------------------|
| `GenesisEngine`, `GenesisScene` | no engine; the room is the scene |
| `Articulation` handle | DDS/ROS joint state and command topics |
| `SimulatedLidar` over `LidarSensorHandle` | Hesai XT16 driver producing the same `[N, n_sectors]` and point tensors |
| `PDActuator.apply(targets)` | low-level joint command with the same gains |
| `ControlLoop` at sim dt | `ControlLoop` on a wall-clock timer |
| `RobotState` | identical dataclass filled from hardware |

Because tasks, skills and controllers only ever see `RobotState` and `Robot`
methods, they cannot tell which side of the table they are on.
