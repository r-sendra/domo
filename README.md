# DOMO — Developmental Orchestration of Motor-skill Onset

An LLM acts as a lifelong *developmental supervisor* for a physical robot
(Unitree Go2): it watches the robot, detects missing skills, specifies them,
trains them via RL in a digital twin, transfers them to hardware, and grows a
compounding skill library.

This repository contains the `domo` library (the system being built, layer by
layer) and `scripts/` (frozen standalone experiments the library is distilled
from — do not import from them).

## Library architecture

```
domo/
├── world.py    THE ENTRY POINT: the digital-twin runtime. A World is a
│               robot spawned in an environment with its sensors — no goal,
│               no rewards, no episodes. The resident robot lives here under
│               whatever Controller governs it. RL is invoked the other way
│               around: when a skill must be learned, a training task is
│               instantiated, trained, and the new skill joins the library.
├── sim/        Physics abstraction. ABCs in sim/base.py (PhysicsEngine, Scene,
│               Articulation, LidarSensorHandle) + backend factory. Genesis is
│               the only backend today (sim/genesis_backend.py) and the ONLY
│               module allowed to import `genesis`. New engines register via
│               domo.sim.register_backend().
├── robot/      RobotSpec (static robot description; GO2 provided), RobotState
│               (per-step snapshot), sensors and actuators — the only
│               components that touch the physics handles. Real-robot
│               deployments reimplement the same sensor/actuator interfaces
│               over DDS/ROS; tasks and controllers cannot tell the difference.
│               lidar_models.py simulates commercial devices — hesai_xt16()
│               is the unit on the real robot (16ch ±15°, 10 Hz, blind zone,
│               σ=1 cm noise, dropout); azimuth samples are a sim-fidelity
│               knob (train at 180, validate at XT16_FULL_AZIMUTH).
├── control/    Engine-free control stack, pure torch. The three-layer
│               hierarchy: Skill (primitives: StandSkill, CPGLocomotionSkill —
│               state → joint targets @ 50 Hz; the M5 library interface),
│               Controller (programmable orchestration: selects skills, sets
│               parameters @ 1–10 Hz; the slot LLM-generated behaviour code
│               targets), ControlLoop (Sim/Real metronome; command_filter is
│               the reserved M7 safety choke point). Plus the building
│               blocks: leg IK/FK, CPG oscillators, PositionController.
│               See examples/basic_examples/skill_demo.py for the workflow.
├── scenes/     Scene builders: ReplicaCAD/Habitat loader + the obstacle
│               arena (shared by RL training and composed-skill evaluation).
├── skills/     The M5 layer: SkillCards (NL description, params, termination,
│               constraints, safety), a symbolic composition grammar —
│                   avoid @ walk(vx=0.6)   layering ("on top of")
│                   A >> B                 sequence     A | B   fallback
│                   .for(T) .until(cond) .repeat(n)     modifiers
│               — a type-checking compiler (command-vs-motor interfaces,
│               param ranges, channel matching), and CompositeSkill: a
│               compiled program IS a Skill, with an execution trace the
│               LLM supervisor reads. library.describe() renders the whole
│               catalog + grammar as the planning prompt.
│               PlanningController: programs are AUTHORED AT RUNTIME inside
│               the controller via plan(state, last_outcome) → program text
│               — the seat for LLM/HRL planners. Never a CLI parameter.
├── tasks/      Vectorised task environments (VecTask API: legged-gym style
│               5-tuple step). Tasks compose scene + robot + rewards; they
│               import domo.sim interfaces, never an engine.
│               go2_walk (joint-PD velocity tracking), go2_cpg_walk (CPG-RL),
│               go2_avoid (frozen-locomotion + lidar avoidance; arena or house).
├── rl/         Optional learning layer: PPO, actor-critic, rollout buffer.
│               Takes any VecTask; imports no physics. ActorCritic.from_state_dict
│               rebuilds any checkpoint (incl. legacy script ones) from weight shapes.
├── hri/        Human-robot interaction (M6 precursor): voice → velocity
│               commands (whisper + Gemini, lazy deps).
├── llm/        LLM clients (GeminiClient — free-tier 2.5-flash — and
│               ScriptedClient for offline runs). Supervisor logic (M1) upcoming.
├── eureka/     The Eureka + DrEureka skill-learning routine (M2+M3):
│               learn_skill(SkillLearningRequest) → LLM reward generation →
│               candidate training in worker subprocesses → ranking on the
│               task's FIXED success metric → reward reflection → iterate;
│               then optionally the DrEureka stage: reward-aware physics
│               prior sweep → LLM-proposed DomainRandomization → robust
│               retrain. Callable from the twin, a CLI, or the future
│               real-robot supervisor. Example: examples/eureka/
│               eureka_getup.py (closes the twin demo's get-up skill gap).
└── utils/      Pure math (quaternion ops, wxyz convention).
```

## Script → library replica map

Originals in `scripts/house_scene/` stay frozen; their library replicas live
in thematic `examples/` folders (same CLIs) on top of `domo/`:

| Original script            | Replica entry point + library home |
|----------------------------|------------------------------------|
| go2_cpg_rl.py              | examples/locomotion/go2_cpg_rl.py → tasks/go2_cpg_walk.py + control/cpg.py |
| go2_cpg_rl_lidar.py        | examples/avoidance/go2_cpg_rl_lidar.py → tasks/go2_avoid.py (scene_kind="arena") |
| go2_cpg_rl_avoid_house.py  | examples/avoidance/go2_cpg_rl_avoid_house.py → tasks/go2_avoid.py ("replica") + scenes/replica.py |
| go2_cpg_rl_voice.py        | examples/hri/go2_cpg_rl_voice.py → hri/voice.py (training: use go2_cpg_rl.py) |
| voice_commander.py         | domo/hri/voice.py |
| evaluate_nav.py            | examples/navigation/evaluate_nav.py → control/navigation.py |
| rl_walk_house.py, replica_rl_locomotion.py, lidar_teleop.py, house_raycast_example.py, test.py | superseded experiments / Genesis API demos — not ported |

Both new-format and legacy script checkpoints load everywhere
(`ActorCritic.from_state_dict` + `domo/checkpoints.py`).

Dependency rule: `utils ← {sim, robot, control, tasks}`, `sim ← robot ← tasks`,
`control ← tasks`, and `rl` only consumes the VecTask API. Nothing outside
`domo/sim/` imports a physics package.

### Conventions

* Quaternions are **wxyz (scalar-first)** everywhere (`domo.utils.rotations`).
* All batched state is `torch.Tensor [n_envs, ...]` on the sim device.
* Joint order for the Go2: `[FR, FL, RR, RL] × [hip, thigh, calf]`.

## Quick start

```bash
conda activate domo
pip install -e .

# Train Go2 velocity-tracking locomotion
python main.py --n-envs 4096 --device cuda --headless

# Evaluate / resume
python main.py --eval   runs/go2_walk/checkpoint_final.pt
python main.py --resume runs/go2_walk/checkpoint_step_xxx.pt

# Tests (pure math + engine-free layers)
pytest tests/

# Composed-skill evaluation (avoidance ON TOP OF locomotion): the mission
# is authored inside a PlanningController in the example — not a CLI flag
python examples/avoidance/go2_cpg_rl_lidar.py \
    --eval runs/go2_avoidance/checkpoint_final.pt \
    --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt

# The digital twin: goal-free World + re-planning mission controller
python examples/twin/twin_demo.py --walk CKPT --avoid CKPT
python examples/twin/twin_demo.py --walk CKPT --catalog   # LLM prompt
```

## Adding a physics backend

Implement the ABCs in `domo/sim/base.py` in a new module, then:

```python
from domo.sim import register_backend
register_backend("mujoco", lambda **kw: MujocoEngine(**kw))
```

## Adding a task

Subclass `domo.tasks.VecTask`, compose a scene and a `Robot`, register reward
terms with `register_rewards({...})` (methods named `_reward_<name>`), and
implement `step` / `reset` / `reset_idx`. See `domo/tasks/go2_walk.py`.
