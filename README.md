# DOMO — Developmental Orchestration of Motor-skill Onset

An LLM acts as a lifelong *developmental supervisor* for a physical robot
(Unitree Go2): it watches the robot, detects missing skills, specifies them,
trains them via RL in a digital twin, transfers them to hardware, and grows a
compounding skill library.

This repository contains the `domo` library (the system being built, layer by
layer) and `scripts/` (frozen standalone experiments the library is distilled
from — do not import from them).

**Documentation:** start with [docs/getting-started.md](docs/getting-started.md);
the full guide (architecture, conventions, running, examples, API reference,
extending, troubleshooting) is indexed in [docs/README.md](docs/README.md).

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
├── llm/        Provider-agnostic LLM clients behind one interface
│               (generate → str), selected by make_llm(provider): "gemini"
│               (direct google-genai), "vllm" (local model via LangChain +
│               vLLM OpenAI-compatible server), "openai", "gemini-lc"
│               (Gemini via LangChain), "scripted" (offline/tests). LangChain
│               is lazy — the core needs no LLM deps.
├── eureka/     The Eureka + DrEureka skill-learning routine (M2+M3),
│               faithful to Yu et al. 2024. learn_skill(SkillLearningRequest):
│               (1) SAFETY-REGULARIZED reward generation (l_task + l_safety) →
│               candidate training in worker subprocesses → rank on the task's
│               FIXED success metric → reward reflection → iterate;
│               (2) DrEureka: Reward-Aware Physics Prior (single-param sweeps →
│               feasible bounds) → LLM samples m independent DomainRandomization
│               configs (clamped to bounds) → train ALL, keep best. Orchestrated
│               with LangGraph (graph.py; use_graph=False for the imperative
│               driver). Callable from the twin, a CLI, or the real-robot
│               supervisor. Example: examples/eureka/eureka_getup.py.
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
pip install -e '.[dev]'

# Engine-free tests (a few seconds)
pytest tests/

# The digital twin on CPU: a compiled skill program drives the robot around a square
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu --steps 500

# Train the CPG gait everything else uses (GPU), then evaluate it
python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless
python examples/locomotion/go2_cpg_rl.py --eval runs/go2_cpg/checkpoint_final.pt --vx 0.5

# Joint-space locomotion baseline (GPU)
python main.py --n-envs 4096 --device cuda --headless
python main.py --eval runs/go2_walk/checkpoint_final.pt

# Composed-skill evaluation (avoidance ON TOP OF locomotion): the mission
# is authored inside a PlanningController in the example, never a CLI flag
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt --cpg-checkpoint policies/walk.pt --headless

# The digital twin: goal-free World + re-planning mission controller
python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt
python examples/twin/twin_demo.py --walk policies/walk.pt --catalog   # LLM prompt

# SLAM in an arena with the live web dashboard
python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080
```

Blessed checkpoints live in `policies/` (`walk.pt`, `avoid.pt`; git-ignored) and
are resolved by name through `domo.policies`; see `policies/README.md` to
promote a new one. Every example is described in
[docs/examples.md](docs/examples.md).

## Development

```bash
pytest tests/                                   # engine-free, ~10 s
ruff check domo examples tests main.py          # lint rules in pyproject.toml
```

Conventions and layering rules: [docs/conventions.md](docs/conventions.md).
Genesis quirks and known errors: [docs/troubleshooting.md](docs/troubleshooting.md).

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
