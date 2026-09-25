# Glossary

Every term this project uses in a narrow, load-bearing sense, with the page
that owns it. When two words look interchangeable here — skill and
controller, task and world — they are not.

Backend
:   A physics engine implementation behind the abstract interfaces in
    `domo/sim/base.py`. Genesis is the only one today, and
    `domo/sim/genesis_backend.py` is the only module in the code base allowed
    to import it. New engines register with `domo.sim.register_backend`.
    [domo.sim](../api/sim.md)

Card
:   The description of a skill that planners read: natural language plus
    typed parameters, termination, constraints and safety.
    `library.describe()` renders the whole catalogue and the grammar as the
    planning prompt. [domo.skills](../api/skills.md)

Command skill
:   A skill that writes or modifies the velocity command of the motor skill
    underneath it instead of writing joint targets — `LidarAvoidanceSkill`,
    `TrajectoryTrackingSkill` (`goto`), `SlamSkill`. It is the left-hand side
    of an `@` layering.
    [Control hierarchy](../concepts/control-hierarchy.md)

Composite skill
:   What `library.compile(text)` returns: a type-checked composition that is
    itself a `Skill`, and that keeps an execution `trace` for the supervisor
    to read back. [The skill grammar](../concepts/grammar.md)

Controller
:   Orchestration code at 1–10 Hz: it activates skills, sets their parameters
    and decides when a leg of a mission is done. This is the slot
    LLM-written behaviour targets. Never call a skill a controller.
    [Control hierarchy](../concepts/control-hierarchy.md)

ControlLoop
:   The metronome. It ticks sensors, the controller, the active skill, the
    reserved safety filter, the actuators and the engine, in that order and
    at the simulation step. `SimControlLoop` and `RealControlLoop` run the
    same sequence. [domo.control](../api/control.md#control-loops)

CPG
:   Central pattern generator: the oscillator bank of Bellegarda and
    Ijspeert's CPG-RL. The learned policy writes oscillator parameters, and
    the oscillators plus analytic leg IK produce the joint targets. This is
    the gait `policies/walk.pt` holds. [domo.control](../api/control.md#cpg)

Digital twin
:   The goal-free runtime: a `World`, a `ControlLoop` and whatever controller
    governs them. It is the entry point of the system — missions end because
    the controller says so, not because an episode terminates.
    [The project](../concepts/project.md#the-twin-first-principle)

Domain randomisation
:   Randomising physics and robot parameters during training so the policy
    survives the sim-to-real gap (M4). On Genesis, friction and mass are
    randomised per environment but PD gains only per reset group.
    [domo.robot](../api/robot.md)

DrEureka
:   The second half of the learning routine: single-parameter sweeps find the
    feasible randomisation bounds (the Reward-Aware Physics Prior), the LLM
    samples several domain-randomisation configurations inside them, all are
    trained, and the best is kept. [domo.eureka](../api/eureka.md)

Eureka
:   The LLM-driven reward search (M2 + M3). The model writes candidate reward
    functions, each trains in its own subprocess, candidates are ranked on
    the task's fixed success metric, and a reflection prompt feeds the next
    iteration. [domo.eureka](../api/eureka.md)

Fitness
:   A task's dense progress proxy in `[0, 1]` (`compute_fitness`),
    time-averaged per episode. It exists to break ties between Eureka
    candidates that all score zero success; any success beats any fitness.
    [domo.eureka](../api/eureka.md)

M1–M8
:   The eight research modules the system is organised in: gap detection,
    specification, training, sim-to-real, skill library, human interaction,
    safety, and the lifelong loop. M1, M2, M3 and M5 are implemented today.
    [The project](../concepts/project.md#the-research-roadmap)

Motor skill
:   A skill that writes joint targets directly — `StandSkill`,
    `CPGLocomotionSkill`, `LearnedJointSkill`. It is the right-hand side of
    an `@` layering, and the bottom of every program.
    [Control hierarchy](../concepts/control-hierarchy.md)

PlanningController
:   A controller whose behaviour is a program text produced by
    `plan(state, last_outcome)` and compiled by the skill library. It is the
    seat reserved for the LLM: programs are authored at runtime, never passed
    in as a command-line flag. [domo.skills](../api/skills.md)

Program
:   A composition of cards written in the skill grammar, for example
    `avoid @ goto(x=2, y=1) @ walk`. `@` layers, `>>` sequences, `|` falls
    back, and `.for(T)`, `.until(cond)`, `.repeat(n)` modify any node.
    [The skill grammar](../concepts/grammar.md)

Sector lidar
:   The form policies consume a lidar in: the raw beams pooled into
    per-sector minimum distances, `[N, n_sectors]` — 36 sectors for the
    avoidance net. The number of azimuth samples raycast is a fidelity knob
    and does not change the observation size. [domo.robot](../api/robot.md)

Skill
:   A motor primitive: `RobotState` in, joint targets out, at 50 Hz. It is
    the unit the library stores, describes with a card and composes. Skills
    sit *below* controllers; that ordering is the whole control hierarchy.
    [domo.control](../api/control.md#skills)

Skill library
:   The M5 layer: the catalogue of cards, the composition grammar, the
    type-checking compiler and the planners on top of them.
    `make_go2_library` builds the Go2 catalogue from policies you pass;
    `stable_go2_library` builds it from the registry.
    [domo.skills](../api/skills.md)

SLAM skill
:   `SlamSkill`: a perception layer that folds the lidar and the pose into a
    log-odds occupancy grid and a 3-D point cloud every tick, and writes zero
    velocity. It is held at loop level rather than composed, because
    re-entering a composed node would reset the map.
    [domo.control](../api/control.md#slam)

Stable policy
:   A blessed checkpoint named in `domo.policies.STABLE` (`walk`, `avoid`)
    with a copy under `policies/`. Skills resolve the symbolic name, never a
    path inside `runs/`. The `.pt` files themselves are git-ignored.
    [Running](../guides/running.md#stable-policies)

Success metric
:   A task's fixed, hand-written criterion (`compute_success`) that a
    generated reward cannot game. It is the first ranking key for Eureka
    candidates, and it never changes between iterations.
    [domo.tasks](../api/tasks.md)

Task (VecTask)
:   A vectorised RL environment — scene, robot, rewards, resets, success
    metric — stepped legged-gym style across thousands of environments. It is
    a temporary reward-bearing lens over the same scene builders the World
    uses, and never the entry point. [domo.tasks](../api/tasks.md)

World
:   Engine, scene, robot and sensors, built and bound, with no goal, no
    reward and no episode. `World(WorldConfig(...))` fixes the construction
    order, and `world.make_loop(controller)` produces the loop that runs it.
    [domo.world](../api/world-and-services.md)
