# Examples

Twelve runnable scripts under `examples/`, one page each. Every page records
what library pieces the script exercises, the exact command lines and their
flags, what the console prints, what gets saved, how long it takes on CPU and
GPU, and what it does not do. The script itself is the reference: each opens
with a module docstring saying the same things more briefly, and each is
organised as a few `build_*` / `run*` / `report` functions plus `parse_args()`
and a short `main()`, so these pages can be read against the source.

## Start with these three

<div class="grid cards" markdown>

-   :material-shoe-print:{ .lg } __Skill demo__

    ---

    The whole control workflow in 180 lines: engine, scene, robot, policy,
    a compiled grammar program, a control loop. Runs on a laptop in a minute.

    [:octicons-arrow-right-24: Skill demo](skill-demo.md)

-   :material-robot-outline:{ .lg } __Digital twin__

    ---

    The same runtime with no task, governed by a planner that writes
    programs, reads outcomes and concedes a skill gap when it runs out.

    [:octicons-arrow-right-24: Digital twin](twin-demo.md)

-   :material-map-outline:{ .lg } __SLAM in an arena__

    ---

    A passive perception layer riding on a patrol: occupancy grid, 3D point
    cloud, a live web dashboard if you want one.

    [:octicons-arrow-right-24: SLAM demo](slam-demo.md)

</div>

## All twelve

| Example | What it shows | GPU? | Assets or keys? | CPU runtime |
|---------|---------------|------|-----------------|-------------|
| [Skill demo](skill-demo.md) | `Skill` / `Controller` / `ControlLoop`; a compiled square route against an open-loop contrast | no | `policies/walk.pt` | ~1.5 min |
| [Digital twin](twin-demo.md) | goal-free `World` + a `PlanningController` that re-plans on outcomes and concedes a skill gap | no | `walk.pt`, optionally `avoid.pt` | ~1–2.5 min |
| [Click to walk](interactive-nav.md) | a click on the dashboard map becomes a goal; the planner authors `avoid @ goto(x, y) @ walk` | no | `walk.pt`, `avoid.pt` (via the registry) | build ~30 s, then interactive |
| [Locomotion (CPG-RL)](locomotion.md) | the PPO + CPG gait every other example stands on; train, evaluate, resume | training only | `walk.pt` for `--eval` | eval ~1.5 min |
| [Obstacle avoidance](avoidance.md) | a 36-sector lidar net writing velocity corrections over a frozen walk, evaluated twin-style | training only | `walk.pt`, `avoid.pt` | eval ~110 s |
| [Avoidance in a house](avoid-house.md) | the same architecture inside a ReplicaCAD apartment | training only | your own checkpoint (assets are vendored) | build alone ~2 min |
| [LLM level designer](gemini-design-nav.md) | an LLM designs one navigation program from an ASCII map; run blind | no | key only for a real provider | 2–3 min |
| [Waypoint navigation](evaluate-nav.md) | the legacy P-controller, stdin intervention, natural-language commands | no | key only for `--gemini` | ~1 min (forward) |
| [SLAM in an arena](slam-demo.md) | `SlamSkill` as a passive observer on a `goto` patrol; occupancy grid + 3D cloud | no | `walk.pt` (via the registry) | 3–4 min |
| [SLAM in an apartment](slam-house.md) | the same layer in ReplicaCAD, driven by a blind wanderer | no | `walk.pt`, `avoid.pt` (assets are vendored) | ~7 min |
| [Voice commands](voice.md) | microphone → Whisper → Gemini → `(vx, vy, vyaw)` into the walk policy | no | microphone, `voice` extra, `GEMINI_API_KEY` | runs until Ctrl+C |
| [Eureka get-up](eureka-getup.md) | the LLM writes rewards, trains candidates, ranks them, deploys the winner | real runs only | none with `--llm scripted` | smoke ~minutes |

`examples/slam/slam_viz.py` is the thirteenth file and not a command-line tool:
it holds the matplotlib and dashboard plumbing the two SLAM examples share,
and is kept out of `domo/` because it imports matplotlib. Its helpers are
tabulated on the [SLAM demo](slam-demo.md#the-slam_viz-helpers) page.

## Conventions

Run every example **from the repository root**:

```bash
python examples/<area>/<name>.py [checkpoint] [flags]
```

The scripts import their siblings by plain module name (`from slam_viz import
…`, `from go2_cpg_rl_lidar import …`), which works because Python puts the
script's own directory on `sys.path`. Run them as scripts, not with
`python -m`.

### Flags you will pass everywhere

| Flag | Where | Meaning |
|------|-------|---------|
| `--device cuda` | every example except `evaluate_nav.py` and the voice example | torch/Genesis device. `cuda` is the default everywhere; `domo.checkpoints.pick_device` falls back to `cpu` when CUDA is unavailable, so the defaults work on a laptop. Pass `--device cpu` to force CPU on a CUDA machine |
| `--headless` | every example | do not open the Genesis viewer. The SLAM examples' `--live` matplotlib window and the web dashboard are independent of it and work headless |
| `--steps N` | twin, navigation and SLAM demos | control steps at 50 Hz (`dt = 0.02 s`), so 3000 steps is one minute of simulated time |

!!! note "Five evaluation paths ignore `--device`"

    The `--eval` branches of `go2_cpg_rl.py`, `go2_cpg_rl_lidar.py` and
    `go2_cpg_rl_avoid_house.py`, plus `evaluate_nav.py` and the voice
    example, all call `pick_device("cuda")` and take CUDA when it is
    present, CPU otherwise — exactly as the frozen scripts they replicate
    did. On a CPU-only machine that is what you want; on a CUDA machine you
    cannot force those paths onto CPU.

### Checkpoints

Everything that walks needs a trained CPG gait; everything that dodges needs
the avoidance net on top of it. The blessed copies live in `policies/` and
are resolved by symbolic name through
[`domo.policies.stable_policy`](../api/world-and-services.md#domopolicies):

| Name | File | What it is |
|------|------|------------|
| `walk` | `policies/walk.pt` | the CPG-RL velocity-tracking gait (promoted from `runs/go2_cpg/checkpoint_final_coupled.pt`) |
| `avoid` | `policies/avoid.pt` | the 36-sector lidar avoidance net writing velocity corrections (promoted from `runs/go2_cpg/checkpoint_final_avoid.pt`) |

!!! warning "`policies/*.pt` are git-ignored"

    `policies/` is a curated local cache; only the registry
    (`domo/policies.py`) and `policies/README.md` are versioned. On a fresh
    clone, copy the two files from a training run or a colleague — without
    them nothing walks. See
    [running](../guides/running.md#stable-policies).

Only the two SLAM examples and [click to walk](interactive-nav.md) default to
the stable names. Every other example takes explicit paths, so the commands on
these pages pass `policies/walk.pt` and `policies/avoid.pt` by hand. `domo.checkpoints` loads both the legacy
script format (`{"config", "model_state"}`) and the library format
(`{"ppo_config", "extra": {"task_config"}}`), rebuilding the network from the
weight shapes.

### CPU or GPU

| Stage | CPU (laptop) | CUDA GPU |
|-------|--------------|----------|
| Genesis build, arena or flat ground | about 30 s | a few seconds |
| Genesis build, ReplicaCAD apartment | about 2 min | tens of seconds |
| stepping one environment | 50–100 control steps/s | 10–100× faster |
| PPO training with thousands of environments | impractical | hours for a gait, see [running](../guides/running.md) |

All demonstrations, evaluations and smoke runs work on CPU with one
environment, and every page starts with the CPU variant. The training entry
points — `go2_cpg_rl.py`, the two avoidance scripts and the learning path of
`eureka_getup.py` — need a GPU for anything beyond a smoke test. Training
also needs the `rl` extra (`tensorboard`), which `PPOTrainer` imports
unconditionally.

## Frozen scripts and their replicas

`scripts/` is frozen: the original experiment scripts are kept as they were
and are not maintained. Five examples are library replicas of them, with the
same CLI where it made sense.

| Frozen script | Library replica | Differences |
|---------------|-----------------|-------------|
| `scripts/house_scene/go2_cpg_rl.py` | [`examples/locomotion/go2_cpg_rl.py`](locomotion.md) | same flags; training and evaluation go through `Go2CPGWalkTask` and `PPOTrainer` |
| `scripts/house_scene/go2_cpg_rl_lidar.py` | [`examples/avoidance/go2_cpg_rl_lidar.py`](avoidance.md) | same flags plus `--lidar`, `--lidar-azimuth`, `--draw-lidar`; evaluation is a `PlanningController` mission in the goal-free `World` |
| `scripts/house_scene/go2_cpg_rl_avoid_house.py` | [`examples/avoidance/go2_cpg_rl_avoid_house.py`](avoid-house.md) | as above, `scene_kind="replica"` |
| `scripts/house_scene/evaluate_nav.py` | [`examples/navigation/evaluate_nav.py`](evaluate-nav.md) | the P-controller moved to `domo.control.navigation`; the script keeps the sim wiring, stdin intervention and the demos |
| `scripts/house_scene/go2_cpg_rl_voice.py` | [`examples/hri/go2_cpg_rl_voice.py`](voice.md) | only the interactive part; the duplicated trainer is gone (use `go2_cpg_rl.py`) |

[Skill demo](skill-demo.md), [twin demo](twin-demo.md),
[click to walk](interactive-nav.md),
[LLM level designer](gemini-design-nav.md), the two SLAM examples
([arena](slam-demo.md), [house](slam-house.md)) and
[Eureka get-up](eureka-getup.md) are library-native and have no frozen
counterpart.

## How the folders map onto the layers

```
examples/
├── basic_examples/skill_demo.py         Skill / Controller / ControlLoop, a compiled square route
├── twin/twin_demo.py                    the goal-free World governed by a PlanningController
├── twin/interactive_nav.py              click the dashboard map, the planner walks the robot there
├── locomotion/go2_cpg_rl.py             train / evaluate / resume the CPG-RL walk (PPO)
├── avoidance/go2_cpg_rl_lidar.py        train / evaluate lidar avoidance in a walled arena
├── avoidance/go2_cpg_rl_avoid_house.py  the same, inside a ReplicaCAD apartment
├── navigation/gemini_design_nav.py      an LLM designs one navigation program, run blind
├── navigation/evaluate_nav.py           waypoint navigation with stdin intervention and NL commands
├── slam/slam_demo.py                    passive SLAM during a perimeter patrol (Hesai XT16)
├── slam/slam_replica_house.py           passive SLAM while wandering a ReplicaCAD house
├── slam/slam_viz.py                     shared plotting / dashboard helpers for the SLAM examples
├── hri/go2_cpg_rl_voice.py              microphone → Whisper → Gemini → velocity command
└── eureka/eureka_getup.py               Eureka / DrEureka learns a get-up skill, then a twin demo
```

The folders follow the layers in
[architecture](../concepts/architecture.md). `basic_examples` and `twin` are
the digital-twin runtime (`World`, `ControlLoop`, skills and controllers);
`locomotion` and `avoidance` are the skill-learning runtime (`VecTask` + PPO)
with a twin-style evaluation; `navigation`, `slam` and `hri` sit on the twin
runtime and add a planner, a perception layer or a voice channel; `eureka` is
the M2/M3 learning loop.
