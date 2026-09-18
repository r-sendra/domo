# Examples

Every runnable script under `examples/` is documented here: what library
pieces it exercises, how it is structured, the exact command lines and flags,
what it prints and saves, how long it takes on CPU and GPU, and what it does
not do. The reference for each section is the script itself; every example
opens with a module docstring that says the same things more briefly, and
each is organised as a few named `build_*` / `run*` / `report` functions plus
a `parse_args()` and a short `main()`, so the sections below can be read
against the source.

## How the examples are organised

```
examples/
├── basic_examples/skill_demo.py         Skill / Controller / ControlLoop, a compiled square route
├── twin/twin_demo.py                    the goal-free World governed by a PlanningController
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

The folders map onto the layers in [architecture.md](architecture.md):
`basic_examples` and `twin` are the digital-twin runtime (`World`,
`ControlLoop`, skills and controllers); `locomotion` and `avoidance` are the
skill-learning runtime (`VecTask` + PPO) with a twin-style evaluation;
`navigation`, `slam` and `hri` sit on the twin runtime and add a planner, a
perception layer or a voice channel; `eureka` is the M2/M3 learning loop.

Run every example from the repository root:

```bash
python examples/<area>/<name>.py [checkpoint] [flags]
```

The scripts import their siblings by plain module name (`from slam_viz
import …`, `from go2_cpg_rl_lidar import …`), which works because Python puts
the script's own directory on `sys.path`. Run them as scripts, not with
`python -m`.

### Common flags

| Flag | Where | Meaning |
|------|-------|---------|
| `--device cuda` | every example except `evaluate_nav.py` | torch/Genesis device. `cuda` is the default everywhere; `domo.checkpoints.pick_device` silently falls back to `cpu` when CUDA is unavailable, so the defaults work on a laptop. Pass `--device cpu` to force CPU on a CUDA machine |
| `--headless` | every example | do not open the Genesis viewer. The SLAM examples' `--live` matplotlib window and the web dashboard are independent of it and work headless |
| `--steps N` | twin, navigation, SLAM demos | control steps at 50 Hz (`dt = 0.02 s`), so 3000 steps is one minute of simulated time |

Three evaluation paths ignore `--device`, exactly as the frozen scripts they
replicate did: the `--eval` branches of `go2_cpg_rl.py`,
`go2_cpg_rl_lidar.py` and `go2_cpg_rl_avoid_house.py`, the voice example and
`evaluate_nav.py` all call `pick_device("cuda")` and take CUDA when it is
present, CPU otherwise. On a CPU-only machine that is what you want; on a
CUDA machine you cannot force those paths onto CPU.

### Checkpoints

Everything that walks needs a trained CPG gait; everything that dodges needs
the avoidance net on top of it. The blessed copies live in `policies/` and
are resolved by symbolic name through `domo.policies.stable_policy`:

| Name | File | What it is |
|------|------|------------|
| `walk` | `policies/walk.pt` | the CPG-RL velocity-tracking gait (promoted from `runs/go2_cpg/checkpoint_final_coupled.pt`) |
| `avoid` | `policies/avoid.pt` | the 36-sector lidar avoidance net that writes velocity corrections (promoted from `runs/go2_cpg/checkpoint_final_avoid.pt`) |

The `.pt` files are git-ignored; copy them from a training run or a colleague
(see [running.md](running.md#stable-policies) and `policies/README.md`). Only
the two SLAM examples default to the stable names. Every other example takes
explicit paths, so the commands below pass `policies/walk.pt` and
`policies/avoid.pt` by hand. `domo.checkpoints` loads both the legacy script
format (`{"config", "model_state"}`) and the library format
(`{"ppo_config", "extra": {"task_config"}}`), rebuilding the network from the
weight shapes.

### CPU or GPU

| Stage | CPU (laptop) | CUDA GPU |
|-------|--------------|----------|
| Genesis build, arena or flat ground | about 30 s | a few seconds |
| Genesis build, ReplicaCAD apartment | about 2 min | tens of seconds |
| stepping one environment | 50–100 control steps/s | 10–100× faster |
| PPO training with thousands of environments | impractical | hours for a gait, see [running.md](running.md) |

All demonstrations, evaluations and smoke runs work on CPU with one
environment; the commands in each section start with the CPU variant. The
training entry points (`go2_cpg_rl.py`, the two avoidance scripts, the
learning path of `eureka_getup.py`) need a GPU for anything beyond a smoke
test. Training also needs the `rl` extra (`tensorboard`), which
`PPOTrainer` imports unconditionally.

### Frozen scripts and their replicas

`scripts/` is frozen: the original experiment scripts are kept as they were
and are not maintained. Five of the examples are library replicas of them,
with the same CLI where it made sense:

| Frozen script | Library replica | Differences |
|---------------|-----------------|-------------|
| `scripts/house_scene/go2_cpg_rl.py` | `examples/locomotion/go2_cpg_rl.py` | same flags; training and evaluation go through `Go2CPGWalkTask` and `PPOTrainer` |
| `scripts/house_scene/go2_cpg_rl_lidar.py` | `examples/avoidance/go2_cpg_rl_lidar.py` | same flags plus `--lidar`, `--lidar-azimuth`, `--draw-lidar`; evaluation is a `PlanningController` mission in the goal-free `World` |
| `scripts/house_scene/go2_cpg_rl_avoid_house.py` | `examples/avoidance/go2_cpg_rl_avoid_house.py` | as above, `scene_kind="replica"` |
| `scripts/house_scene/evaluate_nav.py` | `examples/navigation/evaluate_nav.py` | the P-controller moved to `domo.control.navigation`; the script keeps the sim wiring, stdin intervention and demos |
| `scripts/house_scene/go2_cpg_rl_voice.py` | `examples/hri/go2_cpg_rl_voice.py` | only the interactive part; the duplicated trainer is gone (use `go2_cpg_rl.py`) |

`skill_demo.py`, `twin_demo.py`, `gemini_design_nav.py`, `slam_demo.py`,
`slam_replica_house.py` and `eureka_getup.py` are library-native and have no
frozen counterpart.

---

## `basic_examples/skill_demo.py`: the Skill / Controller / ControlLoop workflow

### What it demonstrates

The researcher's basic loop: build an engine and a scene with `domo.sim`,
put a `Robot(GO2)` in it, load a policy with `domo.checkpoints`, wrap it in
skills and hand the skills to a `Controller` that a `SimControlLoop` ticks.
Two controllers are compared on the same task, walking a 2 m square:

* the default is a program in the skill grammar, compiled by
  `make_go2_library(policy_fn).compile(...)` and hosted in a
  `SingleSkillController`. `goto @ walk` closes the loop on the robot's pose
  every tick, so each leg is a straight line and the program terminates
  itself when the last corner is reached:

  ```
  (goto(x=2, y=0) @ walk >> goto(x=2, y=2) @ walk >>
   goto(x=0, y=2) @ walk >> goto(x=0, y=0) @ walk) | stand.for(2)
  ```

  The `| stand.for(2)` fallback runs if the route fails (a tipped or fallen
  walk). This is the shape of program the LLM planner emits in M5.

* `--open-loop` runs `PatrolController`, a hand-written `Controller` that
  alternates 4 s of forward walking and 2 s of turning on a clock, with a
  reflex that switches to `StandSkill` when roll or pitch exceeds 0.5 rad.
  It has no pose feedback, so velocity error integrates and the square
  smears. The printed distance from the origin is the comparison.

### How it works

* `build_world(device, headless, dt)`: `create_engine("genesis")`, a scene
  with `dt = 0.02`, flat ground, one Go2 at `(0, 0, 0.35)` with
  `kp = 100`, `kd = 2`; builds and binds a single environment.
* `build_controller(policy_fn, device, open_loop, dt)`: either the
  `PatrolController` over `CPGLocomotionSkill` + `StandSkill`, or the
  compiled program in a `SingleSkillController`. Returns the controller and
  a label for the console.
* `main()` loads the policy with `load_locomotion_policy`, calls
  `controller.setup(robot)`, builds a `SimControlLoop(scene, robot,
  controller, dt=0.02)`, resets it and runs `args.steps` steps with a
  callback that prints every 250 steps.
* `print_summary(controller, final, open_loop)`: if the composition
  finished, prints the program's execution `trace` (the log the supervisor
  reads), then the final position and its distance from the origin.

### Run

```bash
# CPU, no viewer, short run (the smoke test)
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu --steps 300

# CPU, the full square
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu

# GPU with the Genesis viewer
python examples/basic_examples/skill_demo.py policies/walk.pt

# the drifting open-loop contrast
python examples/basic_examples/skill_demo.py policies/walk.pt --open-loop
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--open-loop` | off | run the timed PatrolController (drifts) instead of the closed-loop navigation composition |
| `--steps` | `3000` | control steps to run (60 s of simulated time) |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

### What you will see

```
  Frozen locomotion policy: policies/walk.pt
    obs=... act=12 params=...
Running composition: (goto(x=2, y=0) @ walk >> ... ) | stand.for(2)
  step   250 | skill=...              | pos=(+1.10,+0.02) vx=+0.48
  ...
Program trace:
    ...
Done. final pos=(+0.05,-0.03)  |dist from origin|=0.06 m
```

With `--open-loop` the label is `Running open-loop PatrolController`, the
reflex prints `[ctrl] tipping — switching to stand` / `[ctrl] recovered —
resuming walk` when it fires, there is no trace, and the final distance is
noticeably larger. Nothing is saved.

### Runtime

About 30 s to build plus one minute for 3000 steps on CPU; the 300-step
smoke run finishes in well under a minute. Seconds on a GPU.

### Known limitations

* Single environment, flat ground, no sensors: the point is the control
  workflow, not the scene.
* The composition may finish before `--steps` runs out; the loop keeps
  stepping (the program idles on its last node) until the budget is spent.
* The open-loop square is meant to drift; there is no metric beyond the
  distance printed at the end.

---

## `twin/twin_demo.py`: the digital twin governed by a planner

### What it demonstrates

The DOMO runtime with no task: a `World(WorldConfig(scene_kind="arena"))`
spawns the robot, an obstacle arena and a lidar; `make_go2_library` turns
the walk and avoid checkpoints into a skill library; an `ExploreMission`
(a `PlanningController` subclass) authors programs at runtime from the
library and the outcome history. `ExploreMission` is placeholder
intelligence for the M1 LLM supervisor: it sits in the same seat, reads the
same inputs (`library.describe()`, `ProgramOutcome` traces) and produces the
same output (grammar programs).

Each exploration leg is

```
(avoid @ walk(vx=1, vyaw=±0.3)).until(moved(2)) >> stand.for(1)
```

with the yaw sign alternating between legs and the obstacles re-scattered
before each one. After three successful legs the mission declares
`exploration complete`. A failed leg is followed by the recovery program
`stand.for(2)`; after three consecutive failures the mission concedes
`SKILL GAP (get-up)` and stops, which is the cue `eureka_getup.py` answers.

### How it works

* `main()` builds the `World` and, with `--catalog`, prints
  `library.describe()` (the planner-facing catalogue and grammar) and exits.
* `build_library(world, walk_ckpt, avoid_ckpt)`: `load_locomotion_policy`
  for the walk, `load_avoid_policy` (an `ActorCritic` restored from the
  checkpoint, deterministic actions) for avoid, or `zero_avoid_policy` when
  `--avoid` is omitted; then `make_go2_library(walk, avoid, world.lidar)`.
* `ExploreMission.plan(state, last)` is the whole decision logic: it
  reports the last outcome, counts legs and failures, randomises the
  obstacles and returns the next program string or `None` to idle.
* `run_mission(mission, loop, steps)` steps `world.make_loop(mission)`,
  prints a position line every 250 steps and stops early once the mission
  is idle and done.
* `print_history(mission)` lists every program the mission issued with a
  success mark.

### Run

```bash
# CPU, headless, stable checkpoints, short run
python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt \
    --headless --device cpu --steps 1000

# GPU with the viewer
python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt

# what the LLM planner would see (builds the world, then exits)
python examples/twin/twin_demo.py --walk policies/walk.pt --catalog
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--walk` | required | CPG walk checkpoint |
| `--avoid` | none | Avoidance checkpoint (omit → zero corrections) |
| `--steps` | `6000` | step budget; the run ends earlier when the mission is done |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |
| `--catalog` | off | Print the planner-facing catalog and exit |

### What you will see

```
  [mission] '(avoid @ walk(vx=1, vyaw=0.3)).until(moved(2)) >> stand.for(1)' → ok
  step   250 | pos=(+1.85,+0.40) | (avoid @ walk(vx=1, vyaw=-0.3)).until(moved(2)) >> stand.for(1)
  [mission] leg failed — resting before retry
  [mission] 'stand.for(2)' → ok
  ...
  [mission] exploration complete — idling

  mission history (5 programs):
    ✓ (avoid @ walk(vx=1, vyaw=0.3)).until(moved(2)) >> stand.for(1)
    ✗ ...
```

or, after three consecutive failures, `[mission] repeated failures and no
recovery skill in the library — SKILL GAP (get-up). Stopping mission.`
Nothing is saved.

### Runtime

About 30 s to build; 1000 steps is roughly 20 s on CPU, the full 6000-step
budget a couple of minutes. The mission usually finishes its three legs
well before the budget.

### Known limitations

* Without `--avoid` the avoid skill contributes zero corrections and the
  robot walks blind into obstacles; a fall counts as a failed leg.
* `--catalog` still builds the whole world (about 30 s on CPU) before
  printing.
* The mission never re-plans mid-leg; it only reacts to outcomes.

---

## `locomotion/go2_cpg_rl.py`: train, evaluate, resume the CPG-RL walk

### What it demonstrates

The motor skill everything else stands on. A PPO policy modulates a
central-pattern-generator gait so the Go2 tracks a commanded body velocity
`(vx, vy, vyaw)`; `Go2CPGWalkTask` owns the scene, rewards, resets and the
CPG, `PPOTrainer` runs the optimisation. The trained network is the `walk`
entry of the stable-policy registry and is loaded everywhere else with
`load_locomotion_policy`.

### How it works

* `build_configs(args)`: a `Go2CPGWalkConfig` (`dt = 0.02`, 1000-step
  episodes) and a `PPOConfig` with the frozen script's hyper-parameters
  (24-step rollouts, 5 epochs, `lr = 3e-4` with a linear schedule,
  `hidden_size = 512`, `target_kl = 0.02`).
* `make_trainer(args)`: a fresh `PPOTrainer(Go2CPGWalkTask(cfg), ppo_cfg)`,
  or one restored from `--resume` with `configs_from_checkpoint` (both
  checkpoint formats) and `trainer.load_state`.
* `evaluate(checkpoint_path, command, n_episodes=3, headless=False)`:
  one environment, command resampling disabled, three episodes of
  `run_episode`, which pins the command every step and records the
  velocity error and the CPG phases.
* `gait_period(theta_hist, ep_len, dt)`: stride period from the phase wraps
  of oscillator 0.
* `main()`: `--eval` runs the evaluation and returns; otherwise
  `make_trainer(args).train()`.

### Run

```bash
# evaluate on CPU (the viewer opens; see the limitations)
python examples/locomotion/go2_cpg_rl.py --eval policies/walk.pt --vx 0.5

# CPU training smoke test (slow; only checks the pipeline)
python examples/locomotion/go2_cpg_rl.py --n-envs 16 --total-steps 20000 --device cpu

# train on a GPU
python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless

# resume (new-format and legacy script checkpoints)
python examples/locomotion/go2_cpg_rl.py --resume runs/go2_cpg/checkpoint_step_000100000.pt
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-envs` | `4096` | parallel environments |
| `--total-steps` | `80_000_000` | environment steps to train |
| `--rollout-steps` | `24` | steps per rollout before an update |
| `--device` | `cuda` | one of `cpu`, `cuda`, `mps` |
| `--run-dir` | `runs/go2_cpg` | checkpoints, TensorBoard logs and stats |
| `--headless` | on | `store_true` with default `True`: training is always headless, as in the frozen script |
| `--resume` | none | checkpoint to continue training from |
| `--eval` | none | checkpoint to evaluate instead of training |
| `--vx` | `0.5` | forward command during `--eval` (m/s) |
| `--vy` | `0.0` | lateral command during `--eval` (m/s) |
| `--vyaw` | `0.0` | yaw-rate command during `--eval` (rad/s) |

### What you will see

Training prints the `PPOTrainer` progress and writes
`<run-dir>/checkpoint_step_<step:09d>.pt` at the trainer's save interval,
`<run-dir>/checkpoint_final.pt` at the end, and TensorBoard event files in
`--run-dir`. Evaluation prints a tracking line every 50 steps and a summary
per episode:

```
    step   50  vx=+0.47/0.50  vy=-0.01  wz=+0.02  h=0.31  r=[1.00 0.98 1.00 0.99]
    ...
  Episode 1 | return= 812.40 | length=1000 | mean |vx err|=0.041 m/s | gait period~0.62s
```

An episode ends when the task terminates it (a fall or the 1000-step cap).
Evaluation saves nothing.

### Runtime

Training the gait from scratch is hours on a CUDA GPU with 4096
environments. Evaluation is three episodes of up to 1000 steps: a minute or
so on CPU after the build.

### Known limitations

* `--headless` is not forwarded to `evaluate()`, so `--eval` always opens
  the Genesis viewer; on a display-less machine the evaluation cannot run.
* `--eval` picks its device with `pick_device("cuda")` and ignores
  `--device`.
* `--headless` cannot be turned off for training (it is always `True`).

---

## `avoidance/go2_cpg_rl_lidar.py`: lidar avoidance in a walled arena

### What it demonstrates

A small avoidance network (36 lidar sectors → a velocity correction `Δv`)
trained with PPO over a frozen walk policy, in `Go2AvoidTask` with
`scene_kind="arena"`. Evaluation does not use an RL episode counter: it
spawns the goal-free `World` the checkpoint trained in
(`world_from_avoid_config`), builds a library from the two checkpoints and
hands the robot to `AvoidanceMission`, a `PlanningController` that issues

```
(avoid @ walk(vx=0.6)).for(20)
```

once per episode after resetting the robot and re-scattering the obstacles.
Episodes end through the skill cards (avoid's `blocked()` abort, walk's
`tipped()` / `fallen()` aborts, the `.for(20)` horizon). Both sensor models
in `domo.robot.lidar_models` are available: the idealised sector lidar of
the original script and the Hesai XT16.

### How it works

* `lidar_model_from_args(args, default_none=False)`: `--lidar xt16` →
  `hesai_xt16(n_horizontal=args.lidar_azimuth)`; otherwise the generic
  sector lidar, or `None` during evaluation when `--lidar` is unset (keep
  what the checkpoint trained with).
* `build_configs(args)`: `Go2AvoidConfig` for the arena and a `PPOConfig`
  matching the script's `AvoidanceNet` (3 × 128 trunk, 64 head,
  `ent_coef = 0.02`, non-finite guard).
* `make_trainer(args, policy_fn, build_configs_fn)`: the trainer, fresh or
  resumed; the walk checkpoint path is stored in the checkpoint's `extra`
  so evaluation can find it later.
* `evaluate(checkpoint_path, cpg_checkpoint, n_episodes=5, command_vx=0.6,
  headless=False, lidar_model=None, draw_lidar=False)`: rebuilds the task
  config from the checkpoint, applies the optional sensor override, builds
  the `World`, then `build_eval_library` (walk + avoid with the trained
  `avoid_deltas` and `obs_max_range`) and `run_mission`.
* `run_mission(...)` prints a status line every 50 steps and stops when the
  mission is idle with all episodes recorded, or after
  `(n_episodes + 1) × max_episode_steps` steps.
* `main()`: `--eval` → `evaluate(...)`; otherwise `--cpg-checkpoint` is
  required and `make_trainer(...).train()` runs.

### Run

```bash
# evaluate on CPU, headless (the smoke test: about 110 s)
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
    --cpg-checkpoint policies/walk.pt --headless

# evaluate with the viewer and lidar rays drawn
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
    --cpg-checkpoint policies/walk.pt --vx 0.6 --draw-lidar

# preview the trained net on a simulated XT16 (distribution shift, see below)
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
    --cpg-checkpoint policies/walk.pt --lidar xt16 --lidar-azimuth 360 --headless

# train on a GPU
python examples/avoidance/go2_cpg_rl_lidar.py --cpg-checkpoint policies/walk.pt \
    --n-envs 4096 --device cuda --headless
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--cpg-checkpoint` | none | Path to trained CPG locomotion checkpoint. Required for training; during `--eval` it falls back to the path stored in the avoid checkpoint |
| `--n-envs` | `4096` | parallel environments |
| `--total-steps` | `100_000_000` | environment steps to train |
| `--rollout-steps` | `24` | steps per rollout |
| `--device` | `cuda` | one of `cpu`, `cuda`, `mps` |
| `--run-dir` | `runs/go2_avoidance` | checkpoints and logs |
| `--headless` | off | no Genesis viewer (training and evaluation) |
| `--resume` | none | checkpoint to continue training from |
| `--eval` | none | avoidance checkpoint to evaluate |
| `--vx` | `0.6` | forward command of the evaluation program (m/s) |
| `--lidar` | none | Sensor model: idealised script lidar or Hesai XT16. Train default: simple. Eval default: whatever the checkpoint was trained with (pass to override). Choices `simple`, `xt16` |
| `--lidar-azimuth` | `180` | XT16 azimuth samples in sim (multiple of 36; 1980 = full device fidelity) |
| `--draw-lidar` | off | Visualise lidar rays in the eval viewer (slow on macOS) |

### What you will see

```
  Frozen locomotion policy: policies/walk.pt
    step    50  min_lidar=1.84m  vx=+0.55  cmd=(0.60,0.00,0.12)
    ...
  Episode 1/5 | SUCCESS
      ...program trace...

  Episode 2/5 | FAILURE
      ...
  4/5 successful runs
```

`[eval] sensor override: <trained> → <override>` appears when `--lidar` is
given during evaluation, `[eval] step budget exhausted` when the mission did
not finish inside its budget. Training writes
`<run-dir>/checkpoint_step_*.pt`, `checkpoint_final.pt` and TensorBoard
logs; evaluation saves nothing.

### Runtime

Evaluation: five 20 s programs, about 110 s on CPU including the build.
Training: hours on a GPU.

### Known limitations

* `--eval` picks its device with `pick_device("cuda")` and ignores
  `--device`.
* `policies/avoid.pt` is a legacy script checkpoint; whether it carries the
  walk path depends on how it was trained, so always pass
  `--cpg-checkpoint` for evaluation.
* An XT16 override on a net trained with the simple sensor is a
  distribution shift; expect worse success counts. `--lidar-azimuth` must be
  a multiple of 36 so the sectors pool evenly.
* `--draw-lidar` only applies with the viewer and is slow on macOS with the
  XT16 (thousands of rays).

---

## `avoidance/go2_cpg_rl_avoid_house.py`: lidar avoidance in a ReplicaCAD house

### What it demonstrates

The same avoidance architecture, in the Habitat ReplicaCAD apartment
(`scene_kind="replica"`) instead of the randomised arena, with the floor
height (`ground_height = 0.2`) and spawn pose (`(3, -3, 0.44)`, yaw 180°)
of the original experiment. The script imports `evaluate`, `make_trainer`,
`lidar_model_from_args` and `add_common_args` from `go2_cpg_rl_lidar.py`;
only `build_configs` differs.

### How it works

* `build_configs(args)`: `Go2AvoidConfig(scene_kind="replica",
  replica_scene_json=..., replica_asset_root=..., ground_height=0.2,
  base_init_pos=(3.0, -3.0, 0.44), base_init_yaw_deg=180.0)` and the same
  PPO recipe as the arena.
* `main()`: `--eval` → the arena script's `evaluate(...)`, which rebuilds
  the house from the config stored in the checkpoint; otherwise
  `make_trainer(args, policy_fn, build_configs).train()`.

### Run

```bash
# evaluate a house checkpoint on CPU, headless
python examples/avoidance/go2_cpg_rl_avoid_house.py \
    --eval runs/go2_avoid_house/checkpoint_final.pt \
    --cpg-checkpoint policies/walk.pt --headless

# train on a GPU
python examples/avoidance/go2_cpg_rl_avoid_house.py \
    --cpg-checkpoint policies/walk.pt \
    --scene-json scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json \
    --asset-root scripts/house_scene/data/replica_cad/ \
    --n-envs 1024 --device cuda --headless
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--cpg-checkpoint` | none | walk checkpoint; required for training |
| `--scene-json` | `scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json` | ReplicaCAD scene instance |
| `--asset-root` | `scripts/house_scene/data/replica_cad/` | ReplicaCAD asset root |
| `--n-envs` | `1024` | parallel environments (the house is heavier than the arena) |
| `--total-steps` | `100_000_000` | environment steps to train |
| `--rollout-steps` | `24` | steps per rollout |
| `--device` | `cuda` | one of `cpu`, `cuda`, `mps` |
| `--run-dir` | `runs/go2_avoid_house` | checkpoints and logs |
| `--headless` | off | no Genesis viewer |
| `--resume`, `--eval`, `--vx`, `--lidar`, `--lidar-azimuth`, `--draw-lidar` | as in `go2_cpg_rl_lidar.py` | shared through `add_common_args` |

### What you will see

The same console output as the arena script. Training writes under
`runs/go2_avoid_house`.

### Runtime

Building the apartment takes about 2 min on CPU even for one environment;
stepping is slower than the arena. Training is a GPU job.

### Known limitations

* The ReplicaCAD assets must be present under `--asset-root`; they are not
  part of the repository.
* No stable house checkpoint exists in `policies/`; `--eval` needs a run of
  your own.
* The arena script's evaluation limitations apply (device auto-picked,
  `--cpg-checkpoint` recommended).

---

## `navigation/gemini_design_nav.py`: the LLM as level designer

### What it demonstrates

An LLM plans a whole mission once and has no control while it runs. The
robot spawns at `R` in a 10 × 7 grid with three pillars and must reach `G`
in the opposite corner. The prompt contains the task, the skill catalogue
(`library.describe()`) and the ASCII map; the reply must be one composition
program. The grid is the world (cell column = x, row = y, 1 m cells), so
`goto(x=5, y=3) @ walk` waypoints route straight to cells. Two strategies
are open to the designer: a collision-free polyline of `goto` legs, or a
coarse route with `avoid @ goto(x=.., y=..) @ walk` layering reactive lidar
avoidance on each leg. The program is compiled (compile errors are fed back
for up to two repair rounds), executed blind, and the robot's actual path is
overlaid on the map with a verdict.

The `scripted` provider needs no API key and replays a hand-verified route:

```
goto(x=1, y=4) @ walk >> goto(x=6, y=6) @ walk >> goto(x=9, y=6) @ walk
```

With `--scripted-avoid` it replays the beeline
`(avoid @ goto(x=9, y=6) @ walk) | stand.for(2)` instead, which shows the
composition running but often does not finish (see the limitations).

### How it works

* `Arena(rows)` parses the map, renders it (`render(path)` overlays `*`),
  detects hits (robot centre within 0.6 m of a pillar centre) and builds the
  pillars plus a bounding wall pushed two cells out.
* `build_world(arena, device, headless, use_avoid)`: engine, scene, Go2 at
  `R`, and, when avoidance is in the library, a generic sector lidar pooled
  into 36 sectors (the avoid net's observation).
* `build_library(args, lidar, device, use_avoid)`: walk only, or walk +
  avoid with `avoid_deltas = (0.25, 0.5, 1.2)` so avoid can steer but not
  reverse `goto`'s progress.
* `make_designer(args, use_avoid)`: a `ScriptedClient` with the canned
  route, or `make_llm(args.llm)`.
* `design_program(llm, arena, library, device, max_repairs=2)`: prompt,
  `extract_program` on the reply, `library.compile`; on `CompileError` or
  `GrammarError` the error is appended to the prompt and the LLM is asked
  again; after three failed attempts the script exits.
* `run_blind(loop, program, arena, steps)`: steps a `SimControlLoop` with
  the compiled program in a `SingleSkillController`, records visited cells
  and stops on a hit, on reaching the goal (within 0.6 m), when the program
  finishes or when the budget is spent.
* `print_verdict(...)`: the verdict, the map with the path, the program
  trace.

### Run

```bash
# offline, CPU: the canned safe route
python examples/navigation/gemini_design_nav.py policies/walk.pt \
    --avoid-checkpoint policies/avoid.pt --llm scripted --headless --device cpu

# offline, goto-only library (no avoid checkpoint needed)
python examples/navigation/gemini_design_nav.py policies/walk.pt --no-avoid --headless --device cpu

# offline: the reactive 'avoid @ goto @ walk' beeline (may not reach the goal)
python examples/navigation/gemini_design_nav.py policies/walk.pt \
    --avoid-checkpoint policies/avoid.pt --llm scripted --scripted-avoid --headless

# let Gemini design it (GEMINI_API_KEY; GPU and viewer by default)
python examples/navigation/gemini_design_nav.py policies/walk.pt \
    --avoid-checkpoint policies/avoid.pt --llm gemini
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--llm` | `scripted` | scripted (offline) \| gemini \| vllm \| openai. Any name `domo.llm.make_llm` accepts works (also `gemini-lc`); there is no `choices` restriction |
| `--avoid-checkpoint` | `runs/go2_cpg/checkpoint_final_avoid.pt` | trained lidar-avoidance net; enables 'avoid @ ...'. `policies/avoid.pt` is the promoted copy of that file |
| `--no-avoid` | off | omit the avoidance skill (goto-only library) |
| `--scripted-avoid` | off | with --llm scripted, 'design' the beeline 'avoid @ goto @ walk' route instead of the safe polyline (shows reactive avoidance; may not finish) |
| `--steps` | `6000` | step budget for the blind run |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

### What you will see

```
Arena:
    0123456789   (x=col)
y=0 |R         |
y=1 |   #      |
...
[design attempt 1] LLM proposed:
  goto(x=1, y=4) @ walk >> goto(x=6, y=6) @ walk >> goto(x=9, y=6) @ walk

Compiled program:
  goto(x=1, y=4) @ walk >> ...
  step   250 | pos=(+0.40,+1.60) | skill=goto(x=1, y=4) @ walk
  ...
=== REACHED THE GOAL ===
Actual path taken (robot cells = '*'):
    ...
Program trace:
    ...
```

The verdict is one of `REACHED THE GOAL`, `CRASHED into an obstacle` or
`ran out of steps / gave up` (each followed by an emoji in the console). A
compile failure prints `compile error: ...` before the next attempt.
Nothing is saved.

### Runtime

The scripted route takes 2–3 min on CPU; the `--steps 200` smoke run
finishes in about a minute. With a real provider add the LLM latency.

### Known limitations

* The default `--avoid-checkpoint` is a `runs/` path from the original
  experiment; pass `--avoid-checkpoint policies/avoid.pt` or `--no-avoid`.
* The bundled avoid policy is conservative and tends to stall in front of a
  pillar rather than skirt it, so `--scripted-avoid` (and LLM designs that
  rely on avoidance) often end with `ran out of steps`. The goto-only
  strategy is reliable.
* The LLM sees no feedback during execution; a wrong plan is simply wrong.
* `gemini` needs `google-genai` and `GEMINI_API_KEY`; `vllm` / `openai`
  need the `langchain` extra and a running endpoint.

---

## `navigation/evaluate_nav.py`: waypoint navigation with intervention

### What it demonstrates

The legacy waypoint driver, `domo.control.navigation.PositionController`,
driven through two injected callables (`step_fn(cmd)` and `pose_fn()`), so
the controller itself never touches the engine. The script supplies the sim
wiring (a single-environment `Go2CPGWalkTask` used as a stepping harness), a
non-blocking stdin `MissionControl`, a rule-based parser for natural-language
commands (`parse_simple`) and a Gemini parser (`parse_gemini`,
`gemini-2.5-flash`, one call per utterance), plus three demos.

### How it works

* `build_env(device, headless)`: `Go2CPGWalkTask` with one environment, a
  practically infinite episode and no command resampling.
* `make_step_and_pose(env, policy_fn)`: `step_fn` writes the command,
  queries the frozen policy and steps the task, resetting the episode if the
  robot fell; `pose_fn` returns `(x, y, yaw)`.
* `MissionControl` polls stdin with `select` every control step and
  exposes `paused`, `aborted` (read-and-clear), `stopped` and a speed delta
  to the controller.
* `execute_commands(commands, controller)` maps command dicts
  (`forward`, `backward`, `turn`, `go_to`, `stop`) onto
  `controller.go_forward / go_backward / turn / go_to / stop`.
* The demos: `forward` (5 m, turn left 90°, 3 m at 0.5 m/s, turn right
  90°, stop), `waypoints` (a 3 m square through `(3,0) (3,3) (0,3) (0,0)`),
  `interactive` (a `nav>` prompt parsed by rules or Gemini until `quit`).

### Run

```bash
# the smoke test: forward sequence, headless (about 60 s)
python examples/navigation/evaluate_nav.py policies/walk.pt --demo forward --headless

# the default: square waypoints with the viewer
python examples/navigation/evaluate_nav.py policies/walk.pt

# type commands
python examples/navigation/evaluate_nav.py policies/walk.pt --demo interactive
python examples/navigation/evaluate_nav.py policies/walk.pt --demo interactive --gemini
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--demo` | `waypoints` | one of `forward`, `waypoints`, `interactive` |
| `--gemini` | off | parse interactive commands with Gemini instead of the rule-based parser |
| `--headless` | off | no Genesis viewer |

There is no `--device`; the device is `cuda` when available, else `cpu`.

While a demo runs, type a key and Enter on stdin:

| Key | Effect |
|-----|--------|
| `p` / `pause` | pause |
| `r` / `resume` | resume |
| `a` / `abort` | abort the current goal (one abort per goal) |
| `s` / `stop` | stop the mission |
| `+` / `faster`, `-` / `slower` | change speed by 0.2 m/s |
| `?` / `status` | print the intervention state |

The interactive prompt accepts `forward N [at S]`, `backward N`,
`turn N [left|right]`, `go to X Y`, `stop` and `quit`; anything the
rule-based parser does not understand becomes `stop`.

### What you will see

```
  [ctrl] Intervention active — type commands + Enter:
         p=pause r=resume a=abort s=stop +=faster -=slower ?=status

=== Demo: square waypoints ===

  Waypoint 1/4: (3.0,0.0)
  ...controller progress lines...
=== Done ===
```

In interactive mode each parsed command is echoed as `→ [{"type": ...}]`
before it runs. Nothing is saved.

### Runtime

About 60 s on CPU for the forward demo, a few minutes for the square.

### Known limitations

* stdin polling uses `select`, which does not work on Windows or with a
  piped stdin; intervention is then silently unavailable.
* A fall resets the episode: the robot reappears at the spawn while the
  controller keeps its goal.
* `--gemini` needs `google-genai` and `GEMINI_API_KEY`; without them every
  utterance parses to `stop`.

---

## `slam/slam_demo.py`: passive SLAM during a perimeter patrol

### What it demonstrates

`SlamSkill` from `domo.control` is a perception layer: it reads the lidar
and the pose every tick and folds them into a log-odds occupancy grid and a
3D point cloud, but writes zero velocity. Here it observes a `goto` patrol
around the perimeter of a 12 × 9 m arena whose obstacles have different
heights (0.9 m blocks and an L-shaped wall, a 1.8 m pillar, a 0.35 m step
below the lidar's 2D slice), so the point cloud shows structure the flat map
loses. The robot carries a simulated Hesai XT16 (16 channels × 360 azimuth
samples = 5760 beams per scan); SLAM maps a robot-height slice into the grid
and accumulates the full cloud.

The patrol program is

```
(goto(x=11, y=0) @ walk >> goto(x=11, y=8) @ walk >>
 goto(x=0, y=8) @ walk >> goto(x=0, y=0) @ walk) | stand.for(2)
```

SLAM is held at loop level rather than composed as `slam @ goto @ walk`
because a composition resets each layer when its node is re-entered, which
would wipe the map at every waypoint; mapping must persist across the whole
mission.

### How it works

* `build_world(args, device, obstacles, start, h, w)`: the arena boxes and
  walls, the Go2, the XT16 handle (added before `build()`), an optional
  offscreen camera for the dashboard, then `SimulatedLidar` pooled into 36
  sectors for the 2D panel.
* `build_brain(checkpoint, robot, lidar, device, h, w)`: the compiled
  patrol in a `SingleSkillController`, and `SlamSkill(lidar,
  SlamConfig(resolution=0.1, half_extent=8.0, origin=<arena centre>,
  z_band=(0.25, 1.3), map_max_range=12.0))`.
* `attach_dashboard(args, obstacles, h, w)`: `None`, an in-process
  `Dashboard(port, title="DOMO SLAM — arena")` or a
  `DashboardClient(url)`; pushes a scene manifest with the arena boxes and
  the Go2 URDF (`slam_viz.go2_robot_manifest`). `domo.dashboard` is
  imported only here.
* `run(args, loop, program, slam, lidar, camera, dash, recorder, live)`:
  every step drives the loop, calls `slam.update_command(state, dt)`,
  feeds the GIF recorder and the live plot, publishes a snapshot every 10
  steps (about 5 Hz) and prints every 500 steps; stops when the program
  finishes or the budget is spent. Dashboard commands: `pause` toggles
  (the sim freezes but keeps publishing), `reset` restarts the episode and
  the map, `stop` ends the run.
* `report(slam, state, out_path)`: ground truth, the SLAM ASCII map,
  coverage, odometry drift, cloud size, and `save_point_cloud` to `--out`.
  With a dashboard, `hold_dashboard` keeps serving the final state until
  the page's stop button or Ctrl+C.

### Run

```bash
# CPU, headless, stable walk policy, saves slam_cloud.png
python examples/slam/slam_demo.py --headless --device cpu

# also record a GIF of the cloud forming
python examples/slam/slam_demo.py --headless --device cpu --animate slam_cloud.gif

# watch the reconstruction live in a matplotlib window (Genesis stays headless)
python examples/slam/slam_demo.py --headless --device cpu --live

# in-process web dashboard with the camera frame (open http://127.0.0.1:8080)
python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080 --dashboard-camera

# decoupled dashboard: start the server in another shell first
python -m domo.dashboard --port 8080
python examples/slam/slam_demo.py --headless --device cpu --dashboard-url http://127.0.0.1:8080

# GPU with the Genesis viewer
python examples/slam/slam_demo.py
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional, optional) | stable `walk` | CPG locomotion checkpoint (default: stable 'walk') |
| `--steps` | `9000` | step budget; the patrol usually finishes earlier |
| `--azimuth` | `360` | XT16 azimuth samples (multiple of 36; try 720 for a denser cloud) |
| `--out` | `slam_cloud.png` | path for the 3D point-cloud scatter PNG |
| `--animate` | none | path for a GIF animating the cloud forming in real time |
| `--animate-every` | `200` | capture an animation frame every N steps |
| `--live` | off | show a separate live plot of the reconstruction while the sim runs (keep --headless: Genesis draws no window) |
| `--live-every` | `40` | refresh the live plot every N steps |
| `--dashboard PORT` | `0` | host the dashboard IN-process on this port (0 = off) |
| `--dashboard-url URL` | none | push to a DECOUPLED dashboard server (e.g. http://127.0.0.1:8080; start it with 'python -m domo.dashboard --port 8080') |
| `--dashboard-camera` | off | add the Genesis camera panel (renders a frame — small sim-thread cost; off by default) |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

### What you will see

```
Ground-truth arena:
    012345678901  (x)
y=0 |R           |
...
Patrolling: (goto(x=11, y=0) @ walk >> ... ) | stand.for(2)

  step   500 | pos=(+4.80,+0.10) | coverage=12.3% | odom_drift=0.04 m
  ...

Ground truth (# = obstacle):
...
SLAM occupancy map (# occupied, . free, ' ' unknown):
...
Coverage: 61.2% of the map is confidently known.
Odometry drift over the run: 0.31 m (where a real system would fuse scan-matching to correct it).

3D point cloud: 1,234,567 accumulated world points.
  [viz] saved 3D point cloud → slam_cloud.png
```

`--animate` adds `[viz] saved animation (N frames) → <path>`; with a
dashboard the run ends with `[dashboard] still live at <url> — Ctrl+C to
exit.` and the process waits.

### Runtime

3–4 min on CPU for the full patrol (the XT16's 5760 beams per scan are the
main cost); the build is about 30 s. Seconds per lap on a GPU.

### Known limitations

* Ctrl+C is a hard kill: the script restores the OS default `SIGINT`
  handler because Genesis calls defer `KeyboardInterrupt`. Nothing is
  flushed on interrupt; the dashboard's stop button is the graceful path.
* `--live` needs a display and an interactive matplotlib backend; without
  one it prints a notice and continues. Do not combine it with the Genesis
  viewer (two GUI toolkits in one process).
* `--dashboard-camera` only takes effect with the in-process `--dashboard
  PORT`; with `--dashboard-url` the camera is not created. The page has no
  camera panel: the frame is served at `GET /frame.jpg` and can be opened in
  a browser tab (see [api/dashboard.md](api/dashboard.md)).
* The 3D viewer of the dashboard loads three.js from a CDN and the Go2
  meshes from the local server; it needs internet access and has not been
  verified in a sandboxed browser.
* Each obstacle casts an occlusion shadow the single loop never fills; the
  map is meant to have gaps.

---

## `slam/slam_replica_house.py`: passive SLAM in a ReplicaCAD apartment

### What it demonstrates

The same `SlamSkill` in a cluttered, realistic environment. The goal-free
`World` builds the apartment with `scene_kind="replica"`, the XT16 lidar
and an optional offscreen camera; a `HouseWanderer` (`PlanningController`)
authors a blind exploration wander and ticks SLAM in its `update()` so the
map persists across programs. Legs alternate

```
(avoid @ walk(vx=0.4)).for(6)        # forward under lidar avoidance
walk(vyaw=±0.8).for(2)               # turn a random way
```

for up to 80 legs. When a forward leg stalls (avoid's `blocked` or a tip)
it fails and the next plan turns away, so the wanderer backs out of
dead-ends. Avoid's forward authority is capped with
`avoid_deltas = (0.2, 0.5, 1.2)` so it steers around furniture without
cancelling the walk. Because the wander is blind, the map has honest gaps,
which is the motivation for a frontier explorer that reads SLAM's map.

### How it works

* `build_world(args, device)`: `World(WorldConfig(scene_kind="replica",
  ground_height=0.2, base_init_pos=(3, -3, 0.44), base_init_yaw_deg=180,
  lidar_model=hesai_xt16(n_horizontal=args.azimuth), camera_res=...))`.
* `build_brain(args, world)`: walk and avoid from `--cpg-checkpoint` /
  `--avoid-checkpoint` or the stable registry, `make_go2_library(...,
  avoid_deltas=AVOID_DELTAS)`, `SlamSkill(world.lidar,
  SlamConfig(resolution=0.1, half_extent=10.0, origin=<spawn>,
  z_band=(0.4, 1.6), map_max_range=10.0))`, the `HouseWanderer`.
* `attach_dashboard(args)`: as in `slam_demo.py`, but the 3D scene holds
  only the robot (the house meshes are not pushed yet).
* `run(args, world, loop, controller, slam, dash, recorder, live)`: the
  same loop as the arena demo; `reset` from the dashboard also calls
  `controller.restart()`; the run ends when the wanderer idles after its
  last leg or the budget is spent.
* `report(slam, out_path)`: the ASCII map, coverage, cloud size, the PNG.

### Run

```bash
# CPU, headless, stable walk + avoid, saves house_cloud.png
python examples/slam/slam_replica_house.py --headless --device cpu

# live matplotlib window, GIF, dashboard
python examples/slam/slam_replica_house.py --headless --device cpu --live
python examples/slam/slam_replica_house.py --headless --device cpu --animate house.gif
python examples/slam/slam_replica_house.py --headless --device cpu --dashboard 8080

# GPU with the Genesis viewer
python examples/slam/slam_replica_house.py
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--cpg-checkpoint` | stable `walk` | CPG walk checkpoint (default: stable 'walk') |
| `--avoid-checkpoint` | stable `avoid` | avoidance checkpoint (default: stable 'avoid') |
| `--scene-json` | `scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json` | ReplicaCAD scene instance |
| `--asset-root` | `scripts/house_scene/data/replica_cad/` | ReplicaCAD asset root |
| `--azimuth` | `360` | XT16 azimuth samples (multiple of 36) |
| `--steps` | `10000` | step budget |
| `--seed` | `0` | seeds `random` and torch (the wander's turn directions) |
| `--out` | `house_cloud.png` | path for the point-cloud PNG |
| `--animate` | none | path for a GIF animating the cloud forming in real time |
| `--animate-every` | `250` | capture an animation frame every N steps |
| `--live` | off | show a separate live plot of the reconstruction while the sim runs (keep --headless: Genesis draws no window) |
| `--live-every` | `50` | refresh the live plot every N steps |
| `--dashboard PORT` | `0` | host the dashboard IN-process on this port (0 = off) |
| `--dashboard-url URL` | none | push to a DECOUPLED dashboard server (start it with 'python -m domo.dashboard --port 8080') |
| `--dashboard-camera` | off | add the Genesis camera panel (renders a frame — small sim-thread cost; off by default) |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

### What you will see

```
Wandering the ReplicaCAD house (spawn (3.0, -3.0)) ...
  step   500 | pos=(+1.20,-2.90) | leg 4/80 | coverage= 8.1%
  ...
SLAM occupancy map (# occupied, . free, ' ' unknown):
...
Coverage: 34.5% of the map is confidently known (blind wander → honest gaps).
3D point cloud: 2,345,678 accumulated world points.
  [viz] saved 3D point cloud → house_cloud.png
```

### Runtime

About 2 min to build the apartment on CPU, then roughly 5 min for 10 000
steps. Much faster on a GPU.

### Known limitations

* Needs the ReplicaCAD assets under `--asset-root`.
* Same Ctrl+C, `--live`, `--dashboard-camera` and 3D-viewer caveats as
  `slam_demo.py`; in addition the dashboard's 3D panel shows the robot on
  an empty floor because the house meshes are not part of the manifest.
* The wander is blind; coverage depends on the seed and stays partial.

---

## `hri/go2_cpg_rl_voice.py`: voice-commanded locomotion

### What it demonstrates

`domo.hri.VoiceCommander` runs a thread that listens on the microphone,
transcribes with Whisper, asks Gemini to turn the sentence into
`(vx, vy, vyaw)` and writes the result into a shared `CommandState`. The
sim loop reads that state every control step and feeds the frozen walk
policy through a single-environment `Go2CPGWalkTask` used as a stepping
harness.

### How it works

* `build_env(device, headless)`: the task with one environment, a
  practically infinite episode and no command resampling.
* `drive(env, policy_fn, state)`: steps forever, copying the latest voice
  command into `env.commands`, resetting when the robot falls and printing
  every 250 steps.
* `run_voice_eval(checkpoint_path, whisper_model="tiny", initial_vx=0.0,
  headless=False)`: loads the policy, starts the `VoiceCommander`, runs
  `drive` until Ctrl+C, stops the commander.

### Run

```bash
pip install openai-whisper sounddevice google-genai      # or: pip install -e '.[voice,llm]'
export GEMINI_API_KEY=<key>

python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --whisper-model tiny
# no display
python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --headless
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--eval` | required | CPG locomotion checkpoint to drive |
| `--whisper-model` | `tiny` | one of `tiny`, `base`, `small` |
| `--vx` | `0.0` | Initial forward speed before the first voice command |
| `--headless` | off | no Genesis viewer |

There is no `--device`; the device is `cuda` when available, else `cpu`.

### What you will see

The `VoiceCommander`'s own transcription and command messages, then every
250 steps `[sim] cmd=(0.50,0.00,0.00)  vx=+0.48  h=0.31`, and
`[sim] robot fell — resetting` when it does. `Stopping.` on Ctrl+C.
Nothing is saved.

### Runtime

Runs until interrupted. The first run downloads the Whisper model.

### Known limitations

* Not smoke-testable: it needs a microphone, the `voice` extra,
  `google-genai` and `GEMINI_API_KEY`.
* Commands arrive with the latency of transcription plus one LLM call.
* Training the walk policy was removed from this replica; use
  `locomotion/go2_cpg_rl.py`.

---

## `eureka/eureka_getup.py`: learning the missing get-up skill

### What it demonstrates

The M2/M3 answer to the skill gap `twin_demo.py` reports.
`domo.eureka.learn_skill(SkillLearningRequest)` runs Eureka on
`Go2GetUpTask`: the LLM writes candidate reward functions, each candidate
trains in its own subprocess (one Genesis per process), candidates are
ranked on the task's fixed success metric (upright and held for 1 s), a
reflection prompt feeds the next iteration. `--dr` adds the DrEureka stage:
single-parameter sweeps find feasible domain-randomisation bounds, the LLM
proposes several DR configurations, all are trained and the best is kept.
`--demo CKPT` skips learning and deploys a checkpoint in the twin as a
`LearnedJointSkill` to watch the robot get up.

The pipeline is orchestrated with LangGraph when it is installed
(`--no-graph` or a missing `langgraph` selects the imperative driver). The
LLM layer is provider-agnostic; `--llm scripted` replaces it with an
`OfflineClient` that returns a hand-written reward and a fixed DR proposal,
so the whole loop runs without an API key.

### How it works

* `build_request(args, device)`: a `SkillLearningRequest(skill_name="getup",
  task="go2_getup", eureka=EurekaConfig(iterations, samples, n_envs,
  train_steps, device), run_dr=args.dr, dr=DrEurekaConfig(samples,
  retrain_steps), run_root, llm, llm_kwargs, use_graph)`. `--llm-model` and
  `--llm-base-url` only apply to the LangChain providers (`vllm`, `openai`,
  `gemini-lc`).
* `main()` (learning): `learn_skill(request, llm=OfflineClient(...) if
  scripted else None)`, then prints the `--demo` command for the winner's
  checkpoint.
* `build_getup_twin(checkpoint, device, headless)`: a flat-ground `World`
  without lidar, the checkpoint's `ActorCritic` wrapped in
  `LearnedJointSkill(..., obs_builder=build_getup_observation,
  action_scale=0.35)` in a `SingleSkillController`.
* `knock_over(world, ep)`: lays the robot on a random side (roll between
  100° and 170°, alternating sign).
* `demo(checkpoint, device, headless, episodes=3)`: three episodes of 400
  steps (8 s); an episode counts as a success when the base rises above
  0.26 m with roll and pitch under 0.4 rad.

### Run

```bash
# CPU smoke test of the whole loop, offline (minutes)
python examples/eureka/eureka_getup.py --llm scripted --device cpu \
    --n-envs 8 --train-steps 2000 --samples 1 --iterations 1

# learn with Gemini on a GPU (export GEMINI_API_KEY=...)
python examples/eureka/eureka_getup.py --samples 4 --iterations 3 --train-steps 5000000 --device cuda

# a local model served by vLLM (OpenAI-compatible), via LangChain
python examples/eureka/eureka_getup.py --llm vllm --llm-model Qwen/Qwen2.5-Coder-7B-Instruct

# add the DrEureka robustness stage
python examples/eureka/eureka_getup.py --dr --dr-samples 8

# watch a learned checkpoint get up in the twin
python examples/eureka/eureka_getup.py --demo runs/eureka/getup/iter_0/train_0/checkpoint_final.pt --headless --device cpu
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--samples` | `4` | reward candidates per iteration |
| `--iterations` | `3` | Eureka iterations |
| `--train-steps` | `5_000_000` | environment steps per candidate |
| `--n-envs` | `2048` | parallel environments per candidate |
| `--device` | `cuda` | torch/Genesis device |
| `--llm` | `gemini` | LLM provider. vllm/openai use LangChain. Choices `gemini`, `vllm`, `openai`, `gemini-lc`, `scripted` |
| `--llm-model` | none | Model name for vllm/openai (e.g. Qwen/Qwen2.5-Coder-7B-Instruct) |
| `--llm-base-url` | none | Endpoint for vllm/openai (vLLM default http://localhost:8000/v1) |
| `--no-graph` | off | Use the imperative driver instead of LangGraph |
| `--dr` | off | Run the DrEureka robustness stage on the winner |
| `--dr-samples` | `4` | Independent DR configs to train & compare (paper: 16) |
| `--dr-retrain-steps` | none | steps for the DR retraining (default: `--train-steps`) |
| `--run-root` | `runs/eureka` | root of the run tree |
| `--demo` | none | Skip learning; deploy this checkpoint in the twin |
| `--headless` | off | no Genesis viewer |

### What you will see

Learning prints the routine's progress per iteration and candidate
(`training (5,000,000 steps)...`) and ends with

```
To watch it in the twin:
  python examples/eureka/eureka_getup.py --demo runs/eureka/getup/iter_2/train_1/checkpoint_final.pt
```

The run tree is `<run-root>/getup/iter_<k>/` with the prompts and
`reflection.txt` per iteration and `train_<j>/{spec.json, results.json,
checkpoint_final.pt}` per candidate. The demo prints one line per episode:

```
  Episode 1: got up in 1.84s, final height 0.31 m
  Episode 2: did NOT get up, final height 0.12 m
```

and saves nothing.

### Runtime

The CPU smoke command runs in minutes and trains nothing useful. A real
Eureka run is `samples × iterations` PPO trainings of `--train-steps` each:
hours on a GPU. The demo is about a minute on CPU.

### Known limitations

* Each candidate trains in a fresh subprocess with its own Genesis; keep
  `--n-envs` and the number of concurrent candidates within GPU memory.
* `gemini` needs `google-genai` and `GEMINI_API_KEY`; the LangChain
  providers need the `langchain` extra. The scripted provider is the path
  that has been exercised end to end offline.
* A checkpoint that does not stand in the demo is not a bug in the demo:
  the success test mirrors the task's own criterion.

---

## `slam/slam_viz.py`: shared helpers for the SLAM examples

Not a command-line tool. It holds the matplotlib and dashboard plumbing both
SLAM examples share and is kept out of `domo/` because it imports
matplotlib. It never selects a backend (the examples do, `Agg` unless
`--live`) and never imports the dashboard.

| Helper | Signature | Purpose |
|--------|-----------|---------|
| `slam_snapshot` | `slam_snapshot(step, state, slam, lidar=None, cloud_pts=500, status="running") -> dict` | a JSON-serialisable telemetry dict for the dashboard: `t`, `status`, `pose`, `base`, `dof`, `vel`, `height`, `coverage`, `lidar` (when readable), `map` (rows of `#`, `.`, space, max-pooled to about 60 rows), `cloud` (a random subset of `cloud_pts` points) and `bounds` |
| `go2_robot_manifest` | `go2_robot_manifest(spec) -> (robot_dict, root_dir)` | the `"robot"` entry of a scene manifest (`urdf` under `/assets/robot/`, `dof_names` in joint order) and the directory to register as the `robot` asset root; locates Genesis's bundled URDF when the spec path is relative |
| `save_point_cloud` | `save_point_cloud(cloud, path, title="SLAM point cloud")` | a 3D scatter PNG of the world-frame cloud, coloured by height |
| `SlamLiveView` | `SlamLiveView(every=40, max_pts=6000, title="SLAM reconstruction (live)")`, `.update(slam, state)`, `.keep_open()` | a live top-down matplotlib window with the accumulating cloud and the robot trail; a no-op when matplotlib is on `Agg` or no display is available |
| `SlamRecorder` | `SlamRecorder(every=250, max_pts=15000)`, `.capture(slam, state)`, `.save_gif(path, fps=6, title="SLAM point cloud forming")` | keeps a frame every `every` steps in memory and writes a top-down GIF with fixed axes at the end |
