# Running: training, evaluation, checkpoints

This page is the operational guide: which script trains which policy, what
it costs, how to watch it, how to evaluate and resume, where checkpoints
land, how a checkpoint becomes a blessed policy the twin can use, and how
to look at a run through the dashboard. The mechanics behind each command
are in [api/tasks.md](api/tasks.md), [api/rl.md](api/rl.md) and
[api/world-and-services.md](api/world-and-services.md); every example's
full flag table is in [examples.md](examples.md).

## What can be trained

Four policies, four entry points, one trainer (`domo.rl.PPOTrainer`).

| Policy | Task | Script | Default run dir | Role |
|--------|------|--------|-----------------|------|
| joint-space walk | `Go2WalkTask` (45 / 12) | `main.py` | `runs/go2_walk` | the reference velocity-tracking task; not used by the twin |
| CPG walk | `Go2CPGWalkTask` (76 / 12) | `examples/locomotion/go2_cpg_rl.py` | `runs/go2_cpg` | **the gait everything else runs on**: `policies/walk.pt` |
| lidar avoidance | `Go2AvoidTask` (36 / 3) | `examples/avoidance/go2_cpg_rl_lidar.py` (arena), `go2_cpg_rl_avoid_house.py` (ReplicaCAD) | `runs/go2_avoidance`, `runs/go2_avoid_house` | velocity corrections over the frozen CPG walk: `policies/avoid.pt` |
| get-up | `Go2GetUpTask` (42 / 12) | `examples/eureka/eureka_getup.py` | `runs/eureka/getup` | learned through Eureka (LLM-written reward), deployed as a `LearnedJointSkill` |

The avoidance scripts need a trained CPG walk first (`--cpg-checkpoint`);
the get-up routine needs an LLM provider or the `scripted` client for an
offline smoke run ([api/eureka.md](api/eureka.md#quick-start)).

## Training

Training needs the `rl` extra (`tensorboard`): `pip install -e '.[dev,rl]'`.
Every training script is headless; the CPG script and `main.py` accept
`--headless` but ignore it (it is always on), the avoidance scripts need
it passed explicitly on a machine without a display.

```bash
# joint-space walk (main.py)
python main.py --n-envs 4096 --device cuda --headless
python main.py --n-envs 4096 --device cuda --terrain rough --run-dir runs/go2_walk_rough

# CPG walk — the gait the twin uses
python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless

# lidar avoidance in the arena, over the frozen walk
python examples/avoidance/go2_cpg_rl_lidar.py --cpg-checkpoint policies/walk.pt \
    --n-envs 4096 --device cuda --headless

# lidar avoidance in the ReplicaCAD house
python examples/avoidance/go2_cpg_rl_avoid_house.py --cpg-checkpoint policies/walk.pt \
    --n-envs 1024 --device cuda --headless

# get-up through Eureka (needs GEMINI_API_KEY, or --llm vllm / openai / scripted)
python examples/eureka/eureka_getup.py --samples 4 --iterations 3 \
    --train-steps 5000000 --n-envs 2048 --device cuda
```

Each script builds a task config and a `PPOConfig` from its flags, wraps
them in `PPOTrainer(env, ppo_cfg, extra_checkpoint_data={"task_config":
asdict(task_cfg)})` and calls `train()`. The network is the shared-trunk
`ActorCritic`: 512-wide, two trunk layers for the two walks; 128-wide,
three trunk layers, 64-wide heads for avoidance; 256-wide for Eureka
candidates. The minibatch is a quarter of the rollout
(`max(n_envs × rollout_steps // 4, 256)`).

### Budgets

The step budgets below are the script defaults; the wall-clock figures are
what to expect, not guarantees.

| Policy | Default budget | Envs | CUDA GPU | CPU |
|--------|----------------|------|----------|-----|
| joint-space walk | 100 M steps | 4096 | hours | smoke test only |
| CPG walk | 80 M steps | 4096 | hours | smoke test only |
| avoidance (arena) | 100 M steps | 4096 | hours (plus the frozen walk's forward pass every step) | smoke test only |
| avoidance (house) | 100 M steps | 1024 | hours; the house build alone is tens of seconds per process | smoke test only (two-minute build) |
| get-up (Eureka) | `samples × iterations` candidates of 5 M steps each (4 M with `EurekaConfig` defaults) | 2048 per candidate | hours; each candidate is its own Genesis subprocess | smoke test only |

A CPU smoke test checks the pipeline, not the policy: 2–16 envs and a few
thousand steps.

```bash
python main.py --n-envs 8 --total-steps 4000 --device cpu
python examples/locomotion/go2_cpg_rl.py --n-envs 16 --total-steps 20000 --device cpu
python examples/avoidance/go2_cpg_rl_lidar.py --cpg-checkpoint policies/walk.pt \
    --n-envs 8 --total-steps 4000 --device cpu --headless --run-dir runs/smoke_avoid
python examples/eureka/eureka_getup.py --llm scripted --device cpu \
    --n-envs 8 --train-steps 2000 --samples 1 --iterations 1
```

With `--total-steps` smaller than one rollout (`n_envs × rollout_steps`)
the trainer runs zero updates and writes only `checkpoint_final.pt`; keep
the budget at a few rollouts.

`PPOConfig` guards (`target_kl`, `lr_schedule="linear"`, `guard_nonfinite`,
`vloss_skip`) are off by default and not exposed as CLI flags; set them in
the script's `build_configs` when a run diverges
([api/rl.md](api/rl.md#ppoconfig)).

## Monitoring

`PPOTrainer` prints one line every `log_interval` updates (10 by default)

```
  step  9,830,400 | ret  21.317 | len   987 | ploss -0.0123 | vloss  0.4210 | clip 0.08 |  61,440 sps
```

and writes TensorBoard events into the run dir:

```bash
tensorboard --logdir runs
```

| Tag | Watch for |
|-----|-----------|
| `train/mean_return` | should rise; for the walks the tracking terms saturate near `Σ weights × episode_s` |
| `train/mean_ep_len` | should reach the budget (1000 for the walks and avoidance, 400 for get-up); early falls keep it low |
| `train/clip_fraction` | typically 0.05–0.2; a persistent 0.3+ means the policy is moving too fast per update |
| `loss/value` | a blow-up here is what `vloss_skip` guards against |
| `loss/entropy` | decreasing slowly; a collapse to a very negative value means the policy has gone deterministic too early |
| `train/steps_per_sec` | throughput; tens of thousands on a GPU with 4096 envs |
| `train/lr` | flat unless `lr_schedule="linear"` |

Per-term reward means are available in the task's
`extras["episode"]["rew_<term>"]` but are not logged by the trainer; add
them in `update_callback` if you need them.

## Evaluation

Evaluation rebuilds the task (or the twin) from the config stored inside
the checkpoint, so a checkpoint is self-describing: no flag has to match
the training run.

```bash
# joint-space walk: 10 stochastic episodes in one env, viewer on
python main.py --eval runs/go2_walk/checkpoint_final.pt

# CPG walk: three episodes holding a command; prints tracking every 50 steps
python examples/locomotion/go2_cpg_rl.py --eval policies/walk.pt --vx 0.5 --vy 0.0 --vyaw 0.0

# avoidance: an '(avoid @ walk(vx=0.6)).for(20)' mission in the twin, obstacles re-scattered per episode
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
    --cpg-checkpoint policies/walk.pt --headless
python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
    --cpg-checkpoint policies/walk.pt --draw-lidar            # viewer + rays

# house avoidance
python examples/avoidance/go2_cpg_rl_avoid_house.py --eval runs/go2_avoid_house/checkpoint_final.pt \
    --cpg-checkpoint policies/walk.pt --headless

# a learned get-up checkpoint in the twin (three episodes)
python examples/eureka/eureka_getup.py --demo runs/eureka/getup/iter_2/train_1/checkpoint_final.pt \
    --headless --device cpu
```

Points worth knowing:

* `main.py --eval` uses **stochastic** actions (as in training) and always
  opens the viewer; the other evaluations are deterministic (the Gaussian
  mean).
* `go2_cpg_rl.py --eval` does not forward `--headless`, so it always opens
  the viewer and cannot run on a display-less machine.
* The `--eval` branches of `main.py`, `go2_cpg_rl.py` and the two avoidance
  scripts ignore `--device`: they take CUDA when available, CPU otherwise
  (`pick_device("cuda")`).
* The avoidance evaluations run the checkpoint as a skill inside a
  `PlanningController` mission in the goal-free `World`, not inside the
  task, so what you see is what the twin does. `--cpg-checkpoint` may be
  omitted: the avoidance checkpoint stores the walk path it was trained
  over (`extra["cpg_checkpoint"]`).
* `--lidar xt16` on the avoidance evaluation swaps the idealised training
  lidar for the simulated Hesai XT16; expect a distribution shift
  ([examples.md](examples.md#avoidancego2_cpg_rl_lidarpy-lidar-avoidance-in-a-walled-arena)).

## Resuming

```bash
python main.py --resume runs/go2_walk/checkpoint_step_009830400.pt
python examples/locomotion/go2_cpg_rl.py --resume runs/go2_cpg/checkpoint_step_009830400.pt
python examples/avoidance/go2_cpg_rl_lidar.py --resume runs/go2_avoidance/checkpoint_step_009830400.pt \
    --cpg-checkpoint policies/walk.pt
```

Resume rebuilds the task and the `PPOConfig` from the checkpoint (only the
device is overridden), restores weights, optimiser state and the step
counter, and continues. Two things to keep in mind:

* **The budget restarts.** `PPOTrainer.train()` runs
  `total_steps // (rollout_steps × n_envs)` updates regardless of the
  restored step, so a resumed run trains `total_steps` *more* steps. The
  step counter keeps counting, so file names do not collide.
* **Legacy checkpoints resume too**, but only `n_envs`, `dt`,
  `max_episode_steps` and `headless` are recovered from their flat config;
  every other task field comes from the current dataclass defaults.

## Checkpoints

### Where they are

```
runs/<run>/
├── checkpoint_step_000983040.pt     every save_interval updates (100 by default)
├── checkpoint_step_001966080.pt
├── ...
├── checkpoint_final.pt              at the end of train()
└── events.out.tfevents.*            TensorBoard
```

Eureka runs use a tree: `runs/eureka/<skill>/iter_<k>/train_<j>/` with
`spec.json`, `results.json` and `checkpoint_final.pt` per candidate, the
prompts and `reflection.txt` per iteration, and `result.json` at the root
([api/eureka.md](api/eureka.md#run-directory-layout)).

`runs/` and every `*.pt` are git-ignored.

### Formats

Two formats coexist and everything in the library loads both:

| | Library format (`PPOTrainer.save_checkpoint`) | Legacy format (`scripts/`, and today's `policies/*.pt`) |
|---|---|---|
| weights | `model_state` | `model_state` (may carry `_orig_mod.` / `module.` prefixes) |
| optimiser, step, metrics | `optim_state`, `step`, `metrics` | same |
| configs | `ppo_config` (dict of `PPOConfig`) + `extra["task_config"]` (dict of the task dataclass) | one flat `config` dict mixing task and PPO fields |
| architecture | from `ppo_config` or from weight shapes | from weight shapes only |

`ActorCritic.from_state_dict(ckpt["model_state"])` rebuilds the network
from the weight shapes, and `domo.checkpoints.configs_from_checkpoint(ckpt,
kind="cpg_walk" | "avoid")` recovers `(task_cfg, ppo_cfg)` from either
format ([api/rl.md](api/rl.md#checkpoints),
[api/world-and-services.md](api/world-and-services.md#domocheckpoints)).

```python
from domo.checkpoints import load_checkpoint, load_locomotion_policy, configs_from_checkpoint

ckpt = load_checkpoint("runs/go2_cpg/checkpoint_final.pt", "cpu")
task_cfg, ppo_cfg = configs_from_checkpoint(ckpt, kind="cpg_walk")
walk_fn, net = load_locomotion_policy("runs/go2_cpg/checkpoint_final.pt", "cpu")   # obs [N,76] → action [N,12]
```

Loading uses `torch.load(..., weights_only=False)` because the configs are
pickled dataclasses; only load checkpoints you trust.

## Stable policies

Training scatters checkpoints across `runs/`. The twin, the SLAM and
navigation examples and the skill library should not depend on a path
like `runs/go2_cpg/checkpoint_final_coupled.pt`, so `domo.policies` names
the *blessed* ones and expects a copy of each in `policies/` at the repo
root:

| Name | File | Promoted from | Role |
|------|------|---------------|------|
| `walk` | `policies/walk.pt` | `runs/go2_cpg/checkpoint_final_coupled.pt` | the most stable CPG gait |
| `avoid` | `policies/avoid.pt` | `runs/go2_cpg/checkpoint_final_avoid.pt` | the 36-sector lidar avoidance net |

```python
from domo.policies import stable_policy, load_stable_locomotion, load_stable_avoid, stable_go2_library

stable_policy("walk")                      # absolute path; KeyError / FileNotFoundError when missing
walk_fn, net = load_stable_locomotion("cpu")
avoid_fn = load_stable_avoid("cpu")        # obs [N,36] → Δv [N,3], deterministic
library = stable_go2_library(world.lidar, device="cpu")   # walk (+ avoid, slam, blocked/clear with a lidar)
```

The `.pt` files are git-ignored: `policies/` is a curated local cache, the
registry (`domo/policies.py`) and `policies/README.md` are versioned. On a
fresh clone, copy the two files from a training run or a colleague;
without them nothing walks.

### Promoting a checkpoint

Evaluate it, then copy it over the registered file:

```bash
python examples/locomotion/go2_cpg_rl.py --eval runs/go2_cpg/checkpoint_final_newbest.pt --vx 0.5
cp runs/go2_cpg/checkpoint_final_newbest.pt policies/walk.pt
```

A new skill gets a new entry in `STABLE` in `domo/policies.py` and, if it
needs one, a loader next to `load_stable_avoid`
([api/world-and-services.md](api/world-and-services.md#promote-a-policy)).
Keep `policies/README.md` in step. Never change what an existing name
means without telling everyone whose experiments depend on it.

## Running the twin and the demos

The twin-side examples take the gait explicitly or default to the
registry:

```bash
# hand-written controller switching stand / walk (explicit path)
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu --steps 500

# the planner-governed twin with avoidance
python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt --headless --device cpu --steps 1000

# SLAM patrols default to the stable names
python examples/slam/slam_demo.py --headless --device cpu
python examples/slam/slam_replica_house.py --headless --device cpu
```

In your own code, the pattern is world → library → planner → loop:

```python
from domo.world import World, WorldConfig
from domo.policies import stable_go2_library
from domo.skills import PlanningController

world = World(WorldConfig(scene_kind="arena", device="cpu", headless=True))
library = stable_go2_library(world.lidar, device="cpu")

class Patrol(PlanningController):
    def plan(self, state, last):
        return None if last is not None else "(avoid @ walk(vx=0.6)).for(20) >> stand.for(1)"

controller = Patrol(library)
controller.setup(world.robot)
loop = world.make_loop(controller)
loop.reset()
for _ in range(1500):
    loop.step()
```

Every example accepts `--device cpu`; on CPU an arena build takes about
30 s and a ReplicaCAD apartment about two minutes
([examples.md](examples.md#cpu-or-gpu)).

## The dashboard, in brief

`domo.dashboard` is a live web view of the twin: telemetry, a 3D robot,
an optional camera frame, and pause / reset / stop buttons. It is for the
twin and evaluation only, never training. Two ways to use it:

```bash
# decoupled: the server in one shell, the sim pushes to it from another
python -m domo.dashboard --port 8080
python examples/slam/slam_demo.py --headless --device cpu --dashboard-url http://127.0.0.1:8080

# in-process: the example hosts the server itself
python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080 --dashboard-camera
```

Open `http://127.0.0.1:8080`. `publish()` on the sim side only swaps a
snapshot into a slot; a background thread does the network I/O, so the
simulation never waits on the browser. The full API, the telemetry
schema and the scene manifest are in [api/dashboard.md](api/dashboard.md);
the common failure modes are in
[troubleshooting.md](troubleshooting.md#dashboard).

## Reproducibility

* Seed both RNGs. The tasks draw commands, spawns and get-up poses with
  `torch.rand`; rough terrain generation and arena obstacle placement use
  numpy's global RNG. `torch.manual_seed(s)` and `np.random.seed(s)`
  before building the task cover both. None of the entry points seeds by
  default.
* The draw order inside the tasks (commands x, y, yaw; get-up roll, side,
  yaw, joints) is pinned by `tests/test_tasks_common.py`; refactors must
  not change it.
* The config dataclass defaults are part of the checkpoint contract: a
  checkpoint rebuilds its task from the stored config, so changing a
  default changes what old checkpoints mean when the field was not stored
  (legacy format) and what new runs do. `tests/test_tasks_common.py` pins
  the defaults.
* GPU training is not bit-reproducible across runs even when seeded
  (Genesis and cuBLAS nondeterminism); compare curves, not numbers.
* PD-gain domain randomisation is per reset group, not per env, on Genesis
  ([troubleshooting.md](troubleshooting.md#genesis)); friction and mass
  are per env.
