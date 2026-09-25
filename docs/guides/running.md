# Running and training

The operational guide: which entry point trains which policy, what it
costs, how to watch it, how to evaluate and resume, where the checkpoints
land, how one of them becomes a blessed policy the twin can load, and how
to look at a run through the dashboard.

The mechanics behind each command belong to the reference pages —
[domo.tasks](../api/tasks.md), [domo.rl](../api/rl.md) and
[domo.world & services](../api/world-and-services.md) — and every script's
full flag table is on its [example page](../examples/index.md). This page
is only about operating them.

## What can be trained

Four policies, four entry points, one trainer
([`PPOTrainer`](../api/rl.md#ppotrainer)).

| Policy | Task | Entry point | Run dir | Default budget |
|--------|------|-------------|---------|----------------|
| joint-space walk | `Go2WalkTask` (45 / 12) | `main.py` | `runs/go2_walk` | 100 M steps @ 4096 envs |
| CPG walk | `Go2CPGWalkTask` (76 / 12) | `examples/locomotion/go2_cpg_rl.py` | `runs/go2_cpg` | 80 M steps @ 4096 envs |
| lidar avoidance (arena) | `Go2AvoidTask` (36 / 3) | `examples/avoidance/go2_cpg_rl_lidar.py` | `runs/go2_avoidance` | 100 M steps @ 4096 envs |
| lidar avoidance (house) | `Go2AvoidTask` (36 / 3) | `examples/avoidance/go2_cpg_rl_avoid_house.py` | `runs/go2_avoid_house` | 100 M steps @ 1024 envs |
| get-up | `Go2GetUpTask` (42 / 12) | `examples/eureka/eureka_getup.py` | `runs/eureka/getup` | 5 M steps @ 2048 envs, per candidate |

The CPG walk is the gait everything else stands on: it becomes
`policies/walk.pt`. Both avoidance scripts train velocity corrections over
a **frozen** CPG walk and therefore need one first (`--cpg-checkpoint`).
The get-up policy is not trained by a hand-written reward at all — Eureka
writes the reward, so that entry point needs an LLM provider or the
offline `scripted` client ([domo.eureka](../api/eureka.md#quick-start)).

## Training

Training needs the `rl` extra, because `PPOTrainer` imports tensorboard
unconditionally:

```bash
pip install -e '.[genesis,dev,rl]'
```

Every training path is headless. `main.py` and the CPG script accept
`--headless` but ignore it (it is `store_true` with `default=True`, so it
is always on); the two avoidance scripts default to off and need it passed
explicitly on a machine without a display.

=== "CPU smoke run"

    ```bash
    # joint-space walk
    python main.py --n-envs 8 --total-steps 4000 --device cpu

    # CPG walk
    python examples/locomotion/go2_cpg_rl.py --n-envs 16 --total-steps 20000 --device cpu

    # lidar avoidance in the arena
    python examples/avoidance/go2_cpg_rl_lidar.py --cpg-checkpoint policies/walk.pt \
        --n-envs 8 --total-steps 4000 --device cpu --headless --run-dir runs/smoke_avoid

    # get-up through Eureka, offline (no API key)
    python examples/eureka/eureka_getup.py --llm scripted --device cpu \
        --n-envs 8 --train-steps 2000 --samples 1 --iterations 1
    ```

=== "GPU, the real run"

    ```bash
    # joint-space walk
    python main.py --n-envs 4096 --device cuda --headless
    python main.py --n-envs 4096 --device cuda --terrain rough --run-dir runs/go2_walk_rough

    # CPG walk — the gait the twin uses
    python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless

    # lidar avoidance over the frozen walk
    python examples/avoidance/go2_cpg_rl_lidar.py --cpg-checkpoint policies/walk.pt \
        --n-envs 4096 --device cuda --headless

    # the same, in the ReplicaCAD apartment
    python examples/avoidance/go2_cpg_rl_avoid_house.py --cpg-checkpoint policies/walk.pt \
        --n-envs 1024 --device cuda --headless

    # get-up through Eureka (needs GEMINI_API_KEY, or --llm vllm / openai)
    python examples/eureka/eureka_getup.py --samples 4 --iterations 3 \
        --train-steps 5000000 --n-envs 2048 --device cuda
    ```

Each script builds a task config and a `PPOConfig` from its flags, wraps
them in `PPOTrainer(env, ppo_cfg, extra_checkpoint_data={"task_config":
asdict(task_cfg)})` and calls `train()`. The network is the shared-trunk
`ActorCritic`: 512-wide with two trunk layers for the two walks, 128-wide
with three trunk layers and 64-wide heads for avoidance, 256-wide for
Eureka candidates. The minibatch is a quarter of the rollout,
`max(n_envs × rollout_steps // 4, 256)`.

### Budgets

A step budget is env steps, not gradient steps: the trainer runs
`total_steps // (rollout_steps × n_envs)` updates. The numbers below are
the script defaults; the wall-clock column is what to expect, not a
guarantee.

| Policy | Budget | Envs | Updates | On a CUDA GPU |
|--------|--------|------|---------|----------------|
| joint-space walk | 100 M | 4096 | ~1000 | hours |
| CPG walk | 80 M | 4096 | ~813 | hours |
| avoidance (arena) | 100 M | 4096 | ~1017 | hours, plus the frozen walk's forward pass every step |
| avoidance (house) | 100 M | 1024 | ~4069 | hours; the apartment build alone is tens of seconds per process |
| Eureka candidate | 4 M (`EurekaConfig` default) or 5 M (the example CLI) | 2048 | ~81 / ~102 | minutes each, ×`iterations × samples`, each in its own Genesis subprocess |

100 M at 4096 envs is the inherited default of `main.py` and of the two
avoidance scripts, not a recommendation: the CPG walk converges inside
80 M and the avoidance net much sooner. Start from 80 M and stop on the
curves rather than on the budget.

On CPU, a smoke test checks the pipeline and nothing else. Use **2–16
envs** and a few thousand steps, as in the tab above. With `--total-steps`
below one rollout (`n_envs × rollout_steps`) the trainer runs zero updates
and writes only `checkpoint_final.pt`, so keep the budget at a few
rollouts.

The stability guards in `PPOConfig` (`target_kl`, `lr_schedule="linear"`,
`guard_nonfinite`, `vloss_skip`) are off by default and are not exposed as
CLI flags; set them in the script's `build_configs` when a run diverges
([`PPOConfig`](../api/rl.md#ppoconfig)).

## Monitoring

`PPOTrainer` prints one line every `log_interval` updates (10 by default):

```
  step  9,830,400 | ret  21.317 | len   987 | ploss -0.0123 | vloss  0.4210 | clip 0.08 |  61,440 sps
```

and writes the same numbers to TensorBoard events in the run dir:

```bash
tensorboard --logdir runs
```

| Tag | What it is | What to watch for |
|-----|------------|-------------------|
| `train/mean_return` | undiscounted episode return, mean over the last `ep_stat_window` (20) finished episodes | should rise; for the walks the tracking terms saturate near `Σ weights × episode_s` |
| `train/mean_ep_len` | same window, in control steps | should climb to the episode cap (1000 for the walks and avoidance, 400 for get-up); early falls keep it low |
| `train/clip_fraction` | share of samples outside the ratio clip | typically 0.05–0.2; a persistent 0.3+ means the policy moves too far per update |
| `loss/policy` | clipped surrogate loss | noisy around zero; its sign says little on its own |
| `loss/value` | clipped value loss | a blow-up here is what `vloss_skip` guards against |
| `loss/entropy` | mean policy entropy, **not** a loss (the trainer logs `−entropy_loss`) | falls slowly; a fast drop to a large negative value means the Gaussian has collapsed and exploration is over |
| `train/steps_per_sec` | throughput since process start | tens of thousands on a GPU at 4096 envs |
| `train/lr` | current Adam learning rate | flat unless `lr_schedule="linear"` |

Per-term reward means are accumulated by the task into
`extras["episode"]["rew_<term>"]` but the trainer does not log them. Add
them through `update_callback` if you need to see which term is winning.

## Evaluation

Evaluation rebuilds the task, or the twin, from the config stored *inside*
the checkpoint, so a checkpoint is self-describing and no flag has to match
the training run.

=== "Headless (CPU, no display)"

    ```bash
    # avoidance: an '(avoid @ walk(vx=0.6)).for(20)' mission in the twin
    python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
        --cpg-checkpoint policies/walk.pt --headless

    # house avoidance
    python examples/avoidance/go2_cpg_rl_avoid_house.py \
        --eval runs/go2_avoid_house/checkpoint_final.pt \
        --cpg-checkpoint policies/walk.pt --headless

    # a learned get-up checkpoint in the twin (three episodes)
    python examples/eureka/eureka_getup.py \
        --demo runs/eureka/getup/iter_2/train_1/checkpoint_final.pt --headless --device cpu
    ```

=== "With the Genesis viewer"

    ```bash
    # joint-space walk: 10 stochastic episodes in one env
    python main.py --eval runs/go2_walk/checkpoint_final.pt

    # CPG walk: three episodes holding a command, tracking printed every 50 steps
    python examples/locomotion/go2_cpg_rl.py --eval policies/walk.pt --vx 0.5 --vy 0.0 --vyaw 0.0

    # avoidance with the rays drawn
    python examples/avoidance/go2_cpg_rl_lidar.py --eval policies/avoid.pt \
        --cpg-checkpoint policies/walk.pt --draw-lidar
    ```

Four things decide what an evaluation actually measures:

* **Stochastic or deterministic.** `main.py --eval` samples from the
  Gaussian, exactly as training does. Every other evaluation takes the
  mean.
* **Device.** The `--eval` branches of `main.py`, `go2_cpg_rl.py` and the
  two avoidance scripts ignore `--device` and call `pick_device("cuda")`:
  CUDA when present, CPU otherwise.
* **Task or twin.** The avoidance evaluations do not run the RL task. They
  spawn the goal-free `World` the checkpoint trained in, build a skill
  library from the two checkpoints and give the robot to a
  `PlanningController` mission, so what you see is what the twin does.
  `--cpg-checkpoint` may be omitted when the avoidance checkpoint recorded
  the walk path it was trained over (`extra["cpg_checkpoint"]`).
* **Sensor model.** `--lidar xt16` swaps the idealised training lidar for
  a simulated Hesai XT16. That is a distribution shift, and success counts
  drop accordingly ([obstacle avoidance](../examples/avoidance.md)).

!!! warning "`go2_cpg_rl.py --eval` always opens the viewer"

    The CPG script does not forward `--headless` to `evaluate()`, so the
    evaluation path builds a windowed scene even when you asked for
    headless. On a display-less machine it cannot run. Evaluate that gait
    through `examples/basic_examples/skill_demo.py` instead, which is
    headless-clean and loads the same checkpoint.

## Resuming

```bash
python main.py --resume runs/go2_walk/checkpoint_step_009830400.pt
python examples/locomotion/go2_cpg_rl.py --resume runs/go2_cpg/checkpoint_step_009830400.pt
python examples/avoidance/go2_cpg_rl_lidar.py --resume runs/go2_avoidance/checkpoint_step_009830400.pt \
    --cpg-checkpoint policies/walk.pt
```

Resuming rebuilds the task and the `PPOConfig` from the checkpoint (only
the device is overridden), restores the weights, the Adam state and the
step counter, and continues. Legacy checkpoints resume too, but only
`n_envs`, `dt`, `max_episode_steps` and `headless` survive their flat
config; every other task field comes from the current dataclass defaults.

!!! danger "`--resume` extends the budget, it does not honour it"

    `PPOTrainer.train()` computes `n_updates = total_steps //
    (rollout_steps × num_envs)` once, at the top of the loop, and never
    subtracts the restored `global_step`. A run resumed at 40 M with
    `total_steps = 80_000_000` therefore trains **80 M more** steps and
    finishes near 120 M. Lower `--total-steps` by hand if you want a
    specific finishing point. The step counter keeps counting from where
    it was, so checkpoint file names never collide.

## Checkpoints

### Where they land

```
runs/<run>/
├── checkpoint_step_000983040.pt     every save_interval updates (100 by default)
├── checkpoint_step_001966080.pt
├── ...
├── checkpoint_final.pt              written at the end of train()
└── events.out.tfevents.*            TensorBoard
```

Eureka runs use a tree instead: `runs/eureka/<skill>/iter_<k>/train_<j>/`
with `spec.json`, `results.json` and `checkpoint_final.pt` per candidate,
the prompts and `reflection.txt` per iteration, and `result.json` at the
root ([run-directory layout](../api/eureka.md#run-directory-layout)).

`runs/` and every `*.pt` are git-ignored.

### The two formats

Two formats coexist, and everything in the library loads both:

| | Library format (`PPOTrainer.save_checkpoint`) | Legacy format (`scripts/`, and today's `policies/*.pt`) |
|---|---|---|
| weights | `model_state` | `model_state`, possibly with `_orig_mod.` / `module.` prefixes |
| optimiser, step, metrics | `optim_state`, `step`, `metrics` | the same keys |
| configs | `ppo_config` (a `PPOConfig` dict) plus `extra["task_config"]` | one flat `config` dict mixing task and PPO fields |
| architecture | from `ppo_config`, or from the weight shapes | from the weight shapes only |

`ActorCritic.from_state_dict(ckpt["model_state"])` rebuilds the network
from the weight shapes alone, and `configs_from_checkpoint` recovers
`(task_cfg, ppo_cfg)` from either format
([checkpoints](../api/rl.md#checkpoints),
[domo.checkpoints](../api/world-and-services.md#domocheckpoints)).

```python
from domo.checkpoints import configs_from_checkpoint, load_checkpoint, load_locomotion_policy

ckpt = load_checkpoint("runs/go2_cpg/checkpoint_final.pt", "cpu")
task_cfg, ppo_cfg = configs_from_checkpoint(ckpt, kind="cpg_walk")   # (1)!
walk_fn, net = load_locomotion_policy("runs/go2_cpg/checkpoint_final.pt", "cpu")
```

1.  `kind` is `"cpg_walk"` or `"avoid"` — it picks which task dataclass to
    rebuild, because the legacy format does not record it.

Loading uses `torch.load(..., weights_only=False)`, because the configs are
pickled dataclasses. Only load checkpoints you trust.

## Stable policies

Training scatters checkpoints across `runs/`. The twin, the SLAM and
navigation examples and the skill library must not depend on a path like
`runs/go2_cpg/checkpoint_final_coupled.pt`, so
[`domo.policies`](../api/world-and-services.md#domopolicies) names the
*blessed* ones and resolves each to a file in `policies/` at the repo root:

| Name | File | Promoted from | Role |
|------|------|---------------|------|
| `walk` | `policies/walk.pt` | `runs/go2_cpg/checkpoint_final_coupled.pt` | the most stable CPG gait |
| `avoid` | `policies/avoid.pt` | `runs/go2_cpg/checkpoint_final_avoid.pt` | the 36-sector lidar avoidance net |

```python
from domo.policies import load_stable_avoid, load_stable_locomotion, stable_go2_library, stable_policy

stable_policy("walk")                      # absolute path; KeyError / FileNotFoundError if missing
walk_fn, net = load_stable_locomotion("cpu")
avoid_fn = load_stable_avoid("cpu")        # obs [N, 36] → Δv [N, 3], deterministic
library = stable_go2_library(world.lidar, device="cpu")
```

`stable_policy` is deliberately import-light — it resolves a path without
pulling in torch. The loaders around it import `domo.checkpoints` and
`domo.rl` lazily, which means they need the `rl` extra even though nothing
is being trained.

The `.pt` files themselves are git-ignored: `policies/` is a curated local
cache, while the registry (`domo/policies.py`) and `policies/README.md` are
versioned. On a fresh clone you have the names and not the weights, so
copy the files from a training run or a colleague — without them, nothing
walks.

### Promoting a checkpoint

Evaluate the candidate, then copy it over the registered file:

```bash
python examples/locomotion/go2_cpg_rl.py --eval runs/go2_cpg/checkpoint_final_newbest.pt --vx 0.5
cp runs/go2_cpg/checkpoint_final_newbest.pt policies/walk.pt
```

A new skill gets a new entry in `STABLE` in `domo/policies.py` and, if it
needs one, a loader next to `load_stable_avoid`
([promote a policy](../api/world-and-services.md#promote-a-policy)). Keep
`policies/README.md` in step. Never change what an existing name means
without telling everyone whose experiments depend on it.

## Running the twin and the demos

The twin-side examples either take the gait explicitly or fall back to the
registry:

```bash
# hand-written controller switching stand / walk (explicit path)
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu --steps 500

# the planner-governed twin with avoidance
python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt \
    --headless --device cpu --steps 1000

# the SLAM patrols default to the stable names
python examples/slam/slam_demo.py --headless --device cpu
python examples/slam/slam_replica_house.py --headless --device cpu
```

In your own code the pattern is world → library → planner → loop:

```python
from domo.policies import stable_go2_library
from domo.skills import PlanningController
from domo.world import World, WorldConfig

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

Every example accepts `--device cpu`. On CPU an arena build takes about
30 s and a ReplicaCAD apartment about two minutes
([examples](../examples/index.md)).

## The dashboard

[`domo.dashboard`](../api/dashboard.md) is a live web view of the twin:
telemetry, a 3D robot, an optional camera frame and pause / reset / stop
buttons. It is for the twin and for evaluation only — it is never on the
training path, and no training script publishes to it.

=== "Decoupled (two processes)"

    ```bash
    python -m domo.dashboard --port 8080
    python examples/slam/slam_demo.py --headless --device cpu --dashboard-url http://127.0.0.1:8080
    ```

=== "In-process"

    ```bash
    python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080 --dashboard-camera
    ```

Open `http://127.0.0.1:8080`. `publish()` on the simulation side only swaps
a snapshot into a slot; a background thread does the network I/O, so the
simulation never waits on the browser. The API, the telemetry schema and
the scene manifest are in [domo.dashboard](../api/dashboard.md); the
failure modes are in [troubleshooting](troubleshooting.md#dashboard).

## Reproducibility

Both global RNGs matter, and they are drawn from by different things:

| Source of randomness | RNG | How to pin it |
|----------------------|-----|---------------|
| commands, spawn poses, get-up poses, obstacle scatter in `ObstacleArena.randomise` | torch global | `torch.manual_seed(s)` before building the task or the world |
| rough-terrain heightfields (`TerrainConfig.randomize=True`, generated inside Genesis) | numpy global | `np.random.seed(s)` before the build, or set `randomize=False` for Genesis' own fixed seed |
| policy initialisation and action sampling | torch global | the same `torch.manual_seed(s)` |

None of the entry points seeds anything by default, so two runs of the same
command differ. Beyond that:

* The draw order inside the tasks — commands `x`, `y`, `yaw`; get-up roll,
  side, yaw, joints — is pinned by `tests/test_tasks_common.py`. A refactor
  that reorders the draws changes every seeded run.
* Config dataclass defaults are part of the checkpoint contract. A
  checkpoint rebuilds its task from the stored config, so changing a
  default silently changes what old legacy-format checkpoints mean (their
  flat config stored only four fields) as well as what new runs do.
  `tests/test_tasks_common.py` pins the defaults for that reason.
* GPU training is not bit-reproducible across runs even when seeded
  (Genesis and cuBLAS nondeterminism). Compare curves, not numbers.
* PD-gain domain randomisation is per reset group, not per env — see
  [troubleshooting](troubleshooting.md#pd-gain-randomisation-is-not-per-environment).
  Friction and mass are per env.
