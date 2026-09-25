# Locomotion (CPG-RL)

The motor skill everything else stands on. A PPO policy modulates a
central-pattern-generator gait so the Go2 tracks a commanded body velocity
`(vx, vy, vyaw)`. Train it, resume it, or evaluate a checkpoint and watch the
tracking error and stride period.

## What it demonstrates

`examples/locomotion/go2_cpg_rl.py` is the library replica of
`scripts/house_scene/go2_cpg_rl.py` and keeps its CLI. The task owns the
scene, the rewards, the resets and the CPG; the trainer owns the
optimisation; the script is only configuration and a small evaluation
routine. The network it produces is the `walk` entry of the stable-policy
registry and is loaded everywhere else with `load_locomotion_policy`.

-   [`domo.tasks`](../api/tasks.md#go2cpgwalktask): `Go2CPGWalkTask`,
    `Go2CPGWalkConfig`.
-   [`domo.rl`](../api/rl.md#ppotrainer): `PPOTrainer`, `PPOConfig`.
-   `domo.checkpoints`: `configs_from_checkpoint` and `load_checkpoint` read
    both the new and the legacy script checkpoint formats.
-   No skills and no grammar: this is the skill-learning runtime, below the
    [control hierarchy](../concepts/control-hierarchy.md).

## Run it

=== "Evaluate (CPU)"

    ```bash
    python examples/locomotion/go2_cpg_rl.py --eval policies/walk.pt --vx 0.5
    ```

    The Genesis viewer opens whatever you pass — see
    [known limitations](#known-limitations).

=== "Train (GPU)"

    ```bash
    python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless
    ```

=== "Train (CPU smoke)"

    ```bash
    # slow, and only checks the pipeline
    python examples/locomotion/go2_cpg_rl.py --n-envs 16 --total-steps 20000 --device cpu
    ```

=== "Resume"

    ```bash
    python examples/locomotion/go2_cpg_rl.py \
        --resume runs/go2_cpg/checkpoint_step_000100000.pt
    ```

## How it works

-   `build_configs(args)` makes a `Go2CPGWalkConfig` (`dt = 0.02`, 1000-step
    episodes) and a `PPOConfig` with the frozen script's hyper-parameters:
    24-step rollouts, 5 epochs, `lr = 3e-4` on a linear schedule,
    `hidden_size = 512`, `target_kl = 0.02`.
-   `make_trainer(args)` returns a fresh
    `PPOTrainer(Go2CPGWalkTask(cfg), ppo_cfg)`, or one restored from
    `--resume` via `configs_from_checkpoint` plus `trainer.load_state`. The
    task config's device is overwritten from `--device`, because a checkpoint
    may come from another machine.
-   `evaluate(checkpoint_path, command, n_episodes=3, headless=False)` builds
    one environment with command resampling disabled and runs three episodes
    of `run_episode`, which re-pins the command every step (the task would
    otherwise resample it) and records the velocity error and the CPG phases.
-   `gait_period(theta_hist, ep_len, dt)` derives the stride period from the
    phase wraps of oscillator 0, and returns `nan` if there were none.
-   `main()` dispatches: `--eval` evaluates and returns, otherwise
    `make_trainer(args).train()`.

## Flags

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

## What you will see

Training prints the `PPOTrainer` progress and writes
`<run-dir>/checkpoint_step_<step:09d>.pt` at the trainer's save interval,
`<run-dir>/checkpoint_final.pt` at the end, and TensorBoard event files under
`--run-dir`.

Evaluation prints a tracking line every 50 steps and a summary per episode:

```
    step   50  vx=+0.47/0.50  vy=-0.01  wz=+0.02  h=0.31  r=[1.00 0.98 1.00 0.99]
    ...
  Episode 1 | return= 812.40 | length=1000 | mean |vx err|=0.041 m/s | gait period~0.62s
```

An episode ends when the task terminates it — a fall, or the 1000-step cap.
Evaluation saves nothing.

## Runtime

| | Training | Evaluation |
|---|---|---|
| CPU | impractical beyond a smoke test | about a minute after the build |
| GPU | hours at 4096 environments | seconds |

## Known limitations

!!! warning "`--eval` always opens the Genesis viewer"

    `main()` calls `evaluate(args.eval, command=...)` without forwarding
    `args.headless`, and `evaluate`'s own default is `headless=False`. There
    is no way to evaluate headlessly from the CLI, so on a display-less
    machine — a remote box, a container, a CI runner — the evaluation cannot
    run and will hang or crash on window creation. Evaluate on a machine with
    a display, or call `evaluate(path, headless=True)` from your own Python.

-   `--eval` picks its device with `pick_device("cuda")` and ignores
    `--device`: CUDA when present, CPU otherwise.
-   `--headless` cannot be turned *off* for training — it is `store_true` with
    default `True`, mirroring the frozen script.

Once you have a gait, [obstacle avoidance](avoidance.md) trains a correction
network on top of it, and everything in [the twin](twin-demo.md) and
[SLAM](slam-demo.md) loads it as `walk`. For the full training workflow —
monitoring, resuming, promoting a checkpoint — see
[running](../guides/running.md).
