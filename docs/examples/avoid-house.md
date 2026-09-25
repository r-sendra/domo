# Avoidance in a house

The same lidar-avoidance architecture as the arena, moved into the Habitat
ReplicaCAD apartment. Run it when you want the avoidance skill exercised
against real furniture and real doorways instead of scattered cylinders.

## What it demonstrates

`examples/avoidance/go2_cpg_rl_avoid_house.py` is the library replica of
`scripts/house_scene/go2_cpg_rl_avoid_house.py`. It imports `evaluate`,
`make_trainer`, `lidar_model_from_args` and `add_common_args` from
[the arena script](avoidance.md); only `build_configs` differs. That is the
whole point of the file: the scene is a config field, not a code path.

-   [`domo.tasks`](../api/tasks.md#go2avoidtask): `Go2AvoidTask` with
    `scene_kind="replica"`.
-   [`domo.scenes`](../api/world-and-services.md#domoscenes) builds the
    apartment from the ReplicaCAD scene instance, via the task.
-   `domo.rl` and `domo.checkpoints`, exactly as in the arena.
-   The house geometry from the original experiment: `ground_height = 0.2`,
    spawn at `(3, -3, 0.44)` with yaw 180°.

## Run it

=== "Evaluate (CPU)"

    ```bash
    python examples/avoidance/go2_cpg_rl_avoid_house.py \
        --eval runs/go2_avoid_house/checkpoint_final.pt \
        --cpg-checkpoint policies/walk.pt --headless
    ```

=== "Train (GPU)"

    ```bash
    python examples/avoidance/go2_cpg_rl_avoid_house.py \
        --cpg-checkpoint policies/walk.pt \
        --scene-json scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json \
        --asset-root scripts/house_scene/data/replica_cad/ \
        --n-envs 1024 --device cuda --headless
    ```

    `--n-envs` defaults to 1024 rather than 4096: the house is heavier than
    the arena.

## How it works

-   `build_configs(args)` returns a `Go2AvoidConfig` with
    `scene_kind="replica"`, `replica_scene_json`, `replica_asset_root`,
    `ground_height=0.2`, `base_init_pos=(3.0, -3.0, 0.44)` and
    `base_init_yaw_deg=180.0`, plus the same PPO recipe as the arena.
-   `parse_args()` adds `--scene-json` and `--asset-root`, then calls the
    arena's `add_common_args(p)` so `--resume`, `--eval`, `--vx`, `--lidar`,
    `--lidar-azimuth` and `--draw-lidar` keep identical names, defaults and
    help text.
-   `main()` dispatches to the arena script's `evaluate(...)` for `--eval` —
    which rebuilds the house from the config stored in the checkpoint — or to
    `make_trainer(args, policy_fn, build_configs).train()`.

Everything else, including `AvoidanceMission` and the
`(avoid @ walk(vx=0.6)).for(20)` evaluation program, is described on
[the arena page](avoidance.md#how-it-works).

## Flags

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

## What you will see

The same console output as [the arena script](avoidance.md#what-you-will-see):
a status line every 50 steps, `Episode k/5 | SUCCESS` or `FAILURE` with the
program trace, and a final success count. Training writes checkpoints and
TensorBoard logs under `runs/go2_avoid_house`.

## Runtime

!!! warning "Building the apartment takes about two minutes on CPU"

    That is for a *single* environment, before any stepping happens, and it
    happens on every run. Stepping is slower than the arena too. Budget for
    it, and do not use this example as a quick smoke test — the
    [arena version](avoidance.md) is the one that runs in 110 s.

On a GPU the build is tens of seconds. Training is a GPU job in any case.

## Known limitations

-   The ReplicaCAD meshes are vendored in the repository under the default
    `--asset-root` (`scripts/house_scene/data/replica_cad/`), so no download
    is needed; `scripts/house_scene/data/download_data.py` re-fetches them if
    they go missing. The apartment still costs about two minutes of build time
    on every run.
-   No stable house checkpoint exists in `policies/`, so `--eval` needs a run
    of your own. The registry only blesses `walk` and `avoid` (see
    [running](../guides/running.md#stable-policies)).
-   The arena script's evaluation limitations apply: the device is auto-picked
    and `--device` ignored, and `--cpg-checkpoint` should always be passed.

For a passive pass over the same apartment — mapping it instead of dodging
through it — see [SLAM in an apartment](slam-house.md), which needs no house
checkpoint because it drives with the stable `walk` and `avoid` policies.
