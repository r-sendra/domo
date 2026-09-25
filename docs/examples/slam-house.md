# SLAM in an apartment

The same passive mapping layer, in a cluttered, realistic environment. A
planner authors a blind wander through a Habitat ReplicaCAD apartment while
SLAM maps everything the lidar sees. The map comes out with honest gaps, which
is the point.

## What it demonstrates

`examples/slam/slam_replica_house.py` is library-native. Where
[the arena demo](slam-demo.md) drives a fixed patrol, this one hands the
driving to a `PlanningController` that re-plans every few seconds and ticks
SLAM from inside its own `update()`, so the map persists across programs.

-   [`domo.world`](../api/world-and-services.md#domoworld): `World` with
    `scene_kind="replica"`, the XT16 lidar and an optional offscreen camera.
-   [`domo.control`](../api/control.md#slam): `SlamSkill` and `SlamConfig`,
    identical to the arena.
-   [`domo.skills`](../api/skills.md#planning): `PlanningController`,
    `make_go2_library` with capped avoid authority.
-   [`domo.policies`](../api/world-and-services.md#domopolicies): both `walk`
    and `avoid` default to the stable registry.

`HouseWanderer` alternates two kinds of leg, for up to 80 legs:

```
(avoid @ walk(vx=0.4)).for(6)        # forward under lidar avoidance
walk(vyaw=±0.8).for(2)               # turn a random way
```

When a forward leg stalls — avoid's `blocked` condition, or a tip — the leg
fails and the next plan turns away, so the wanderer backs out of dead-ends on
its own. Avoid's forward authority is capped with
`avoid_deltas = (0.2, 0.5, 1.2)`, so it steers around furniture without ever
cancelling the walk; the conservative policy would otherwise freeze the robot
in a cluttered house.

Because the wander is blind, coverage stays partial. That gap is the
motivation for a frontier explorer that *reads* SLAM's map and chooses where
to look next — the researcher stand-in `HouseWanderer` is explicitly a
placeholder for it.

## Run it

=== "CPU (laptop)"

    ```bash
    # stable walk + avoid; saves house_cloud.png
    python examples/slam/slam_replica_house.py --headless --device cpu
    ```

=== "Live plot / GIF / dashboard"

    ```bash
    python examples/slam/slam_replica_house.py --headless --device cpu --live
    python examples/slam/slam_replica_house.py --headless --device cpu --animate house.gif
    python examples/slam/slam_replica_house.py --headless --device cpu --dashboard 8080
    ```

=== "GPU with the viewer"

    ```bash
    python examples/slam/slam_replica_house.py
    ```

## How it works

-   `build_world(args, device)` returns a `World` configured with
    `scene_kind="replica"`, `ground_height=0.2`, spawn `(3, -3, 0.44)` at yaw
    180°, `lidar_model=hesai_xt16(n_horizontal=args.azimuth)` and a camera
    resolution only when the in-process dashboard wants one.
-   `build_brain(args, world)` loads walk and avoid from `--cpg-checkpoint` /
    `--avoid-checkpoint` or the stable registry, builds the library with
    `avoid_deltas=(0.2, 0.5, 1.2)`, creates `SlamSkill(world.lidar,
    SlamConfig(resolution=0.1, half_extent=10.0, origin=<spawn>,
    z_band=(0.4, 1.6), map_max_range=10.0))` and the `HouseWanderer`.
-   `HouseWanderer.update(state, dt)` calls `slam.update_command(state, dt)`
    before delegating to `PlanningController.update`, which is what keeps the
    map alive across program boundaries.
-   `attach_dashboard(args)` works as in the arena demo, but the 3D scene
    holds only the robot — the house meshes are not pushed yet.
-   `run(...)` is the arena demo's loop with one addition: a dashboard `reset`
    also calls `controller.restart()`. The run ends when the wanderer idles
    after its last leg, or the budget is spent.
-   `report(slam, out_path)` prints the ASCII map and coverage, then saves the
    cloud PNG.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--cpg-checkpoint` | stable `walk` | CPG walk checkpoint |
| `--avoid-checkpoint` | stable `avoid` | avoidance checkpoint |
| `--scene-json` | `scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json` | ReplicaCAD scene instance |
| `--asset-root` | `scripts/house_scene/data/replica_cad/` | ReplicaCAD asset root |
| `--azimuth` | `360` | XT16 azimuth samples (multiple of 36) |
| `--steps` | `10000` | step budget |
| `--seed` | `0` | seeds `random` and torch — the wander's turn directions |
| `--out` | `house_cloud.png` | path for the point-cloud PNG |
| `--animate` | none | path for a GIF animating the cloud forming |
| `--animate-every` | `250` | capture an animation frame every N steps |
| `--live` | off | show a separate live plot while the sim runs (keep `--headless`) |
| `--live-every` | `50` | refresh the live plot every N steps |
| `--dashboard PORT` | `0` | host the dashboard in-process on this port (0 = off) |
| `--dashboard-url URL` | none | push to a decoupled dashboard server |
| `--dashboard-camera` | off | add the Genesis camera frame (small sim-thread cost) |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

## What you will see

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

Saved: the scatter PNG at `--out`, and the GIF at `--animate` if requested.

## Runtime

| | Build | 10 000 steps |
|---|---|---|
| CPU | about 2 min | about 5 min |
| GPU | tens of seconds | much faster |

## Known limitations

!!! warning "The apartment rebuilds from scratch on every run"

    Loading the ReplicaCAD meshes takes about two minutes on CPU before the
    first control step, every time. Use [the arena demo](slam-demo.md) when
    you want a quick look at the same mapping layer, and run this one in the
    background rather than waiting on it.

    The assets themselves are vendored in the repository under
    `scripts/house_scene/data/replica_cad/` (the default `--asset-root`), so
    a fresh clone has them. `scripts/house_scene/data/download_data.py`
    re-fetches them from the `ai-habitat/ReplicaCAD_dataset` dataset if they
    are ever missing.

-   The same ++ctrl+c++, `--live` and `--dashboard-camera` caveats as
    [the arena demo](slam-demo.md#known-limitations). In addition, the
    dashboard's 3D panel shows the robot on an empty floor, because the house
    meshes are not part of the manifest.
-   The wander is blind. Coverage depends on `--seed` and stays partial by
    construction.

For the reactive half of this example studied on its own — training the
avoidance network, and running it in the same apartment with a reward — see
[obstacle avoidance](avoidance.md) and
[avoidance in a house](avoid-house.md).
