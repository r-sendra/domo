"""
SLAM demo: the robot patrols an obstacle arena while a passive SLAM layer
builds an occupancy map of what it sees.

`SlamSkill` is the perception layer — it reads the lidar + pose each tick and
folds them into a log-odds occupancy grid, but contributes ZERO velocity, so
it never steers. A plain `goto` patrol loop drives the robot around the
perimeter (so the lidar sweeps the central block from every side); SLAM rides
on top and maps.

High fidelity: the robot carries a simulated **Hesai XT16** (16 channels ×
360 azimuth = 5760 beams/scan). SLAM consumes the raw 3D world points, maps a
robot-height slice into the occupancy grid, and accumulates the full 3D point
cloud — saved as a scatter plot at the end (`--out`).

Why SLAM is hosted as a persistent observer here (not `slam @ leg @ walk` per
leg): a composition resets each layer when its node is (re)entered, which would
wipe the map at every waypoint. Mapping is a process that must persist across
the whole mission, so it lives at the loop level. (`slam @ goto @ walk` still
composes fine for single-leg, per-leg mapping.)

    # uses the stable 'walk' policy by default; saves a cloud PNG
    python examples/slam/slam_demo.py --headless --device cpu --out slam_cloud.png
    # watch the reconstruction build LIVE in a separate window (Genesis headless)
    python examples/slam/slam_demo.py --headless --device cpu --live
    # serve a live web dashboard (open http://127.0.0.1:8080); +camera panel
    python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080 --dashboard-camera
"""

import argparse
import time

import torch

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.control import (SimControlLoop, SingleSkillController, SlamConfig,
                          SlamSkill)
from domo.policies import stable_policy
from domo.robot import GO2, Robot, SimulatedLidar
from domo.robot.lidar_models import hesai_xt16
from domo.sim import SimConfig, ViewerConfig, create_engine
from domo.skills import make_go2_library
from slam_viz import (SlamLiveView, SlamRecorder, go2_robot_manifest,
                      save_point_cloud, slam_snapshot)
# NOTE: domo.dashboard (all web server/client code) is imported lazily, only
# when --dashboard/--dashboard-url is set — no web code loads otherwise.

# Ground-truth arena. '.' = free (shown as space), R = spawn; the rest are
# obstacles of different HEIGHTS (below), so the 3D cloud shows real structure
# and the 2D occupancy map shows how flattening loses it:
#   #  0.9 m block/wall     L  0.9 m L-shaped wall
#   T  1.8 m tall pillar    o  0.35 m low step (below the lidar → 2D-invisible)
# Cell = 1 m, world x = col, y = row. Obstacles are interior; the perimeter is
# clear so the patrol loops without collision — but each obstacle casts an
# occlusion shadow the single loop can't fill (that's the exploration problem).
ARENA = [
    "R...........",
    "............",
    "..####......",
    "..####......",
    ".........T..",
    "............",
    ".LL.....oo..",
    ".L..........",
    "............",
]
CELL = 1.0
WALL_H = 1.0
HEIGHTS = {"#": 0.9, "T": 1.8, "L": 0.9, "o": 0.35}
BOX_COLORS = {"#": "#8b949e", "T": "#d29922", "L": "#3fb950", "o": "#6e7681"}


def arena_manifest_boxes(obstacles, h, w):
    """Box specs (size/pos/color) for the 3D viewer — mirrors build_arena()."""
    boxes = [{"size": [CELL, CELL, HEIGHTS[ch]],
              "pos": [ox * CELL, oy * CELL, HEIGHTS[ch] / 2],
              "color": BOX_COLORS.get(ch, "#8b949e")}
             for ox, oy, ch in obstacles]
    wx0, wx1 = -1.5 * CELL, (w - 0.5) * CELL
    wy0, wy1 = -1.5 * CELL, (h - 0.5) * CELL
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    sx, sy, t, wh = wx1 - wx0, wy1 - wy0, 0.1, WALL_H
    for size, pos in [((sx, t, wh), (cx, wy0, wh / 2)),
                      ((sx, t, wh), (cx, wy1, wh / 2)),
                      ((t, sy, wh), (wx0, cy, wh / 2)),
                      ((t, sy, wh), (wx1, cy, wh / 2))]:
        boxes.append({"size": list(size), "pos": list(pos), "color": "#484f58"})
    return boxes

# Perimeter patrol (world coords) — tours all four sides of the arena.
PATROL = ("goto(x=11, y=0) @ walk >> goto(x=11, y=8) @ walk >> "
          "goto(x=0, y=8) @ walk >> goto(x=0, y=0) @ walk")


def parse_arena(rows):
    obstacles, start = [], None
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in HEIGHTS:
                obstacles.append((x, y, ch))
            elif ch == "R":
                start = (x, y)
    return obstacles, start, len(rows), max(len(r) for r in rows)


def render_truth(rows):
    out = ["    " + "".join(str(x % 10) for x in range(len(rows[0]))) + "  (x)"]
    for y, row in enumerate(rows):
        out.append(f"y={y} |" + "".join(" " if c == "." else c for c in row) + "|")
    return "\n".join(out)


def build_arena(scene, obstacles, h, w):
    for ox, oy, ch in obstacles:
        hh = HEIGHTS[ch]
        scene.add_box(size=(CELL, CELL, hh),
                      pos=(ox * CELL, oy * CELL, hh / 2), fixed=True)
    # Bounding walls, one cell of clearance beyond the grid.
    wx0, wx1 = -1.5 * CELL, (w - 0.5) * CELL
    wy0, wy1 = -1.5 * CELL, (h - 0.5) * CELL
    cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
    span_x, span_y, t, wh = wx1 - wx0, wy1 - wy0, 0.1, WALL_H
    for size, pos in [((span_x, t, wh), (cx, wy0, wh / 2)),
                      ((span_x, t, wh), (cx, wy1, wh / 2)),
                      ((t, span_y, wh), (wx0, cy, wh / 2)),
                      ((t, span_y, wh), (wx1, cy, wh / 2))]:
        scene.add_box(size=size, pos=pos, fixed=True)


def main():
    # Ctrl+C should kill immediately: Genesis build/step are long C calls that
    # defer Python's KeyboardInterrupt, so use the OS default SIGINT (hard kill).
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="CPG locomotion checkpoint (default: stable 'walk')")
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--azimuth", type=int, default=360,
                   help="XT16 azimuth samples (multiple of 36; try 720 for a "
                        "denser cloud)")
    p.add_argument("--out", type=str, default="slam_cloud.png",
                   help="path for the 3D point-cloud scatter PNG")
    p.add_argument("--animate", type=str, default=None,
                   help="path for a GIF animating the cloud forming in real time")
    p.add_argument("--animate-every", type=int, default=200,
                   help="capture an animation frame every N steps")
    p.add_argument("--live", action="store_true", default=False,
                   help="show a separate live plot of the reconstruction while "
                        "the sim runs (keep --headless: Genesis draws no window)")
    p.add_argument("--live-every", type=int, default=40,
                   help="refresh the live plot every N steps")
    p.add_argument("--dashboard", type=int, default=0, metavar="PORT",
                   help="host the dashboard IN-process on this port (0 = off)")
    p.add_argument("--dashboard-url", type=str, default=None, metavar="URL",
                   help="push to a DECOUPLED dashboard server (e.g. "
                        "http://127.0.0.1:8080; start it with "
                        "'python -m domo.dashboard --port 8080')")
    p.add_argument("--dashboard-camera", action="store_true", default=False,
                   help="add the Genesis camera panel (renders a frame — small "
                        "sim-thread cost; off by default)")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    # Pick the matplotlib backend before any pyplot use: Agg for file output,
    # a GUI backend when a live window is requested.
    import matplotlib
    if not args.live:
        matplotlib.use("Agg")
    device = pick_device(args.device)
    dt = 0.02

    obstacles, start, h, w = parse_arena(ARENA)
    print("Ground-truth arena:\n" + render_truth(ARENA))

    # --- world ------------------------------------------------------------
    engine = create_engine("genesis", device=device)
    scene = engine.create_scene(SimConfig(
        dt=dt, device=device, headless=args.headless, solver_iterations=100,
        viewer=ViewerConfig(camera_pos=(w * 0.5, -4.0, h))))
    scene.add_ground()
    build_arena(scene, obstacles, h, w)
    sx, sy = start
    robot = Robot(GO2, scene, engine.device, kp=100.0, kd=2.0,
                  base_init_pos=(sx * CELL, sy * CELL, 0.35))

    lidar_model = hesai_xt16(n_horizontal=args.azimuth)
    lidar_handle = scene.add_lidar(robot.articulation,
                                   lidar_model.to_lidar_config())
    camera = None
    if args.dashboard and args.dashboard_camera:      # must precede build()
        camera = scene.add_camera(res=(360, 260), pos=(w * 0.5, -5.0, h * 1.1),
                                  lookat=(w / 2 - 0.5, h / 2 - 0.5, 0.3))
    scene.build(n_envs=1)
    robot.bind(n_envs=1)
    lidar = SimulatedLidar(lidar_handle, lidar_model, 1, n_sectors=36,
                           device=engine.device, control_dt=dt)

    # --- driving brain (patrol) + SLAM perception layer -------------------
    checkpoint = args.checkpoint or stable_policy("walk")
    policy_fn, _ = load_locomotion_policy(checkpoint, str(engine.device))
    library = make_go2_library(policy_fn)
    program = library.compile(f"({PATROL}) | stand.for(2)",
                              device=str(engine.device))
    controller = SingleSkillController(program)
    controller.setup(robot)

    # High-fidelity 3D mapping from the XT16's raw world points.
    slam = SlamSkill(lidar, SlamConfig(
        resolution=0.1, half_extent=8.0, origin=(w / 2 - 0.5, h / 2 - 0.5),
        z_band=(0.25, 1.3), map_max_range=12.0))
    slam.setup(robot)

    # --- run: drive, and let SLAM observe every tick ----------------------
    loop = SimControlLoop(scene, robot, controller, dt=dt, sensors=[lidar])
    state = loop.reset()
    slam.reset_idx(torch.arange(1))
    recorder = SlamRecorder(every=args.animate_every) if args.animate else None
    live = SlamLiveView(every=args.live_every) if args.live else None
    dash = None
    if args.dashboard_url or args.dashboard:     # LAZY: only now is web code loaded
        from domo.dashboard import Dashboard, DashboardClient
        dash = (DashboardClient(args.dashboard_url).start() if args.dashboard_url
                else Dashboard(port=args.dashboard,
                               title="DOMO SLAM — arena").start())
        robot_m, robot_root = go2_robot_manifest(GO2)   # push 3D scene once
        dash.set_scene({"boxes": arena_manifest_boxes(obstacles, h, w),
                        "robot": robot_m, "up": "z"},
                       roots={"robot": robot_root})
    print(f"Patrolling: {program.source}\n")

    def publish(i, status):
        if dash and (i % 10 == 0 or status != "running"):
            frame = camera.render() if camera is not None else None
            dash.publish(slam_snapshot(i, state, slam, lidar, status=status), frame)

    i, paused = 0, False
    while i < args.steps:
        if dash:                            # dashboard buttons → sim control
            c = dash.poll_command()
            if c == "stop":
                print("  [dashboard] stop"); break
            if c == "pause":
                paused = not paused
                print(f"  [dashboard] {'paused' if paused else 'resumed'}")
            if c == "reset":
                print("  [dashboard] reset")
                state = loop.reset(); slam.reset_idx(torch.arange(1)); i = 0
        if paused:
            publish(i, "paused"); time.sleep(0.05); continue

        state = loop.step()                 # drives robot + ticks the lidar
        slam.update_command(state, dt)      # fold the scan into the map (adds 0)
        if recorder:
            recorder.capture(slam, state)
        if live:
            live.update(slam, state)        # redraw the live plot as we go
        publish(i, "running")               # ~5 Hz, non-blocking
        if (i + 1) % 500 == 0:
            print(f"  step {i + 1:5d} | pos=("
                  f"{state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f}) | "
                  f"coverage={slam.coverage()[0].item() * 100:4.1f}% | "
                  f"odom_drift={slam.odometry_drift(state)[0].item():.2f} m")
        if program.finished:
            break
        i += 1

    # --- result -----------------------------------------------------------
    print("\nGround truth (# = obstacle):\n" + render_truth(ARENA))
    print("\nSLAM occupancy map (# occupied, . free, ' ' unknown):\n"
          + slam.render_ascii(step=2))
    print(f"\nCoverage: {slam.coverage()[0].item() * 100:.1f}% of the map is "
          f"confidently known.")
    print(f"Odometry drift over the run: "
          f"{slam.odometry_drift(state)[0].item():.2f} m "
          f"(where a real system would fuse scan-matching to correct it).")

    cloud = slam.point_cloud()
    print(f"\n3D point cloud: {cloud.shape[0]:,} accumulated world points.")
    save_point_cloud(cloud, args.out, title="SLAM point cloud — Hesai XT16")
    if recorder:
        recorder.save_gif(args.animate, title="SLAM cloud forming (top-down)")
    if live:
        live.keep_open()
    if dash:
        dash.publish(slam_snapshot(i, state, slam, lidar, status="stopped"))
        print(f"  [dashboard] still live at http://127.0.0.1:{args.dashboard} "
              f"— Ctrl+C to exit.")
        try:
            while True:
                if dash.poll_command() == "stop":
                    break
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass
        dash.stop()


if __name__ == "__main__":
    main()
