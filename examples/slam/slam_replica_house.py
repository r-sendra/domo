"""
SLAM in a real ReplicaCAD apartment.

The Go2 wanders a Habitat/ReplicaCAD house — walking forward with reactive
lidar avoidance and turning when it stalls — while the passive SLAM layer
maps everything it sees. This is the same `SlamSkill` as slam_demo, now in a
cluttered, realistic environment instead of a hand-built arena: the occupancy
map recovers the room/wall layout, and the 3D point cloud reconstructs the
furniture. Because a blind wander can't reach every corner, the map has
honest gaps — the motivation for the frontier explorer.

The digital twin (goal-free World) builds the scene + robot + XT16 lidar; a
PlanningController authors the wander program each cycle (`avoid @ walk`, then
a turn); SLAM rides on top as a persistent observer, ticked every step.

    # stable walk+avoid policies by default; save a cloud PNG
    python examples/slam/slam_replica_house.py --headless --device cpu --out house.png
    # watch the house map build LIVE in a separate window (Genesis headless)
    python examples/slam/slam_replica_house.py --headless --device cpu --live

Assets: scripts/house_scene/data/replica_cad/ (scene apt_0 by default).
"""

import argparse
import random
import time

import torch
from slam_viz import SlamLiveView, SlamRecorder, go2_robot_manifest, save_point_cloud, slam_snapshot

from domo.checkpoints import load_checkpoint, load_locomotion_policy, pick_device
from domo.control import SlamConfig, SlamSkill
from domo.policies import stable_policy
from domo.rl import ActorCritic, clean_state_dict
from domo.robot.lidar_models import hesai_xt16
from domo.skills import PlanningController, make_go2_library
from domo.world import World, WorldConfig

# NOTE: domo.dashboard (all web server/client code) is imported lazily, only
# when --dashboard/--dashboard-url is set — no web code loads otherwise.

SCENE_JSON = ("scripts/house_scene/data/replica_cad/configs/scenes/"
              "apt_0.scene_instance.json")
ASSET_ROOT = "scripts/house_scene/data/replica_cad/"
SPAWN = (3.0, -3.0, 0.44)          # house spawn from the avoidance experiments
SPAWN_YAW_DEG = 180.0
GROUND_H = 0.2                     # ReplicaCAD floor height


def load_avoid_policy(path, device):
    ckpt = load_checkpoint(path, device)
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(device)
    return lambda obs: net.get_action(obs, deterministic=True)[0]


class HouseWanderer(PlanningController):
    """
    Authors an exploration wander as skill programs: walk forward under lidar
    avoidance for a few seconds, then turn a random way and repeat. On a stall
    (avoid's blocked / a tip) the leg fails and the next plan() turns away —
    so it backs out of dead-ends on its own. SLAM (held here, ticked every
    step) maps throughout. This is the researcher stand-in for the frontier
    planner that will read SLAM's map and choose *where* to look next.
    """

    def __init__(self, library, slam, vx=0.4, max_legs=80):
        super().__init__(library, decision_interval=5)
        self.slam = slam
        self.vx = vx
        self.max_legs = max_legs
        self.n_legs = 0

    def update(self, state, dt):
        self.slam.update_command(state, dt)          # persistent mapping
        return super().update(state, dt)

    def plan(self, state, last):
        self.n_legs += 1
        if self.n_legs > self.max_legs:
            return None                              # done → idle (stand)
        if self.n_legs % 2 == 1:
            return f"(avoid @ walk(vx={self.vx})).for(6)"
        return f"walk(vyaw={random.choice([-0.8, 0.8]):.1f}).for(2)"


def main():
    # Ctrl+C should kill immediately: Genesis build/step are long C calls that
    # defer Python's KeyboardInterrupt, so use the OS default SIGINT (hard kill).
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    p = argparse.ArgumentParser()
    p.add_argument("--cpg-checkpoint", default=None,
                   help="CPG walk checkpoint (default: stable 'walk')")
    p.add_argument("--avoid-checkpoint", default=None,
                   help="avoidance checkpoint (default: stable 'avoid')")
    p.add_argument("--scene-json", default=SCENE_JSON)
    p.add_argument("--asset-root", default=ASSET_ROOT)
    p.add_argument("--azimuth", type=int, default=360,
                   help="XT16 azimuth samples (multiple of 36)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="house_cloud.png")
    p.add_argument("--animate", default=None,
                   help="path for a GIF animating the cloud forming in real time")
    p.add_argument("--animate-every", type=int, default=250,
                   help="capture an animation frame every N steps")
    p.add_argument("--live", action="store_true", default=False,
                   help="show a separate live plot of the reconstruction while "
                        "the sim runs (keep --headless: Genesis draws no window)")
    p.add_argument("--live-every", type=int, default=50,
                   help="refresh the live plot every N steps")
    p.add_argument("--dashboard", type=int, default=0, metavar="PORT",
                   help="host the dashboard IN-process on this port (0 = off)")
    p.add_argument("--dashboard-url", type=str, default=None, metavar="URL",
                   help="push to a DECOUPLED dashboard server (start it with "
                        "'python -m domo.dashboard --port 8080')")
    p.add_argument("--dashboard-camera", action="store_true", default=False,
                   help="add the Genesis camera panel (renders a frame — small "
                        "sim-thread cost; off by default)")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Backend before any pyplot use: Agg for files, GUI when --live.
    import matplotlib
    if not args.live:
        matplotlib.use("Agg")
    device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dt = 0.02

    # --- the twin: goal-free World builds the house + robot + XT16 lidar ----
    cam_res = (360, 260) if (args.dashboard and args.dashboard_camera) else None
    world = World(WorldConfig(
        device=device, dt=dt, headless=args.headless,
        scene_kind="replica", ground_height=GROUND_H,
        replica_scene_json=args.scene_json, replica_asset_root=args.asset_root,
        base_init_pos=SPAWN, base_init_yaw_deg=SPAWN_YAW_DEG,
        lidar_model=hesai_xt16(n_horizontal=args.azimuth),
        camera_res=cam_res, camera_pos=(SPAWN[0] + 5, SPAWN[1] - 5, 4.0),
        camera_lookat=(SPAWN[0], SPAWN[1], 0.5)), n_envs=1)

    # --- skills: walk + avoid from the stable registry, layered library ------
    cpg_ckpt = args.cpg_checkpoint or stable_policy("walk")
    avoid_ckpt = args.avoid_checkpoint or stable_policy("avoid")
    walk_policy, _ = load_locomotion_policy(cpg_ckpt, str(device))
    avoid_policy = load_avoid_policy(avoid_ckpt, str(device))
    # Cap avoid's forward/back authority so it STEERS around furniture but can
    # never cancel the walk's forward progress (the conservative policy would
    # otherwise freeze the robot in a cluttered house).
    library = make_go2_library(walk_policy, avoid_policy, world.lidar,
                               avoid_deltas=(0.2, 0.5, 1.2))

    # --- SLAM perception layer over the house --------------------------------
    slam = SlamSkill(world.lidar, SlamConfig(
        resolution=0.1, half_extent=10.0, origin=(SPAWN[0], SPAWN[1]),
        z_band=(GROUND_H + 0.2, GROUND_H + 1.4), map_max_range=10.0))
    controller = HouseWanderer(library, slam, vx=0.4)
    controller.setup(world.robot)
    slam.setup(world.robot)

    # --- run: wander + map ---------------------------------------------------
    loop = world.make_loop(controller)
    state = loop.reset()
    slam.reset_idx(torch.arange(1))
    recorder = SlamRecorder(every=args.animate_every) if args.animate else None
    live = SlamLiveView(every=args.live_every) if args.live else None
    dash = None
    if args.dashboard_url or args.dashboard:     # LAZY: only now is web code loaded
        from domo.dashboard import Dashboard, DashboardClient
        from domo.robot import GO2
        dash = (DashboardClient(args.dashboard_url).start() if args.dashboard_url
                else Dashboard(port=args.dashboard,
                               title="DOMO SLAM — house").start())
        robot_m, robot_root = go2_robot_manifest(GO2)   # robot renders; meshes=Phase B
        dash.set_scene({"boxes": [], "robot": robot_m, "up": "z"},
                       roots={"robot": robot_root})
    print(f"Wandering the ReplicaCAD house (spawn {SPAWN[:2]}) ...")

    def publish(i, status):
        if dash and (i % 10 == 0 or status != "running"):
            frame = world.camera.render() if world.camera is not None else None
            dash.publish(slam_snapshot(i, state, slam, world.lidar, status=status),
                         frame)

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
                state = loop.reset(); slam.reset_idx(torch.arange(1))
                controller.n_legs = 0; controller.program = None
                controller.active = controller.IDLE; i = 0
        if paused:
            publish(i, "paused"); time.sleep(0.05); continue

        state = loop.step()
        if recorder:
            recorder.capture(slam, state)
        if live:
            live.update(slam, state)
        publish(i, "running")
        if (i + 1) % 500 == 0:
            print(f"  step {i + 1:5d} | pos=("
                  f"{state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f}) | "
                  f"leg {controller.n_legs}/{controller.max_legs} | "
                  f"coverage={slam.coverage()[0].item() * 100:4.1f}%")
        if controller.idle and controller.n_legs > controller.max_legs:
            break
        i += 1

    # --- result --------------------------------------------------------------
    print("\nSLAM occupancy map (# occupied, . free, ' ' unknown):\n"
          + slam.render_ascii(step=3))
    print(f"\nCoverage: {slam.coverage()[0].item() * 100:.1f}% of the map is "
          f"confidently known (blind wander → honest gaps).")
    cloud = slam.point_cloud()
    print(f"3D point cloud: {cloud.shape[0]:,} accumulated world points.")
    save_point_cloud(cloud, args.out, title="SLAM point cloud — ReplicaCAD house")
    if recorder:
        recorder.save_gif(args.animate, title="SLAM house map forming (top-down)")
    if live:
        live.keep_open()
    if dash:
        dash.publish(slam_snapshot(i, state, slam, world.lidar, status="stopped"))
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
