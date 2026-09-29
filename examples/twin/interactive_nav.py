"""
Click a point — on the dashboard map or in the viewer — the robot walks there.

The twin stands still until someone clicks. The dashboard's top-down panel
turns a click into a world (x, y) and posts it on the command channel as
`goto:<x>,<y>`; with `--click-viewer` the Genesis window is a second click
surface, where a left-click on the floor is raycast to the same world (x, y).
Either way the loop hands that goal to a `PlanningController`, which AUTHORS
the program that walks there — `avoid @ goto(x, y) @ walk` — reports whether
it arrived, and goes back to idling. The point of the example is that seat:
the goal arrives as data, the *program* is written inside `plan()`, which is
exactly where the M1 supervisor will sit.

Library pieces exercised: domo.world (goal-free World, scene_kind="arena",
36-sector lidar), domo.policies (stable 'walk' + 'avoid' through
stable_go2_library), domo.skills (PlanningController, make_go2_library, the
'goto' nav card), domo.sim (`Scene.on_ground_click` — viewer picking behind
the engine-agnostic contract, so the click path never touches genesis),
domo.dashboard (the command channel; imported LAZILY, only when
--dashboard/--dashboard-url is given). No frozen-script counterpart.

    # decoupled (recommended): the server in one shell ...
    python -m domo.dashboard --port 8080
    # ... the twin in another, then click the point-cloud panel
    python examples/twin/interactive_nav.py --headless --device cpu \\
        --dashboard-url http://127.0.0.1:8080

    # in-process server instead; open http://127.0.0.1:8080
    python examples/twin/interactive_nav.py --headless --device cpu --dashboard 8080
    # no web at all: click the floor in the Genesis window (needs a display)
    python examples/twin/interactive_nav.py --device cpu --click-viewer
    # plain 'goto(...) @ walk', without the conservative avoid layer
    python examples/twin/interactive_nav.py --headless --device cpu \\
        --dashboard 8080 --no-avoid

Prints every goal it accepts, every arrival or abandonment, a position line
every 250 steps, and the goal history at the end. Saves nothing. Without a
dashboard flag and without --click-viewer there is no way to send a goal, so
the robot just stands there — and no web code is imported at all.
"""

import argparse
import contextlib
import os
import signal
import threading
import time
from collections import deque

import torch

from domo.checkpoints import load_checkpoint, load_locomotion_policy, pick_device
from domo.policies import load_stable_avoid, load_stable_locomotion, stable_go2_library
from domo.rl import ActorCritic, clean_state_dict
from domo.scenes import ObstacleArenaConfig
from domo.skills import PlanningController, PlanOutcome, make_go2_library
from domo.world import World, WorldConfig

# NOTE: domo.dashboard (all web server/client code, and `parse_goto`) is
# imported lazily, only when --dashboard/--dashboard-url is set.

CONTROL_DT = 0.02
# Sparser than the training arena, and with NO steps or balls: a 4 cm step is
# invisible to the 0.35 m lidar and a loose ball rolls underfoot, so both trip
# the robot — and a fall ends the session (nothing in the library gets up).
ARENA = ObstacleArenaConfig(n_chairs=3, n_sofas=2, n_pillars=3,
                            n_steps=0, n_balls=0)
MAP_MARGIN = 0.5                   # top-down panel extent beyond the walls (m)
# [x_min, x_max, y_min, y_max] of the clickable panel: the whole walled arena,
# so every click lands on a point inside it.
BOUNDS = [-(ARENA.half_size + MAP_MARGIN), ARENA.half_size + MAP_MARGIN,
          -(ARENA.half_size + MAP_MARGIN), ARENA.half_size + MAP_MARGIN]
# Cap avoid's forward/back authority so it STEERS around obstacles but can
# never cancel the nav command: the bundled avoid net is conservative enough
# to deadlock against goto's min_speed floor otherwise. (Δvx, Δvy, Δvyaw)
AVOID_DELTAS = (0.25, 0.5, 1.2)

FALLEN_H = 0.18                    # base height below which walk calls it a fall
CLOUD_MAX = 1500                   # rolling top-down cloud size (points)
CLOUD_STRIDE = 3                   # keep every Nth hit of a scan
CLOUD_Z_MIN = 0.1                  # drop ground returns (m)
REPORT_EVERY = 250                 # progress line cadence (steps)
IDLE_STEPS_NO_DASH = 250           # steps to stand for when no dashboard is wired
PUBLISH_EVERY = 10                 # dashboard snapshot cadence (~5 Hz at 50 Hz)
PAUSE_IDLE_S = 0.05                # sleep while the dashboard has us paused
HOLD_POLL_S = 0.3                  # command poll cadence after the run
CAMERA_RES = (360, 260)            # offscreen dashboard camera


def load_avoid_policy(path, device):
    """Callable obs → raw Δv from a trained avoidance checkpoint."""
    ckpt = load_checkpoint(path, device)
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(device)

    def avoid_policy(obs):
        return net.get_action(obs, deterministic=True)[0]
    return avoid_policy


# ---------------------------------------------------------------------------
# The brain: a planner whose only input is "someone clicked here"
# ---------------------------------------------------------------------------

class ClickToGoController(PlanningController):
    """
    Idles until a goal is posted, then writes the program that goes there.

    `set_goal` / `cancel` are the whole external interface — the loop calls
    them with whatever the dashboard sent. Everything else is authorship:
    `plan()` composes the goal into grammar text, and a new goal arriving
    mid-walk preempts the running program (clicking again re-targets).
    """

    def __init__(self, library, use_avoid: bool = True,
                 decision_interval: int = 10):
        super().__init__(library, decision_interval=decision_interval)
        self.use_avoid = use_avoid
        self.goal: tuple[float, float] | None = None          # pending
        self.active_goal: tuple[float, float] | None = None   # being walked to
        self.arrivals: list[tuple[tuple[float, float], bool]] = []

    # --- what the dashboard can ask for ---------------------------------

    def set_goal(self, x: float, y: float) -> None:
        """Queue a goal; picked up at the next decision tick."""
        self.goal = (float(x), float(y))

    def cancel(self) -> None:
        """Forget the pending goal and stop walking to the current one."""
        self.goal = None
        self._abandon("goal cleared")

    @property
    def goal_xy(self) -> tuple[float, float] | None:
        """The goal the sim has accepted (what the page should draw)."""
        return self.active_goal or self.goal

    @property
    def status_text(self) -> str:
        """Telemetry `status`: what the robot is doing, in words."""
        if self.active_goal is None:
            return "idle"
        return "walking to ({:.2f}, {:.2f})".format(*self.active_goal)

    # --- authorship -----------------------------------------------------

    def plan(self, state, last):
        """A pending goal becomes a program; nothing pending → idle (stand)."""
        if last is not None and self.active_goal is not None:
            self.arrivals.append((self.active_goal, last.succeeded))
            gx, gy = self.active_goal
            print(f"  [nav] ({gx:+.2f}, {gy:+.2f}) → "
                  f"{'ARRIVED' if last.succeeded else 'FAILED'} "
                  f"('{last.program}') — idling")
            if not last.succeeded and float(state.base_pos[0, 2]) < FALLEN_H:
                # The same M1 skill-gap moment twin_demo reaches: nothing in
                # the library can get the robot back on its feet, so every
                # further program fails instantly on walk's fallen() failsafe.
                print("  [nav] the robot is DOWN and the library has no get-up "
                      "skill — press ⟳ reset to teleport back to spawn.")
            self.active_goal = None
        if self.goal is None:
            return None
        gx, gy = self.goal
        self.goal = None                       # consumed: one program per goal
        self.active_goal = (gx, gy)
        stack = f"goto(x={gx}, y={gy}) @ walk"
        return f"avoid @ {stack}" if self.use_avoid else stack

    def decide(self, state) -> None:
        """Let a fresh goal interrupt the walk in progress, then plan as usual."""
        if self.goal is not None and self.program is not None:
            self._abandon("re-targeted")
        super().decide(state)

    def _abandon(self, why: str) -> None:
        """Drop the running program and idle; recorded as a failed outcome."""
        if self.program is None:
            return
        print(f"  [nav] {why}: dropping '{self.program.source}'")
        self.history.append(PlanOutcome(self.program.source, False))
        if self.active_goal is not None:
            self.arrivals.append((self.active_goal, False))
        self.program = None
        self.active_goal = None
        self.activate(self.IDLE)


class GoalSlot:
    """
    One-slot mailbox for goals clicked in the viewer window.

    `Scene.on_ground_click` fires on the viewer's UI thread, mid-frame, so the
    callback must not reach into the controller (its `plan()` runs on the sim
    thread). It only drops the newest (x, y) here; the control loop `take()`s
    it and calls `set_goal` itself, exactly as it does for a dashboard goal.
    A second click before the loop looks simply overwrites the first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._goal: tuple[float, float] | None = None

    def set(self, x: float, y: float) -> None:
        """The viewer callback: cheap, thread-safe, never blocks the UI."""
        with self._lock:
            self._goal = (float(x), float(y))

    def take(self) -> tuple[float, float] | None:
        """Pop the pending goal (None if nobody clicked since the last call)."""
        with self._lock:
            goal, self._goal = self._goal, None
            return goal


# ---------------------------------------------------------------------------
# World / brain / dashboard
# ---------------------------------------------------------------------------

def build_world(args, device) -> World:
    """The twin: goal-free walled arena + Go2 + the 36-sector lidar avoid needs."""
    want_camera = args.dashboard_camera and (args.dashboard or args.dashboard_url)
    cam = CAMERA_RES if want_camera else None
    world = World(WorldConfig(
        device=device, dt=CONTROL_DT, headless=args.headless,
        scene_kind="arena", arena=ARENA, camera_res=cam,
        camera_pos=(ARENA.half_size + 2.0, -(ARENA.half_size + 2.0), 4.0),
        camera_lookat=(0.0, 0.0, 0.3)), n_envs=1)
    world.randomise_obstacles()        # obstacles are parked until scattered
    return world


def build_library(args, world):
    """walk (+ capped avoid) from the stable registry unless paths were given."""
    device = str(world.device)
    lidar = None if args.no_avoid else world.lidar     # no lidar → no avoid card
    if args.walk is None and args.avoid is None:
        return stable_go2_library(lidar, device, avoid_deltas=AVOID_DELTAS)
    walk_fn, _ = (load_locomotion_policy(args.walk, device) if args.walk
                  else load_stable_locomotion(device))
    avoid_fn = None
    if lidar is not None:
        avoid_fn = (load_avoid_policy(args.avoid, device) if args.avoid
                    else load_stable_avoid(device))
    return make_go2_library(walk_fn, avoid_fn, lidar, avoid_deltas=AVOID_DELTAS)


def build_brain(args, world) -> ClickToGoController:
    """The library the planner writes against, plus the planner itself."""
    controller = ClickToGoController(build_library(args, world),
                                     use_avoid=not args.no_avoid)
    controller.setup(world.robot)
    return controller


def robot_manifest(spec):
    """
    Scene-manifest entry + asset root so the 3D panel renders the robot.
    (Mirrors `examples/slam/slam_viz.go2_robot_manifest`; examples cannot
    import each other across folders.)
    """
    urdf = spec.urdf_path
    if not os.path.isabs(urdf):
        import genesis  # heavy: only to locate its asset dir
        urdf = os.path.join(os.path.dirname(genesis.__file__), "assets", urdf)
    urdf = os.path.abspath(urdf)
    root = os.path.dirname(os.path.dirname(urdf))     # <robot>/ (urdf/ + meshes/)
    rel = os.path.relpath(urdf, root).replace(os.sep, "/")
    return {"urdf": "/assets/robot/" + rel,
            "dof_names": list(spec.joint_names)}, root


def wall_boxes():
    """The arena's four walls for the 3D viewer (mirrors ObstacleArena)."""
    s = ARENA.half_size
    t, h = ARENA.wall_thickness, ARENA.wall_height
    return [{"size": [sx, sy, h], "pos": [wx, wy, h / 2], "color": "#484f58"}
            for wx, wy, sx, sy in [(0.0, s, 2 * s + 2 * t, t),
                                   (0.0, -s, 2 * s + 2 * t, t),
                                   (s, 0.0, t, 2 * s),
                                   (-s, 0.0, t, 2 * s)]]


def attach_dashboard(args):
    """In-process or decoupled dashboard (or None); pushes the 3D scene once."""
    if not (args.dashboard_url or args.dashboard):
        return None
    from domo.dashboard import Dashboard, DashboardClient  # LAZY: web code only now
    from domo.robot import GO2
    dash = (DashboardClient(args.dashboard_url).start() if args.dashboard_url
            else Dashboard(port=args.dashboard,
                           title="DOMO twin — click to walk").start())
    robot_m, robot_root = robot_manifest(GO2)
    dash.set_scene({"boxes": wall_boxes(), "robot": robot_m, "up": "z"},
                   roots={"robot": robot_root})
    return dash


def hold_dashboard(dash, args) -> None:
    """Keep serving the final state until Ctrl+C or the page's stop button."""
    where = args.dashboard_url or f"http://127.0.0.1:{args.dashboard}"
    print(f"  [dashboard] still live at {where} — Ctrl+C to exit.")
    try:
        while dash.poll_command() != "stop":
            time.sleep(HOLD_POLL_S)
    except KeyboardInterrupt:
        pass
    dash.stop()


# ---------------------------------------------------------------------------
# Telemetry: what the clickable panel draws
# ---------------------------------------------------------------------------

def collect_hits(cloud: deque, lidar) -> None:
    """Fold this scan's obstacle hits (world frame) into the rolling cloud."""
    if lidar is None:
        return
    # Before the first scan a backend may have no buffer to read; the exact
    # exception is backend-specific, so any failure just skips this scan.
    with contextlib.suppress(Exception):
        pts, valid = lidar.read_points()
        hits = pts[0][valid[0]]
        hits = hits[hits[:, 2] > CLOUD_Z_MIN][::CLOUD_STRIDE]
        cloud.extend([round(float(v), 2) for v in p] for p in hits)


def snapshot(step, state, controller, lidar, cloud, status):
    """A JSON-serialisable telemetry dict for the dashboard page."""
    snap = {
        "t": int(step),
        "status": status,
        "pose": [round(float(state.base_pos[0, 0]), 2),
                 round(float(state.base_pos[0, 1]), 2),
                 round(float(state.base_euler[0, 2]), 2)],
        # Full base pose (x,y,z, qw,qx,qy,qz) + joint angles for the 3D viewer.
        "base": [round(float(v), 3) for v in state.base_pos[0].tolist()]
                + [round(float(v), 4) for v in state.base_quat[0].tolist()],
        "dof": [round(float(v), 4) for v in state.dof_pos[0].tolist()],
        "vel": [round(float(state.base_lin_vel[0, 0]), 2),
                round(float(state.base_lin_vel[0, 1]), 2),
                round(float(state.base_ang_vel[0, 2]), 2)],
        "height": round(float(state.base_pos[0, 2]), 3),
        "cloud": list(cloud),
        "bounds": BOUNDS,              # the panel's extent = the click frame
    }
    if lidar is not None:
        with contextlib.suppress(Exception):
            snap["lidar"] = [round(float(v), 2) for v in lidar.read()[0].tolist()]
    goal = controller.goal_xy
    if goal is not None:
        snap["goal"] = [round(goal[0], 2), round(goal[1], 2)]
    return snap


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args, world, loop, controller, dash, viewer_goal=None):
    """
    Idle, walk where someone clicked, report. Dashboard commands: `goto:x,y`
    sets a goal (a new one re-targets), `clear` cancels it, `pause` toggles
    (the sim freezes but keeps publishing), `reset` teleports back to spawn,
    `stop` ends the run. `viewer_goal` is the mailbox the viewer's ground
    clicks land in — both surfaces feed the same `set_goal`.
    """
    parse_goto = None
    if dash is not None:
        from domo.dashboard import parse_goto  # LAZY: only with a dashboard
    state = loop.reset()
    cloud: deque = deque(maxlen=CLOUD_MAX)

    def publish(i, forced=None):
        if dash is None:
            return
        if i % PUBLISH_EVERY == 0 or forced:
            frame = world.camera.render() if world.camera is not None else None
            dash.publish(snapshot(i, state, controller, world.lidar, cloud,
                                  forced or controller.status_text), frame)

    i, paused = 0, False
    while i < args.steps:
        c = dash.poll_command() if dash is not None else None
        goal = parse_goto(c) if parse_goto is not None else None
        if goal is not None:
            print(f"  [dashboard] goal ({goal[0]:+.2f}, {goal[1]:+.2f})")
            controller.set_goal(*goal)
        elif c == "clear":
            controller.cancel()
        elif c == "stop":
            print("  [dashboard] stop")
            break
        elif c == "pause":
            paused = not paused
            print(f"  [dashboard] {'paused' if paused else 'resumed'}")
        elif c == "reset":
            print("  [dashboard] reset")
            controller.cancel()
            state = loop.reset()
            cloud.clear()
            i = 0
        if viewer_goal is not None:
            clicked = viewer_goal.take()
            if clicked is not None:
                print(f"  [viewer] goal ({clicked[0]:+.2f}, {clicked[1]:+.2f})")
                controller.set_goal(*clicked)
        if paused:
            publish(i, "paused")
            time.sleep(PAUSE_IDLE_S)
            continue

        state = loop.step()
        collect_hits(cloud, world.lidar)
        publish(i)
        if (i + 1) % REPORT_EVERY == 0:
            print(f"  step {i + 1:5d} | pos=("
                  f"{state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f}) | "
                  f"{controller.status_text}")
        i += 1
    publish(i, "stopped")          # final snapshot: the page keeps showing it


def stand_still(args, loop, controller) -> None:
    """No click surface at all: hold the stand pose briefly and say so."""
    print("  no click surface — nothing can send a goal, so the robot idles. "
          "Re-run with --dashboard PORT (or --dashboard-url URL) and click the "
          "point-cloud panel, or with --click-viewer and click the floor in "
          "the Genesis window.")
    steps = min(args.steps, IDLE_STEPS_NO_DASH)
    loop.reset()
    for _ in range(steps):
        loop.step()
    print(f"  idled {steps} steps in '{controller.status_text}'.")


def report(controller) -> None:
    if not controller.arrivals:
        print("\n  no goals were clicked.")
        return
    print(f"\n  goals ({len(controller.arrivals)}):")
    for (gx, gy), ok in controller.arrivals:
        print(f"    {'✓' if ok else '✗'} ({gx:+.2f}, {gy:+.2f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--walk", type=str, default=None,
                   help="CPG walk checkpoint (default: stable 'walk')")
    p.add_argument("--avoid", type=str, default=None,
                   help="avoidance checkpoint (default: stable 'avoid')")
    p.add_argument("--no-avoid", action="store_true", default=False,
                   help="compose plain 'goto(...) @ walk' — no avoid layer "
                        "(see the deadlock note in the docs)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--dashboard", type=int, default=0, metavar="PORT",
                   help="host the dashboard IN-process on this port (0 = off)")
    p.add_argument("--dashboard-url", type=str, default=None, metavar="URL",
                   help="push to a DECOUPLED dashboard server (start it with "
                        "'python -m domo.dashboard --port 8080')")
    p.add_argument("--dashboard-camera", action="store_true", default=False,
                   help="add the Genesis camera panel (renders a frame — small "
                        "sim-thread cost; off by default)")
    p.add_argument("--click-viewer", action="store_true", default=False,
                   help="also take goals from left-clicks on the floor of the "
                        "Genesis viewer window (needs a display; combinable "
                        "with the dashboard flags)")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    if args.click_viewer and args.headless:
        p.error("--click-viewer needs the interactive viewer, so it cannot be "
                "combined with --headless: drop one of the two flags.")
    return args


def main():
    # Ctrl+C should kill immediately: Genesis build/step are long C calls that
    # defer Python's KeyboardInterrupt, so use the OS default SIGINT (hard kill).
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(0)               # reproducible obstacle scatter

    # --- the twin: goal-free arena + robot + lidar --------------------------
    world = build_world(args, device)

    # --- the planner that will be handed clicks ----------------------------
    controller = build_brain(args, world)
    loop = world.make_loop(controller)

    # --- the click channels -------------------------------------------------
    dash = attach_dashboard(args)
    viewer_goal = None
    if args.click_viewer:
        # Engine-agnostic viewer picking: the backend raycasts the click onto
        # the arena floor and calls us on ITS UI thread, so we only store.
        viewer_goal = GoalSlot()
        world.scene.on_ground_click(viewer_goal.set,
                                    ground_z=world.cfg.ground_height)
    if dash is None and viewer_goal is None:
        stand_still(args, loop, controller)
        return
    program = f"{'avoid @ ' if not args.no_avoid else ''}goto(x, y) @ walk"
    if dash is not None:
        print("Click the point-cloud panel to send the robot there "
              f"(programs: '{program}').")
    if viewer_goal is not None:
        print("Left-click the floor in the Genesis viewer window to send the "
              f"robot there (programs: '{program}').")
    print()

    # --- run: idle, walk where the clicks point ----------------------------
    run(args, world, loop, controller, dash, viewer_goal)

    # --- result -------------------------------------------------------------
    report(controller)
    if dash is not None:
        hold_dashboard(dash, args)


if __name__ == "__main__":
    main()
