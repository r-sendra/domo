"""
Waypoint navigation on a trained CPG policy — library replica of
scripts/house_scene/evaluate_nav.py.

The P-controller lives in domo.control.navigation (engine-agnostic, driven
through injected step/pose callables); this script provides the sim wiring,
stdin intervention, and the demo modes. Natural-language commands are parsed
by a small rule-based parser, or by Gemini with `--gemini`.

Library pieces exercised: domo.control (PositionController, NavConfig),
domo.tasks (Go2CPGWalkTask as a single-env stepping harness),
domo.checkpoints.

    python examples/navigation/evaluate_nav.py CHECKPOINT --demo waypoints
    python examples/navigation/evaluate_nav.py CHECKPOINT --demo forward --headless
    python examples/navigation/evaluate_nav.py CHECKPOINT --demo interactive
    python examples/navigation/evaluate_nav.py CHECKPOINT --demo interactive --gemini
    # CPU: same commands (device is auto-picked: cuda if available, else cpu)

While a demo runs, type p/r/a/s/+/-/? + Enter on stdin to pause, resume,
abort the current goal, stop, change speed, or print status. Prints the
controller's progress and a "=== Done ===" footer; saves nothing.
`--gemini` needs `pip install google-genai` and GEMINI_API_KEY.
"""

import argparse
import contextlib
import json
import os
import select
import sys

import torch

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.control import NavConfig, PositionController
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask

SPEED_STEP = 0.2            # m/s per '+'/'-' keypress
GEMINI_MODEL = "gemini-2.5-flash"

# ==========================================================================
#  Stdin intervention (non-blocking, no background thread)
# ==========================================================================

class MissionControl:
    """p=pause r=resume a=abort s=stop +=faster -=slower ?=status"""

    def __init__(self):
        self._paused = False
        self._aborted = False
        self._stopped = False
        self._speed_delta = 0.0

    def start(self):
        print("  [ctrl] Intervention active — type commands + Enter:")
        print("         p=pause r=resume a=abort s=stop +=faster -=slower ?=status")

    def poll(self):
        """Consume one stdin line if present; never blocks the control loop."""
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return
            line = sys.stdin.readline().strip().lower()
        except (OSError, ValueError):
            return          # stdin closed / not selectable (e.g. piped or Windows)
        if line:
            self._apply(line)

    def _apply(self, line: str) -> None:
        if line in ("p", "pause"):
            self._paused = True
            print("  [ctrl] PAUSED — type 'r' to resume")
        elif line in ("r", "resume"):
            self._paused = False
            print("  [ctrl] RESUMED")
        elif line in ("a", "abort"):
            self._aborted = True
            self._paused = False
            print("  [ctrl] ABORTING current goal")
        elif line in ("s", "stop"):
            self._stopped = True
            self._aborted = True
            self._paused = False
            print("  [ctrl] STOPPING mission")
        elif line in ("+", "faster"):
            self._speed_delta += SPEED_STEP
            print(f"  [ctrl] Speed +{SPEED_STEP}")
        elif line in ("-", "slower"):
            self._speed_delta -= SPEED_STEP
            print(f"  [ctrl] Speed -{SPEED_STEP}")
        elif line in ("?", "status"):
            print(f"  [ctrl] paused={self._paused} aborted={self._aborted} "
                  f"stopped={self._stopped} "
                  f"speed_delta={self._speed_delta:+.1f}")

    @property
    def paused(self):
        return self._paused

    @property
    def aborted(self):
        v = self._aborted
        self._aborted = False           # read-and-clear: one abort per goal
        return v

    @property
    def stopped(self):
        return self._stopped

    def pop_speed_delta(self):
        d = self._speed_delta
        self._speed_delta = 0.0
        return d


# ==========================================================================
#  Command parsing
# ==========================================================================

def parse_simple(text: str) -> list:
    """Rule-based parser — no LLM needed for basic commands."""
    t = text.lower().strip()
    words = t.split()
    speed = None
    for i, w in enumerate(words):
        if w == "at" and i + 1 < len(words):
            with contextlib.suppress(ValueError):    # "at full" → no speed
                speed = float(words[i + 1])

    def nums():
        return [float(w) for w in words if _is_number(w)]

    try:
        if t.startswith(("forward", "go forward")):
            n = nums()
            return [{"type": "forward", "distance": n[0] if n else 1.0,
                     "speed": speed}]
        if t.startswith(("backward", "go backward")):
            n = nums()
            return [{"type": "backward", "distance": n[0] if n else 1.0,
                     "speed": speed}]
        if t.startswith("turn"):
            n = nums()
            angle = n[0] if n else 90.0
            if "right" in t:
                angle = -abs(angle)
            return [{"type": "turn", "angle_deg": angle}]
        if t.startswith("go to"):
            n = nums()
            if len(n) >= 2:
                return [{"type": "go_to", "x": n[0], "y": n[1], "speed": speed}]
        if t in ("stop", "halt"):
            return [{"type": "stop"}]
    except (IndexError, ValueError):
        pass
    return [{"type": "stop"}]


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


SYSTEM_PROMPT_NAV = """You are a navigation command interpreter for a quadruped robot.
Convert natural language into a JSON array of navigation commands.
Output ONLY the JSON array — no markdown, no explanation.

Command types:
  {"type":"forward",  "distance":float, "speed":float|null}
  {"type":"backward", "distance":float, "speed":float|null}
  {"type":"turn",     "angle_deg":float}          // positive=left, negative=right
  {"type":"go_to",    "x":float, "y":float, "speed":float|null}
  {"type":"stop"}

Speed is optional — omit or set null to use default.
Distances in metres. Angles in degrees.

Examples:
  "go 5 metres forward"             → [{"type":"forward","distance":5.0,"speed":null}]
  "turn right 90 degrees"           → [{"type":"turn","angle_deg":-90.0}]
  "go to 3 2 then turn left"        → [{"type":"go_to","x":3.0,"y":2.0,"speed":null},{"type":"turn","angle_deg":90.0}]
  "stop"                            → [{"type":"stop"}]"""


def parse_gemini(text: str) -> list:
    """LLM parser: one Gemini call per utterance (google-genai, GEMINI_API_KEY)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  pip install google-genai")
        return [{"type": "stop"}]
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("  Set GEMINI_API_KEY")
        return [{"type": "stop"}]
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=[
            types.Part(text=SYSTEM_PROMPT_NAV + f'\n\nUser said: "{text}"')])],
        config=types.GenerateContentConfig(temperature=0.1,
                                           max_output_tokens=300))
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def execute_commands(commands: list, controller: PositionController):
    for cmd in commands:
        if controller.ctrl.stopped:
            break
        t = cmd.get("type", "stop")
        speed = cmd.get("speed", None)
        if t == "forward":
            controller.go_forward(float(cmd.get("distance", 1.0)), speed=speed)
        elif t == "backward":
            controller.go_backward(float(cmd.get("distance", 1.0)), speed=speed)
        elif t == "turn":
            controller.turn(float(cmd.get("angle_deg", 90.0)))
        elif t == "go_to":
            controller.go_to(float(cmd.get("x", 0.0)), float(cmd.get("y", 0.0)),
                             speed=speed)
        elif t == "stop":
            controller.stop()


# ==========================================================================
#  Demos
# ==========================================================================

def run_demo_forward(controller, use_gemini=False):
    print("\n=== Demo: forward sequence ===")
    execute_commands([
        {"type": "forward", "distance": 5.0, "speed": None},
        {"type": "turn", "angle_deg": 90.0},
        {"type": "forward", "distance": 3.0, "speed": 0.5},
        {"type": "turn", "angle_deg": -90.0},
        {"type": "stop"},
    ], controller)
    print("=== Done ===")


def run_demo_waypoints(controller, use_gemini=False):
    print("\n=== Demo: square waypoints ===")
    for i, (x, y) in enumerate([(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]):
        if controller.ctrl.stopped:
            break
        print(f"\n  Waypoint {i + 1}/4: ({x},{y})")
        controller.go_to(x, y)
    controller.stop()
    print("=== Done ===")


def run_demo_interactive(controller, use_gemini=False):
    print("\n=== Interactive navigation ===")
    print("  forward N [at S] | backward N | turn N [left|right] | "
          "go to X Y | stop | quit")
    while not controller.ctrl.stopped:
        try:
            text = input("  nav> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() == "quit":
            break
        cmds = parse_gemini(text) if use_gemini else parse_simple(text)
        print(f"  → {cmds}")
        execute_commands(cmds, controller)
    controller.stop()
    print("=== Done ===")


DEMOS = {"forward": run_demo_forward,
         "waypoints": run_demo_waypoints,
         "interactive": run_demo_interactive}


# ==========================================================================
#  Wiring
# ==========================================================================

def build_env(device: str, headless: bool) -> Go2CPGWalkTask:
    """Single-env walk task as a stepping harness: no episode end, no resampling."""
    return Go2CPGWalkTask(Go2CPGWalkConfig(
        n_envs=1, device=device, headless=headless,
        max_episode_steps=1_000_000, resampling_time_s=1e9))


def make_step_and_pose(env, policy_fn):
    """The two callables the engine-agnostic PositionController is driven through."""
    obs_holder = {"obs": env.reset()[0]}

    def step_fn(cmd: torch.Tensor):
        env.commands[:] = cmd.to(env.device)
        with torch.no_grad():
            act = policy_fn(obs_holder["obs"])
        obs_holder["obs"], _, _, reset_buf, _ = env.step(act)
        if reset_buf[0]:                     # fell: fresh episode, keep navigating
            obs_holder["obs"], _ = env.reset()

    def pose_fn():
        s = env.robot.state
        return (float(s.base_pos[0, 0]), float(s.base_pos[0, 1]),
                float(s.base_euler[0, 2]))

    return step_fn, pose_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str)
    p.add_argument("--demo", type=str, default="waypoints",
                   choices=["forward", "waypoints", "interactive"])
    p.add_argument("--gemini", action="store_true")
    p.add_argument("--headless", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    # --- policy + sim harness (device auto-picked, as in the frozen script) --
    device = pick_device("cuda")
    policy_fn, _ = load_locomotion_policy(args.checkpoint, device)
    env = build_env(device, args.headless)
    step_fn, pose_fn = make_step_and_pose(env, policy_fn)

    # --- controller with stdin intervention ---------------------------------
    ctrl = MissionControl()
    ctrl.start()
    controller = PositionController(step_fn, pose_fn,
                                    NavConfig(dt=env.dt), intervention=ctrl)

    # --- run the chosen demo ------------------------------------------------
    DEMOS[args.demo](controller, use_gemini=args.gemini)


if __name__ == "__main__":
    main()
