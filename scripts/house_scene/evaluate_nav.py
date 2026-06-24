"""
evaluate_nav.py
===============
Position-controlled navigation demo using the trained Go2 CPG-RL policy.

Requires go2_cpg_rl.py in the same directory.

NOTE: For turning to work properly the policy MUST be trained with:
    "ang_vel_range": [-1.0, 1.0]
If your checkpoint was trained with [0.0, 0.0] (no turning), retrain first.

Usage:
    # Hardcoded forward sequence
    python evaluate_nav.py --checkpoint ../../runs/go2_cpg/checkpoint_final.pt --demo forward

    # Navigate waypoints
    python evaluate_nav.py --checkpoint ../../runs/go2_cpg/checkpoint_final.pt --demo waypoints

    # Type commands (no LLM)
    python evaluate_nav.py --checkpoint ../../runs/go2_cpg/checkpoint_final.pt --demo interactive

    # Type natural language commands (Gemini)
    python evaluate_nav.py --checkpoint ../../runs/go2_cpg/checkpoint_final.pt --demo interactive-llm

    # Voice commands (Whisper + Gemini)
    python evaluate_nav.py --checkpoint ../../runs/go2_cpg/checkpoint_final.pt --demo voice
"""

import os
import sys
import math
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_cpg_rl import Go2CPGEnv, ActorCritic, clean_state_dict, OBS_DIM

import genesis as gs
from genesis.utils.geom import quat_to_xyz


# ==========================================================================
#  Intervention system
# ==========================================================================

class MissionControl:
    """
    Non-blocking intervention system.
    Checks stdin for commands each simulation step using select()
    so no background thread is needed — avoids macOS Tcl/Tk conflicts.

    Commands (type and press Enter):
        p / pause    — pause current goal
        r / resume   — resume after pause
        a / abort    — abort current goal
        s / stop     — stop everything
        + / faster   — increase speed by 0.2 m/s
        - / slower   — decrease speed by 0.2 m/s
        ? / status   — print current state
    """

    def __init__(self):
        self._paused      = False
        self._aborted     = False
        self._stopped     = False
        self._speed_delta = 0.0

    def start(self):
        print("  [ctrl] Intervention active — type commands + Enter:")
        print("         p=pause  r=resume  a=abort  s=stop  "
              "+=faster  -=slower  ?=status")

    def stop(self):
        pass   # nothing to clean up

    def poll(self):
        """
        Call this each simulation step to check for pending stdin input.
        Non-blocking — returns immediately if nothing is typed.
        """
        import select
        try:
            # Check if stdin has data available without blocking
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                return
            line = sys.stdin.readline().strip().lower()
            if not line:
                return
            if line in ("p", "pause"):
                self._paused  = True
                print("  [ctrl] PAUSED — type 'r' to resume")
            elif line in ("r", "resume"):
                self._paused  = False
                print("  [ctrl] RESUMED")
            elif line in ("a", "abort"):
                self._aborted = True
                self._paused  = False
                print("  [ctrl] ABORTING current goal")
            elif line in ("s", "stop"):
                self._stopped = True
                self._aborted = True
                self._paused  = False
                print("  [ctrl] STOPPING mission")
            elif line in ("+", "faster"):
                self._speed_delta += 0.2
                print(f"  [ctrl] Speed +0.2")
            elif line in ("-", "slower"):
                self._speed_delta -= 0.2
                print(f"  [ctrl] Speed -0.2")
            elif line in ("?", "status"):
                print(f"  [ctrl] paused={self._paused}  "
                      f"aborted={self._aborted}  stopped={self._stopped}  "
                      f"speed_delta={self._speed_delta:+.1f}")
        except Exception:
            pass

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def aborted(self) -> bool:
        v = self._aborted
        self._aborted = False
        return v

    @property
    def stopped(self) -> bool:
        return self._stopped

    def pop_speed_delta(self) -> float:
        d = self._speed_delta
        self._speed_delta = 0.0
        return d


# ==========================================================================
#  Pose sources
# ==========================================================================

class SimPoseSource:
    def __init__(self, env: Go2CPGEnv):
        self.env = env

    def get(self):
        pos  = self.env.base_pos[0].cpu().numpy()
        quat = self.env.base_quat[0].cpu().numpy()
        yaw  = float(quat_to_xyz(torch.tensor(quat).unsqueeze(0))[0, 2].item())
        return float(pos[0]), float(pos[1]), yaw


class RealRobotPoseSource:
    """Swap in when deploying on real Go2."""
    def __init__(self, sport_client):
        self.client = sport_client

    def get(self):
        state = self.client.GetState()
        return (
            float(state.position[0]),
            float(state.position[1]),
            float(state.imu_state.rpy[2]),
        )


# ==========================================================================
#  Position controller
# ==========================================================================

class PositionController:

    DEFAULT_VX   = 0.6    # m/s
    DEFAULT_VYAW = 0.8    # rad/s

    def __init__(
        self,
        env:          Go2CPGEnv,
        net:          ActorCritic,
        pose_source,
        ctrl:         MissionControl,
        device:       str   = "cpu",
        max_vx:       float = 0.8,
        max_vyaw:     float = 0.8,
        Kp_lin:       float = 0.8,
        Kp_ang:       float = 1.5,
        tol_pos:      float = 0.15,
        tol_ang:      float = 0.05,
    ):
        self.env         = env
        self.net         = net
        self.pose_source = pose_source
        self.ctrl        = ctrl
        self.device      = device
        self.max_vx      = max_vx
        self.max_vyaw    = max_vyaw
        self.Kp_lin      = Kp_lin
        self.Kp_ang      = Kp_ang
        self.tol_pos     = tol_pos
        self.tol_ang     = tol_ang
        self._obs        = None
        self._cmd        = torch.zeros(1, 3, device=device)

    def reset(self):
        obs, _ = self.env.reset()
        self._obs = obs
        self._cmd[:] = 0.0

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def go_forward(self, distance: float, speed: float = None) -> str:
        """
        Go straight forward by `distance` metres at optional `speed` m/s.
        Returns: 'done' | 'aborted' | 'stopped'
        """
        x, y, yaw = self.pose_source.get()
        tx = x + distance * math.cos(yaw)
        ty = y + distance * math.sin(yaw)
        spd = speed or self.max_vx
        print(f"  → go_forward({distance:.2f}m @ {spd:.1f}m/s)  "
              f"target=({tx:.2f},{ty:.2f})")
        return self._drive_to(tx, ty, override_vx=spd)

    def go_backward(self, distance: float, speed: float = None) -> str:
        x, y, yaw = self.pose_source.get()
        tx = x - distance * math.cos(yaw)
        ty = y - distance * math.sin(yaw)
        spd = speed or self.max_vx
        print(f"  → go_backward({distance:.2f}m @ {spd:.1f}m/s)")
        return self._drive_to(tx, ty, reverse=True, override_vx=spd)

    def turn(self, angle_deg: float) -> str:
        """
        Turn by `angle_deg` degrees in place.
        Positive = left (CCW), negative = right (CW).
        Returns: 'done' | 'aborted' | 'stopped'
        """
        _, _, yaw = self.pose_source.get()
        target_yaw = self._wrap(yaw + math.radians(angle_deg))
        print(f"  → turn({angle_deg:+.1f}°)  "
              f"target={math.degrees(target_yaw):.1f}°")
        return self._rotate_to(target_yaw)

    def go_to(self, x: float, y: float,
              speed: float = None,
              final_yaw_deg: float = None) -> str:
        print(f"  → go_to({x:.2f},{y:.2f})")
        result = self._drive_to(x, y, override_vx=speed)
        if result == "done" and final_yaw_deg is not None:
            result = self._rotate_to(math.radians(final_yaw_deg))
        return result

    def stop(self):
        self._cmd[:] = 0.0
        for _ in range(20):
            self._step()
        print("  → stop")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drive_to(
        self,
        tx: float, ty: float,
        reverse:     bool  = False,
        override_vx: float = None,
        timeout_s:   float = 60.0,
    ) -> str:
        max_steps = int(timeout_s / self.env.dt)
        vx_limit  = override_vx or self.max_vx

        for step in range(max_steps):
            # Intervention checks
            if self.ctrl.stopped:
                self.stop()
                return "stopped"
            if self.ctrl.aborted:
                self.stop()
                return "aborted"
            while self.ctrl.paused:
                self._cmd[:] = 0.0
                self._step()
                if self.ctrl.stopped or self.ctrl.aborted:
                    return "stopped" if self.ctrl.stopped else "aborted"

            # Speed adjustment from +/- commands
            delta = self.ctrl.pop_speed_delta()
            if delta != 0.0:
                vx_limit = float(np.clip(vx_limit + delta, 0.1, 3.0))
                print(f"    speed adjusted to {vx_limit:.1f} m/s")

            x, y, yaw = self.pose_source.get()
            dx   = tx - x
            dy   = ty - y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < self.tol_pos:
                self._cmd[:] = 0.0
                return "done"

            desired_yaw = math.atan2(dy, dx)
            if reverse:
                desired_yaw = self._wrap(desired_yaw + math.pi)

            yaw_err   = self._wrap(desired_yaw - yaw)
            alignment = math.cos(yaw_err)
            sign      = -1.0 if reverse else 1.0

            vx   = sign * float(np.clip(
                self.Kp_lin * dist * max(alignment, 0.0),
                0.15, vx_limit
            ))
            vyaw = float(np.clip(
                self.Kp_ang * yaw_err, -self.max_vyaw, self.max_vyaw
            ))

            self._cmd[0, 0] = vx
            self._cmd[0, 1] = 0.0
            self._cmd[0, 2] = vyaw
            self._step()

            if step % 100 == 0:
                print(f"    dist={dist:.2f}m  vx={vx:.2f}  vyaw={vyaw:.2f}  "
                      f"pos=({x:.2f},{y:.2f})  yaw={math.degrees(yaw):.1f}°")

        return "timeout"

    def _rotate_to(self, target_yaw: float, timeout_s: float = 15.0) -> str:
        max_steps = int(timeout_s / self.env.dt)

        for _ in range(max_steps):
            if self.ctrl.stopped:
                self.stop(); return "stopped"
            if self.ctrl.aborted:
                self.stop(); return "aborted"
            while self.ctrl.paused:
                self._cmd[:] = 0.0
                self._step()

            _, _, yaw = self.pose_source.get()
            yaw_err   = self._wrap(target_yaw - yaw)

            if abs(yaw_err) < self.tol_ang:
                self._cmd[:] = 0.0
                return "done"

            self._cmd[0, 0] = 0.0
            self._cmd[0, 1] = 0.0
            self._cmd[0, 2] = float(np.clip(
                self.Kp_ang * yaw_err,
                -self.max_vyaw, self.max_vyaw
            ))
            self._step()

        return "timeout"

    def _step(self):
        self.ctrl.poll()   # check for intervention commands non-blocking
        self.env.commands[:] = self._cmd
        with torch.no_grad():
            act, _, _ = self.net.get_action(self._obs, deterministic=True)
        self._obs, _, _, _, _ = self.env.step(act)

    @staticmethod
    def _wrap(a: float) -> float:
        while a >  math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


# ==========================================================================
#  Gemini NL parser
# ==========================================================================

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
  "go 5 metres forward"                → [{"type":"forward","distance":5.0,"speed":null}]
  "go forward 3 metres at 1.5 m/s"    → [{"type":"forward","distance":3.0,"speed":1.5}]
  "turn right 90 degrees"              → [{"type":"turn","angle_deg":-90.0}]
  "go to 3 2 then turn left"           → [{"type":"go_to","x":3.0,"y":2.0,"speed":null},{"type":"turn","angle_deg":90.0}]
  "stop"                               → [{"type":"stop"}]
  "run forward 10 metres"              → [{"type":"forward","distance":10.0,"speed":2.0}]
  "walk slowly 2 metres"               → [{"type":"forward","distance":2.0,"speed":0.3}]"""


def parse_gemini(text: str) -> list:
    import json
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
    client   = genai.Client(api_key=key)
    response = client.models.generate_content(
        model    = "gemini-2.5-flash",
        contents = [types.Content(role="user", parts=[
            types.Part(text=SYSTEM_PROMPT_NAV + f'\n\nUser said: "{text}"')
        ])],
        config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=300),
    )
    raw = response.text.strip().replace("```json","").replace("```","").strip()
    return json.loads(raw)


def parse_simple(text: str) -> list:
    """Rule-based parser — no LLM needed for basic commands."""
    t = text.lower().strip()
    words = t.split()
    speed = None

    # extract "at N m/s" or "at N ms" pattern
    for i, w in enumerate(words):
        if w == "at" and i + 1 < len(words):
            try:
                speed = float(words[i+1])
            except ValueError:
                pass

    try:
        if t.startswith("forward") or t.startswith("go forward"):
            nums = [float(w) for w in words if _is_number(w)]
            d = nums[0] if nums else 1.0
            return [{"type": "forward", "distance": d, "speed": speed}]
        if t.startswith("backward") or t.startswith("go backward"):
            nums = [float(w) for w in words if _is_number(w)]
            d = nums[0] if nums else 1.0
            return [{"type": "backward", "distance": d, "speed": speed}]
        if t.startswith("turn"):
            nums = [float(w) for w in words if _is_number(w)]
            angle = nums[0] if nums else 90.0
            if "right" in t:
                angle = -abs(angle)
            return [{"type": "turn", "angle_deg": angle}]
        if t.startswith("go to"):
            nums = [float(w) for w in words if _is_number(w)]
            if len(nums) >= 2:
                return [{"type": "go_to", "x": nums[0], "y": nums[1],
                         "speed": speed}]
        if t == "stop" or t == "halt":
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


def execute_commands(commands: list, controller: PositionController):
    """Execute a list of parsed command dicts."""
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
            controller.go_to(
                float(cmd.get("x", 0.0)),
                float(cmd.get("y", 0.0)),
                speed=speed,
            )
        elif t == "stop":
            controller.stop()


# ==========================================================================
#  Demo modes
# ==========================================================================

def run_demo_forward(controller):
    print("\n=== Demo: forward sequence ===")
    controller.reset()
    execute_commands([
        {"type": "forward",  "distance": 5.0, "speed": None},
        {"type": "turn",     "angle_deg": 90.0},
        {"type": "forward",  "distance": 3.0, "speed": 0.5},
        {"type": "turn",     "angle_deg": -90.0},
        {"type": "stop"},
    ], controller)
    print("=== Done ===")


def run_demo_waypoints(controller):
    print("\n=== Demo: square waypoints ===")
    controller.reset()
    waypoints = [(3.0,0.0), (3.0,3.0), (0.0,3.0), (0.0,0.0)]
    for i, (x, y) in enumerate(waypoints):
        if controller.ctrl.stopped:
            break
        print(f"\n  Waypoint {i+1}/{len(waypoints)}: ({x},{y})")
        controller.go_to(x, y)
    controller.stop()
    print("=== Done ===")


def run_demo_interactive(controller, use_gemini=False):
    print("\n=== Interactive navigation ===")
    if use_gemini:
        print("Using Gemini for NL parsing. Examples:")
        print("  'go 3 metres forward at 1 m/s'")
        print("  'turn right 45 degrees'")
        print("  'run forward 5 metres'")
    else:
        print("Rule-based parser. Commands:")
        print("  forward N [at S]  |  backward N  |  turn N [left|right]")
        print("  go to X Y  |  stop")
    print("Intervention: p=pause r=resume a=abort s=stop +=faster -=slower\n")
    controller.reset()

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


def run_demo_voice(controller):
    print("\n=== Voice navigation (Whisper + Gemini) ===")
    print("Say navigation commands. Say 'quit' or press Ctrl+C to exit.\n")
    print("Intervention: p=pause r=resume a=abort s=stop\n")

    try:
        import whisper
        import sounddevice as sd
    except ImportError:
        print("  pip install openai-whisper sounddevice")
        return

    wmodel = whisper.load_model("tiny")
    controller.reset()

    try:
        while not controller.ctrl.stopped:
            audio = sd.rec(int(3.0*16000), samplerate=16000,
                           channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()
            if float(np.sqrt(np.mean(audio**2))) < 0.01:
                continue
            text = wmodel.transcribe(audio, fp16=False)["text"].strip()
            if not text:
                continue
            print(f"  [voice] Heard: \"{text}\"")
            if "quit" in text.lower():
                break
            cmds = parse_gemini(text)
            print(f"  → {cmds}")
            execute_commands(cmds, controller)
    except KeyboardInterrupt:
        pass

    controller.stop()
    print("=== Done ===")


# ==========================================================================
#  Loader + main
# ==========================================================================

def load_policy_and_env(checkpoint_path: str, headless: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg    = ckpt["config"]
    sd     = clean_state_dict(ckpt["model_state"])

    obs_dim = sd["trunk.0.weight"].shape[1]
    act_dim = sd["log_std"].shape[0]
    print(f"  Checkpoint: obs_dim={obs_dim}  act_dim={act_dim}  "
          f"step={ckpt.get('step',0):,}")

    if obs_dim != OBS_DIM:
        print(f"  [warn] obs_dim mismatch: ckpt={obs_dim} script={OBS_DIM}")
        sys.exit(1)

    env = Go2CPGEnv(
        n_envs            = 1,
        headless          = headless,
        max_episode_steps = 50000,
        dt                = cfg["dt"],
        device            = device,
    )
    net = ActorCritic(obs_dim, act_dim, cfg.get("hidden_size", 512))
    net.load_state_dict(sd)
    net.eval().to(device)
    return env, net, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--demo", default="forward",
                   choices=["forward","waypoints","interactive",
                            "interactive-llm","voice"])
    p.add_argument("--headless",  action="store_true")
    p.add_argument("--max-vx",    type=float, default=0.8)
    p.add_argument("--max-vyaw",  type=float, default=0.8)
    args = p.parse_args()

    print(f"\nLoading: {args.checkpoint}")
    env, net, device = load_policy_and_env(args.checkpoint, args.headless)

    ctrl       = MissionControl()
    ctrl.start()

    controller = PositionController(
        env         = env,
        net         = net,
        pose_source = SimPoseSource(env),
        ctrl        = ctrl,
        device      = device,
        max_vx      = args.max_vx,
        max_vyaw    = args.max_vyaw,
    )

    try:
        if args.demo == "forward":
            run_demo_forward(controller)
        elif args.demo == "waypoints":
            run_demo_waypoints(controller)
        elif args.demo == "interactive":
            run_demo_interactive(controller, use_gemini=False)
        elif args.demo == "interactive-llm":
            run_demo_interactive(controller, use_gemini=True)
        elif args.demo == "voice":
            run_demo_voice(controller)
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()
