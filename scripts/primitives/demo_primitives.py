"""
scripts/demo_primitives.py
--------------------------
Demonstrates Go2 factory-equivalent motion primitives in Genesis simulation.

Primitives implemented (matching SportClient API names):

  Pose-based (open-loop joint angle targets):
    BalanceStand()        — default standing pose
    StandDown()           — lower body / crouch
    Sit()                 — sit with rear legs folded
    Hello()               — raise FL paw (dar la patita)
    Stretch()             — play bow, rear up (levantar el culo)
    WiggleHips()          — sway hips side to side
    Euler(roll, pitch)    — tilt body in place

  Velocity-based (handcoded CPG trot gait):
    Move(vx, vy, vyaw)    — walk/trot/run at given velocity
    SpeedLevel(level)     — 0=walk, 1=trot, 2=run (preset speeds)

  The velocity primitives use a Central Pattern Generator (CPG):
  each leg follows a sinusoidal phase with diagonal pairs 180° offset
  (trot gait). No RL policy or checkpoint needed.

Usage:
    # All pose primitives in sequence
    python scripts/demo_primitives.py

    # Single primitive
    python scripts/demo_primitives.py --primitive Hello
    python scripts/demo_primitives.py --primitive Sit
    python scripts/demo_primitives.py --primitive Stretch

    # Walk forward 0.4 m/s for 8 seconds
    python scripts/demo_primitives.py --primitive Move --vx 0.4

    # Trot forward 0.8 m/s
    python scripts/demo_primitives.py --primitive Move --vx 0.8

    # Run 1.5 m/s
    python scripts/demo_primitives.py --primitive Move --vx 1.5

    # Turn left while walking
    python scripts/demo_primitives.py --primitive Move --vx 0.5 --vyaw 0.5

    # Speed level presets (walk/trot/run)
    python scripts/demo_primitives.py --primitive SpeedLevel --level 0
    python scripts/demo_primitives.py --primitive SpeedLevel --level 1
    python scripts/demo_primitives.py --primitive SpeedLevel --level 2

    # Run all (poses + walk + trot + run)
    python scripts/demo_primitives.py --primitive all

    # Headless (no viewer, for testing on H200)
    python scripts/demo_primitives.py --headless

    # List all primitives
    python scripts/demo_primitives.py --list
"""

import argparse
import math
import numpy as np
import torch
import genesis as gs


# ==========================================================================
#  Joint order — matches Go2WalkEnv MOTOR_JOINT_NAMES
# ==========================================================================

MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Index helpers
FR_HIP, FR_THIGH, FR_CALF = 0, 1, 2
FL_HIP, FL_THIGH, FL_CALF = 3, 4, 5
RR_HIP, RR_THIGH, RR_CALF = 6, 7, 8
RL_HIP, RL_THIGH, RL_CALF = 9, 10, 11


# ==========================================================================
#  Pose definitions
# ==========================================================================

def _pose(
    fr_hip=0.0, fr_thigh=0.8,  fr_calf=-1.5,
    fl_hip=0.0, fl_thigh=0.8,  fl_calf=-1.5,
    rr_hip=0.0, rr_thigh=1.0,  rr_calf=-1.5,
    rl_hip=0.0, rl_thigh=1.0,  rl_calf=-1.5,
) -> np.ndarray:
    return np.array([
        fr_hip, fr_thigh, fr_calf,
        fl_hip, fl_thigh, fl_calf,
        rr_hip, rr_thigh, rr_calf,
        rl_hip, rl_thigh, rl_calf,
    ], dtype=np.float32)


POSE_STAND = _pose()

POSE_STAND_DOWN = _pose(
    fr_thigh=1.4, fr_calf=-2.6,
    fl_thigh=1.4, fl_calf=-2.6,
    rr_thigh=1.5, rr_calf=-2.6,
    rl_thigh=1.5, rl_calf=-2.6,
)

POSE_SIT = _pose(
    fr_thigh=0.5, fr_calf=-1.0,
    fl_thigh=0.5, fl_calf=-1.0,
    rr_thigh=2.8, rr_calf=-2.5,
    rl_thigh=2.8, rl_calf=-2.5,
)

POSE_HELLO = _pose(
    fl_hip=-0.5, fl_thigh=-0.3, fl_calf=0.5,   # FL raised
    fr_hip= 0.0, fr_thigh= 0.8, fr_calf=-1.5,
    rr_hip=-0.15, rr_thigh=1.0, rr_calf=-1.5,
    rl_hip= 0.15, rl_thigh=1.0, rl_calf=-1.5,
)

POSE_STRETCH = _pose(
    fr_thigh=1.6, fr_calf=-2.5,
    fl_thigh=1.6, fl_calf=-2.5,
    rr_thigh=0.4, rr_calf=-0.9,
    rl_thigh=0.4, rl_calf=-0.9,
)


# ==========================================================================
#  CPG gait constants
# ==========================================================================

# Trot phase offsets [FR, FL, RR, RL] — diagonal pairs in sync
TROT_PHASE = [0.0, math.pi, math.pi, 0.0]

# Speed presets (vx m/s): level 0=walk, 1=trot, 2=run
SPEED_PRESETS = {0: 0.4, 1: 0.8, 2: 1.5}
SPEED_NAMES   = {0: "Walk", 1: "Trot", 2: "Run"}


# ==========================================================================
#  Demo class
# ==========================================================================

class Go2PrimitiveDemo:

    DT = 0.02   # 50 Hz control rate

    def __init__(self, headless: bool = False):
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(backend=backend, logging_level="warning")

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos    = (1.5, -1.5, 1.0),
                camera_lookat = (0.0,  0.0,  0.3),
                camera_fov    = 50,
                max_FPS       = 60,
            ),
            sim_options=gs.options.SimOptions(dt=self.DT, substeps=2),
            show_viewer=not headless,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = "urdf/go2/urdf/go2.urdf",
                pos  = (0.0, 0.0, 0.42),
                quat = (0.0, 0.0, 0.0, 1.0),
            )
        )

        self.scene.build(n_envs=1)

        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in MOTOR_JOINT_NAMES
        ]

        self._dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Pose gains — stiff for static holds
        self._set_gains(kp=40.0, kv=2.0)

        # Warm up in standing pose
        self._set_pose(POSE_STAND)
        for _ in range(50):
            self.scene.step()

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_gains(self, kp: float, kv: float):
        self.robot.set_dofs_kp([kp] * 12, dofs_idx_local=self.motor_dofs)
        self.robot.set_dofs_kv([kv] * 12, dofs_idx_local=self.motor_dofs)

    def _set_pose(self, target: np.ndarray):
        t = torch.tensor(target, device=self._dev).unsqueeze(0)
        self.robot.control_dofs_position(t, dofs_idx_local=self.motor_dofs)

    def _get_pose(self) -> np.ndarray:
        return self.robot.get_dofs_position(
            dofs_idx_local=self.motor_dofs
        )[0].cpu().numpy()

    def _interpolate(self, start: np.ndarray, target: np.ndarray,
                     duration: float = 1.5):
        n = int(duration / self.DT)
        for i in range(n):
            alpha = 0.5 * (1 - math.cos(math.pi * i / n))
            self._set_pose(start + alpha * (target - start))
            self.scene.step()

    def _hold(self, pose: np.ndarray, duration: float):
        for _ in range(int(duration / self.DT)):
            self._set_pose(pose)
            self.scene.step()

    def _follow_camera(self):
        if self.scene.viewer is None:
            return
        pos = self.robot.get_pos()[0].cpu().numpy()
        self.scene.viewer.set_camera_pose(
            pos    = pos + np.array([-1.5, -1.0, 0.8]),
            lookat = pos,
        )

    # ------------------------------------------------------------------
    # Pose primitives
    # ------------------------------------------------------------------

    def BalanceStand(self):
        print("  → BalanceStand")
        self._interpolate(self._get_pose(), POSE_STAND, 1.0)
        self._hold(POSE_STAND, 2.0)

    def StandDown(self):
        print("  → StandDown")
        self._interpolate(self._get_pose(), POSE_STAND_DOWN, 1.5)
        self._hold(POSE_STAND_DOWN, 2.0)
        self._interpolate(self._get_pose(), POSE_STAND, 1.5)

    def Sit(self):
        print("  → Sit")
        self._interpolate(self._get_pose(), POSE_SIT, 2.0)
        self._hold(POSE_SIT, 2.0)
        self._interpolate(self._get_pose(), POSE_STAND, 2.0)

    def Hello(self):
        print("  → Hello (dar la patita)")
        self._interpolate(self._get_pose(), POSE_HELLO, 1.5)
        # Sway the paw
        for cycle in range(3):
            sway = POSE_HELLO.copy()
            sway[FL_HIP] = -0.5 + 0.2 * math.sin(cycle * math.pi)
            self._interpolate(self._get_pose(), sway, 0.4)
        self._hold(POSE_HELLO, 0.5)
        self._interpolate(self._get_pose(), POSE_STAND, 1.5)

    def Stretch(self):
        print("  → Stretch (levantar el culo)")
        self._interpolate(self._get_pose(), POSE_STRETCH, 2.0)
        self._hold(POSE_STRETCH, 2.0)
        self._interpolate(self._get_pose(), POSE_STAND, 2.0)

    def WiggleHips(self):
        print("  → WiggleHips")
        freq, amp = 1.5, 0.25
        for i in range(int(4.0 / self.DT)):
            t    = i * self.DT
            sway = amp * math.sin(2 * math.pi * freq * t)
            pose = POSE_STAND.copy()
            pose[RR_HIP] =  sway
            pose[RL_HIP] = -sway
            pose[FR_HIP] = -sway * 0.3
            pose[FL_HIP] =  sway * 0.3
            self._set_pose(pose)
            self.scene.step()

    def Euler(self, roll: float = 0.0, pitch: float = 0.0):
        print(f"  → Euler: roll={roll:.2f} pitch={pitch:.2f} rad")
        target = POSE_STAND.copy()
        target[FR_THIGH] += pitch * 0.4;  target[FL_THIGH] += pitch * 0.4
        target[RR_THIGH] -= pitch * 0.4;  target[RL_THIGH] -= pitch * 0.4
        target[FR_THIGH] -= roll  * 0.3;  target[RR_THIGH] -= roll  * 0.3
        target[FL_THIGH] += roll  * 0.3;  target[RL_THIGH] += roll  * 0.3
        self._interpolate(self._get_pose(), target, 1.5)
        self._hold(target, 2.0)
        self._interpolate(self._get_pose(), POSE_STAND, 1.5)

    # ------------------------------------------------------------------
    # Velocity primitives — handcoded CPG trot gait
    # ------------------------------------------------------------------
    #
    # Central Pattern Generator (CPG):
    #   - Each leg oscillates sinusoidally at frequency `freq`
    #   - Diagonal pairs (FR+RL, FL+RR) are 180° out of phase → trot
    #   - Swing phase (sin > 0): thigh lifts, calf tucks → foot rises
    #   - Stance phase (sin < 0): thigh sweeps forward/back → propulsion
    #   - Forward motion scales the swing amplitude with |vx|
    #   - Turn: front hips abduct asymmetrically with vyaw

    def _cpg_pose(
        self,
        t:     float,
        vx:    float,
        vy:    float,
        vyaw:  float,
        freq:  float,
        lift:  float,
        swing: float,
    ) -> np.ndarray:
        pose = POSE_STAND.copy()

        swing_fwd = swing * min(1.0, abs(vx)  / 1.5 + 0.15)
        swing_lat = 0.15  * min(1.0, abs(vy)  / 0.5)
        swing_yaw = 0.15  * min(1.0, abs(vyaw)/ 1.0)

        legs = [
            # (thigh_idx, calf_idx, hip_idx, phase_offset, is_left, is_front)
            (FR_THIGH, FR_CALF, FR_HIP, TROT_PHASE[0], False, True),
            (FL_THIGH, FL_CALF, FL_HIP, TROT_PHASE[1], True,  True),
            (RR_THIGH, RR_CALF, RR_HIP, TROT_PHASE[2], False, False),
            (RL_THIGH, RL_CALF, RL_HIP, TROT_PHASE[3], True,  False),
        ]

        for th, cf, hp, ph, is_left, is_front in legs:
            phase = 2 * math.pi * freq * t + ph
            sin_p = math.sin(phase)
            cos_p = math.cos(phase)

            if sin_p > 0:
                # Swing: lift foot
                pose[th] += lift * sin_p
                pose[cf] -= lift * sin_p * 0.5
            else:
                # Stance: propulsion sweep
                pose[th] -= swing_fwd * cos_p * (1 if vx >= 0 else -1)

            # Lateral
            lat_sign = 1.0 if is_left else -1.0
            pose[hp] += lat_sign * swing_lat * (1 if vy >= 0 else -1)

            # Yaw: front legs steer
            if is_front:
                pose[hp] += swing_yaw * (1 if vyaw >= 0 else -1)

        return pose

    def Move(
        self,
        vx:       float = 0.5,
        vy:       float = 0.0,
        vyaw:     float = 0.0,
        duration: float = 8.0,
    ):
        """
        Walk/trot/run at given velocity using CPG gait.
        Equivalent to SportClient.Move(vx, vy, vyaw).

        vx:    forward velocity (m/s). [-1.5, 1.5]
        vy:    lateral velocity (m/s). [-0.5, 0.5]
        vyaw:  yaw rate (rad/s).       [-1.0,  1.0]
        """
        speed = abs(vx)
        if speed < 0.6:
            freq, lift, gait = 1.5, 0.25, "Walk"
        elif speed < 1.2:
            freq, lift, gait = 2.5, 0.30, "Trot"
        else:
            freq, lift, gait = 3.5, 0.20, "Run"

        print(f"  → Move ({gait}): vx={vx:.2f} vy={vy:.2f} "
              f"vyaw={vyaw:.2f}  [{freq}Hz]")

        # Gait gains — softer than pose for dynamic motion
        self._set_gains(kp=20.0, kv=0.5)

        n_steps    = int(duration / self.DT)
        ramp_steps = int(0.5 / self.DT)   # 0.5s ramp-up

        t = 0.0
        for step in range(n_steps):
            r  = min(1.0, step / ramp_steps)
            pose = self._cpg_pose(
                t, vx*r, vy*r, vyaw*r,
                freq=freq, lift=0.25, swing=0.25
            )
            self._set_pose(pose)
            self.scene.step()
            t += self.DT

            if step % 5 == 0:
                self._follow_camera()
            if step % 100 == 0:
                pos = self.robot.get_pos()[0].cpu().numpy()
                print(f"    t={t:.1f}s  "
                      f"x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}")

        # Restore pose gains and return to stand
        self._set_gains(kp=40.0, kv=2.0)
        self._interpolate(self._get_pose(), POSE_STAND, 1.0)

    def SpeedLevel(self, level: int = 1, duration: float = 8.0):
        """
        Walk/run at preset speed level.
        Equivalent to SportClient.SpeedLevel(level).

        level: 0=Walk (0.4 m/s), 1=Trot (0.8 m/s), 2=Run (1.5 m/s)
        """
        vx   = SPEED_PRESETS.get(level, 0.8)
        name = SPEED_NAMES.get(level, "Trot")
        print(f"  → SpeedLevel {level} ({name}): {vx} m/s")
        self.Move(vx=vx, duration=duration)

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(self):
        sequence = [
            ("BalanceStand", {}),
            ("StandDown",    {}),
            ("Sit",          {}),
            ("Hello",        {}),
            ("Stretch",      {}),
            ("WiggleHips",   {}),
            ("Euler",        {"pitch":  0.3}),
            ("Euler",        {"roll":   0.3}),
            ("Move",         {"vx": 0.4, "duration": 5.0}),
            ("Move",         {"vx": 0.8, "duration": 5.0}),
            ("Move",         {"vx": 1.5, "duration": 5.0}),
            ("Move",         {"vx": 0.5, "vyaw": 0.5, "duration": 5.0}),
            ("BalanceStand", {}),
        ]

        print(f"\nRunning {len(sequence)} primitives\n")
        for name, kwargs in sequence:
            print(f"\n[{name}]")
            getattr(self, name)(**kwargs)
            self._hold(POSE_STAND, 0.5)
        print("\nDone.")


# ==========================================================================
#  Entry point
# ==========================================================================

PRIMITIVE_LIST = [
    "BalanceStand", "StandDown", "Sit", "Hello",
    "Stretch", "WiggleHips", "Euler", "Move", "SpeedLevel",
]

def main():
    parser = argparse.ArgumentParser(
        description="Go2 motion primitives in Genesis simulation"
    )
    parser.add_argument("--primitive", "-p", default=None,
                        choices=PRIMITIVE_LIST + ["all"],
                        help="Primitive to run (default: all poses)")
    parser.add_argument("--headless",  action="store_true")
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--vx",        type=float, default=0.5)
    parser.add_argument("--vy",        type=float, default=0.0)
    parser.add_argument("--vyaw",      type=float, default=0.0)
    parser.add_argument("--duration",  type=float, default=8.0)
    parser.add_argument("--level",     type=int,   default=1)
    parser.add_argument("--roll",      type=float, default=0.0)
    parser.add_argument("--pitch",     type=float, default=0.3)
    args = parser.parse_args()

    if args.list:
        print("\nAvailable primitives:\n")
        descs = {
            "BalanceStand": "Default standing pose",
            "StandDown":    "Lower body / crouch",
            "Sit":          "Sit with rear legs folded",
            "Hello":        "Raise FL paw (dar la patita)",
            "Stretch":      "Play bow (levantar el culo)",
            "WiggleHips":   "Sway hips side to side",
            "Euler":        "Tilt body  --roll R --pitch P",
            "Move":         "Walk/run   --vx V --vy V --vyaw W --duration T",
            "SpeedLevel":   "Preset speed --level 0/1/2 --duration T",
        }
        for name, desc in descs.items():
            print(f"  {name:15s}  {desc}")
        print()
        return

    demo = Go2PrimitiveDemo(headless=args.headless)

    p = args.primitive
    if p is None or p == "all":
        demo.run_all()
    elif p == "Move":
        demo.Move(vx=args.vx, vy=args.vy, vyaw=args.vyaw,
                  duration=args.duration)
    elif p == "SpeedLevel":
        demo.SpeedLevel(level=args.level, duration=args.duration)
    elif p == "Euler":
        demo.Euler(roll=args.roll, pitch=args.pitch)
    else:
        getattr(demo, p)()


if __name__ == "__main__":
    main()
