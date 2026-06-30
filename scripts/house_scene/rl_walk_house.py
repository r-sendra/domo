"""
go2_cpg_nav.py
==============
Visual CPG-RL: Go2 locomotion + obstacle avoidance with randomised obstacles.

Extends go2_cpg_rl.py (Bellegarda & Ijspeert RA-L 2022) with:
  1. LiDAR sensor    — 36 horizontal rays, fast, above chassis
  2. Obstacle env    — random mix of cylinders, boxes, spheres each episode
  3. Expanded obs    — 76 proprioception + 36 LiDAR sectors = 112 dims
  4. Avoidance reward — penalty for approaching obstacles

Following: Visual CPG-RL (Bellegarda, Shafiee, Ijspeert — ICRA 2024)
  "the agent learns to coordinate rhythmic oscillator behaviour to
   track velocity commands while overriding them to avoid collisions"

LiDAR optimisations:
  - 36 rays only (vs 8192 default) — 227x fewer rays
  - Updated every LIDAR_INTERVAL steps (10 Hz vs 50 Hz control)
  - draw_debug=False during training

Obstacle configuration:
  - N_OBSTACLES total per env, mix of cylinders / boxes / spheres
  - Randomised size, position, orientation each episode
  - Placed in ring OBSTACLE_RING_MIN → OBSTACLE_RING_MAX from spawn
  - Inner exclusion zone keeps robot spawn clear

Usage:
    # Train on H200
    python go2_cpg_nav.py --n-envs 4096 --device cuda --headless
    # Evaluate on Mac
    python go2_cpg_nav.py --eval ../../runs/go2_cpg_nav/checkpoint_final.pt

    # Resume
    python go2_cpg_nav.py --resume ../../runs/go2_cpg_nav/checkpoint_step_XXXXXXXXX.pt
"""

import os
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
)


# ==========================================================================
#  Constants — locomotion (identical to go2_cpg_rl.py)
# ==========================================================================

MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
FOOT_LINK_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint":   0.0, "FR_thigh_joint":  0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint":   0.0, "FL_thigh_joint":  0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint":   0.0, "RR_thigh_joint":  1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint":   0.0, "RL_thigh_joint":  1.0, "RL_calf_joint": -1.5,
}

L_HIP, L_THIGH, L_CALF = 0.0955, 0.213, 0.213
LEG_SIDE_SIGN = [-1.0, +1.0, -1.0, +1.0]

A_CONV  = 150.0
D_STEP  = 0.20
H_NOM   = 0.30
G_CLEAR = 0.08
G_PEN   = 0.02
CPG_DT  = 0.001

TROT_PHASE = [0.0, math.pi, math.pi, 0.0]

MU_MIN,  MU_MAX  = 1.0, 2.0
OMG_MIN, OMG_MAX = 1.5, 3.5
PSI_MAX          = 1.5
MU_MID,  MU_HALF  = 0.5*(MU_MAX+MU_MIN),  0.5*(MU_MAX-MU_MIN)
OMG_MID, OMG_HALF = 0.5*(OMG_MAX+OMG_MIN), 0.5*(OMG_MAX-OMG_MIN)

CONTACT_FORCE_THRESH = 1.0

# ==========================================================================
#  Constants — LiDAR
# ==========================================================================

N_LIDAR_SECTORS  = 36         # horizontal sectors (obs size stays 36)
N_LIDAR_ELEV     = 5          # elevation layers: -30°, -15°, 0°, +10°, +20°
# Downward bias catches low objects; z=0.35 keeps sensor above chassis
# Total rays = 36 × 5 = 180
LIDAR_MAX_RANGE  = 4.0        # metres
LIDAR_INTERVAL   = 5          # update every N steps (10 Hz at 50 Hz control)
LIDAR_POS_OFFSET = (0.0, 0.0, 0.35)   # above chassis — safe from self-hits

# Collision thresholds for reward
LIDAR_COLLISION  = 0.20       # metres — episode termination
LIDAR_DANGER     = 0.50       # metres — strong quadratic penalty
LIDAR_CAUTION    = 1.20       # metres — linear penalty
LIDAR_ANTICIPATE = 2.50       # metres — early steering signal

# ==========================================================================
#  Constants — obstacles
# ==========================================================================

N_OBS_CHAIRS   = 3    # chair = 4 thin cylinders in a square
N_OBS_SOFAS    = 2    # sofa/cabinet = wide flat box
N_OBS_PILLARS  = 3    # lamp/column = tall thin cylinder
N_OBS_STEPS    = 2    # book/step = low flat box
N_OBS_BALLS    = 2    # ball/vase = sphere
N_OBSTACLES    = N_OBS_CHAIRS*4 + N_OBS_SOFAS + N_OBS_PILLARS + N_OBS_STEPS + N_OBS_BALLS
# Arena — square 8×8m (half-side = 4m)
ARENA_HALF        = 4.0
ARENA_WALL_HEIGHT = 0.6
ARENA_WALL_THICK  = 0.15
OBSTACLE_RING_MIN = 2.0    # safe spawn clearance — chair legs won't be at collision dist
OBSTACLE_RING_MAX = 3.5

# Chair: 4 thin legs in a square
CHAIR_LEG_RADIUS = 0.03
CHAIR_LEG_HEIGHT = 0.45
CHAIR_LEG_SPREAD = 0.25    # half-spacing between legs

# Sofa/cabinet: wide flat box
SOFA_WIDTH_RANGE  = (0.8,  1.6)
SOFA_DEPTH_RANGE  = (0.3,  0.5)
SOFA_HEIGHT_RANGE = (0.35, 0.50)

# Pillar/lamp: tall thin cylinder
PILLAR_RADIUS_RANGE = (0.04, 0.10)
PILLAR_HEIGHT_RANGE = (0.60, 1.20)

# Step/book: very low flat box
STEP_SIZE_RANGE   = (0.20, 0.50)
STEP_HEIGHT_RANGE = (0.04, 0.15)

# Ball/vase
SPHERE_RAD_RANGE  = (0.06, 0.16)

# ==========================================================================
#  Observation dims
# ==========================================================================

PROP_DIM  = 76                        # proprioception (matches go2_cpg_rl.py)
OBS_DIM   = PROP_DIM + N_LIDAR_SECTORS  # 76 + 36 = 112 (sectors, not raw rays)
ACT_DIM   = 12


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def clean_state_dict(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()}


# ==========================================================================
#  Kinematics (identical to go2_cpg_rl.py)
# ==========================================================================

def leg_fk(qh, qt, qc, side_sign, l1, l2, l3):
    s   = side_sign
    z_s = -l2*torch.cos(qt) - l3*torch.cos(qt+qc)
    px  = -l2*torch.sin(qt) - l3*torch.sin(qt+qc)
    py  =  s*l1*torch.cos(qh) - z_s*torch.sin(qh)
    pz  =  s*l1*torch.sin(qh) + z_s*torch.cos(qh)
    return px, py, pz


def leg_ik(px, py, pz, side_sign, l1, l2, l3):
    l1_s = side_sign * l1
    L    = torch.sqrt(torch.clamp(py*py + pz*pz - l1*l1, min=1e-8))
    z_s  = -L
    qh   = torch.atan2(L*py + l1_s*pz, l1_s*py - L*pz)
    D2   = px*px + L*L
    ck   = torch.clamp((D2-l2*l2-l3*l3)/(2.0*l2*l3), -1.0, 1.0)
    qc   = -torch.acos(ck)
    A    = l2 + l3*ck
    B    = l3*torch.sin(qc)
    qt   = torch.atan2(B*z_s - A*px, -B*px - A*z_s)
    return qh, qt, qc


# ==========================================================================
#  Environment
# ==========================================================================

class Go2CPGNavEnv:
    """
    CPG-RL locomotion environment with randomised obstacles and LiDAR.
    Observation: proprioception (76) + LiDAR sector mins (36) = 112 dims.
    """

    ACT_DIM = ACT_DIM

    def __init__(self, n_envs=4096, dt=0.02, max_episode_steps=1000,
                 headless=True, device="cuda"):
        self.n_envs   = n_envs
        self.num_envs = n_envs
        self.dt       = dt
        self.device   = torch.device(device)

        self.env_cfg = {
            "base_init_pos":  [0.0, 0.0, 0.42],
            "base_init_quat": [0.0, 0.0, 0.0, 1.0],
            "episode_length_s":  max_episode_steps * dt,
            "resampling_time_s": 4.0,
            "kp": 100.0, "kd": 2.0,
            "termination_if_pitch_greater_than": 1.0,
            "termination_if_roll_greater_than":  1.0,
            "termination_if_height_less_than":   0.18,
        }
        self.obs_scales = {
            "lin_vel": 2.0, "ang_vel": 0.25,
            "dof_pos": 1.0, "dof_vel": 0.05,
        }
        self.reward_cfg = {
            "tracking_sigma": 0.5,
            "reward_scales": {
                "tracking_lin_vel_x":  0.75,
                "tracking_lin_vel_y":  0.75,
                "tracking_ang_vel":    0.75,
                "lin_vel_z":          -2.00,
                "ang_vel_xy":         -0.05,
                "work":               -0.001,
                "obstacle_avoidance": -4.00,
                "survival":           +0.10,
                "forward_progress":   +0.50,   # reward actual movement — breaks standing still
                "stall_penalty":      -0.30,   # penalise near-zero velocity near obstacles
            },
        }

        # Avoidance curriculum — disabled during eval (n_envs=1)
        self.curriculum_steps    = 20_000_000
        self.curriculum_step_ctr = 0
        self._use_curriculum     = (n_envs > 1)   # off during eval
        self.command_cfg = {
            "lin_vel_x_range": [0.5, 1.5],   # min 0.5 — always commanded to move
            "lin_vel_y_range": [-0.3, 0.3],
            "ang_vel_range":   [-1.0, 1.0],
        }

        self.reward_scales      = {k: v*dt for k, v in self.reward_cfg["reward_scales"].items()}
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / dt)
        self.num_commands       = 3
        self.n_cpg_substeps     = max(1, round(dt / CPG_DT))
        self.cpg_dt             = dt / self.n_cpg_substeps

        # ── Genesis ───────────────────────────────────────────────────────
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(backend=backend, precision="32",
                logging_level="warning", performance_mode=True)

        self.scene = gs.Scene(
            sim_options    = gs.options.SimOptions(dt=dt, substeps=2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS      = int(0.5/dt),
                camera_pos   = (3.0, -3.0, 2.5),
                camera_lookat= (0.0,  0.0, 0.3),
                camera_fov   = 50,
            ),
            vis_options  = gs.options.VisOptions(n_rendered_envs=1),
            rigid_options= gs.options.RigidOptions(
                dt=dt, constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True, enable_joint_limit=True,
                iterations=100,
            ),
            renderer    = gs.renderers.Rasterizer(),   # needed for camera.render()
            show_viewer = not headless,
        )

        # ── Ground ────────────────────────────────────────────────────────
        self.scene.add_entity(gs.morphs.Plane())

        # ── Arena walls — simple square 6×6m ─────────────────────────────
        s = ARENA_HALF
        t = ARENA_WALL_THICK
        h = ARENA_WALL_HEIGHT
        # Four walls: (pos_x, pos_y, size_x, size_y)
        wall_defs = [
            ( 0.0,   s,   2*s + 2*t, t),   # north
            ( 0.0,  -s,   2*s + 2*t, t),   # south
            ( s,     0.0, t,          2*s), # east
            (-s,     0.0, t,          2*s), # west
        ]
        for wx, wy, sx, sy in wall_defs:
            self.scene.add_entity(
                gs.morphs.Box(
                    size  = (sx, sy, h),
                    pos   = (wx, wy, h / 2),
                    fixed = True,
                )
            )
        self.arena_term_dist = s + 0.5   # termination threshold
        print(f"  Arena: {2*s:.0f}×{2*s:.0f}m square, walls h={h}m")

        # ── Obstacles — realistic indoor objects ──────────────────────────
        # Pre-allocated at build time; positions randomised each episode.
        # Types: chairs (4-leg clusters), sofas, pillars, steps, balls.
        print(f"\n  Building obstacles: "
              f"{N_OBS_CHAIRS} chairs, {N_OBS_SOFAS} sofas, "
              f"{N_OBS_PILLARS} pillars, {N_OBS_STEPS} steps, "
              f"{N_OBS_BALLS} balls → {N_OBSTACLES} total entities")

        self.obstacles = []   # list of (type, entity, height, metadata)

        # ── Chairs: 4 thin cylinder legs per chair ────────────────────────
        # Each chair = 4 entities. Positions set together in _randomise_obstacles.
        # leg_offsets: corners of a square with side CHAIR_LEG_SPREAD*2
        leg_offsets = [
            ( CHAIR_LEG_SPREAD,  CHAIR_LEG_SPREAD),
            ( CHAIR_LEG_SPREAD, -CHAIR_LEG_SPREAD),
            (-CHAIR_LEG_SPREAD,  CHAIR_LEG_SPREAD),
            (-CHAIR_LEG_SPREAD, -CHAIR_LEG_SPREAD),
        ]
        for c in range(N_OBS_CHAIRS):
            legs = []
            for dx, dy in leg_offsets:
                e = self.scene.add_entity(
                    gs.morphs.Cylinder(
                        radius = CHAIR_LEG_RADIUS,
                        height = CHAIR_LEG_HEIGHT,
                        pos    = (99.0 + dx, 99.0 + dy, CHAIR_LEG_HEIGHT/2),
                        fixed  = True,
                    )
                )
                legs.append(e)
            # Store as one logical chair: all 4 legs together
            self.obstacles.append(("chair", legs, CHAIR_LEG_HEIGHT, leg_offsets))

        # ── Sofas/cabinets: wide flat box ─────────────────────────────────
        for i in range(N_OBS_SOFAS):
            t = i / max(N_OBS_SOFAS-1, 1)
            w = SOFA_WIDTH_RANGE[0]  + (SOFA_WIDTH_RANGE[1]  - SOFA_WIDTH_RANGE[0])  * t
            d = SOFA_DEPTH_RANGE[0]  + (SOFA_DEPTH_RANGE[1]  - SOFA_DEPTH_RANGE[0])  * t
            h = SOFA_HEIGHT_RANGE[0] + (SOFA_HEIGHT_RANGE[1] - SOFA_HEIGHT_RANGE[0]) * t
            e = self.scene.add_entity(
                gs.morphs.Box(size=(w, d, h), pos=(99.0, 99.0, h/2), fixed=True)
            )
            self.obstacles.append(("sofa", e, h, None))

        # ── Pillars/lamps: tall thin cylinder ─────────────────────────────
        for i in range(N_OBS_PILLARS):
            t = i / max(N_OBS_PILLARS-1, 1)
            r = PILLAR_RADIUS_RANGE[0] + (PILLAR_RADIUS_RANGE[1] - PILLAR_RADIUS_RANGE[0]) * t
            h = PILLAR_HEIGHT_RANGE[0] + (PILLAR_HEIGHT_RANGE[1] - PILLAR_HEIGHT_RANGE[0]) * t
            e = self.scene.add_entity(
                gs.morphs.Cylinder(radius=r, height=h,
                                   pos=(99.0, 99.0, h/2), fixed=True)
            )
            self.obstacles.append(("pillar", e, h, None))

        # ── Steps/books: very low flat box ────────────────────────────────
        for i in range(N_OBS_STEPS):
            t = i / max(N_OBS_STEPS-1, 1)
            s = STEP_SIZE_RANGE[0]   + (STEP_SIZE_RANGE[1]   - STEP_SIZE_RANGE[0])   * t
            h = STEP_HEIGHT_RANGE[0] + (STEP_HEIGHT_RANGE[1] - STEP_HEIGHT_RANGE[0]) * t
            e = self.scene.add_entity(
                gs.morphs.Box(size=(s, s, h), pos=(99.0, 99.0, h/2), fixed=True)
            )
            self.obstacles.append(("step", e, h, None))

        # ── Balls/vases: sphere ───────────────────────────────────────────
        for i in range(N_OBS_BALLS):
            t = i / max(N_OBS_BALLS-1, 1)
            r = SPHERE_RAD_RANGE[0] + (SPHERE_RAD_RANGE[1] - SPHERE_RAD_RANGE[0]) * t
            e = self.scene.add_entity(
                gs.morphs.Sphere(radius=r, pos=(99.0, 99.0, r), fixed=True)
            )
            self.obstacles.append(("ball", e, r*2, None))

        # ── Robot ─────────────────────────────────────────────────────────
        self.base_init_pos  = torch.tensor(
            self.env_cfg["base_init_pos"],  device=self.device)
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = "urdf/go2/urdf/go2.urdf",
                pos  = self.base_init_pos.cpu().numpy(),
                quat = self.base_init_quat.cpu().numpy(),
            )
        )

        # ── LiDAR ─────────────────────────────────────────────────────────
        # 36 horizontal rays — minimal ray count for maximum speed.
        # Updated every LIDAR_INTERVAL steps (10 Hz).
        self.lidar = self.scene.add_sensor(
            gs.sensors.Lidar(
                pattern = gs.sensors.SphericalPattern(
                    fov      = (360.0, 50.0),   # 360° horiz, 50° vert
                    n_points = (N_LIDAR_SECTORS, N_LIDAR_ELEV),  # 36×5=180
                ),
                entity_idx         = self.robot.idx,
                pos_offset         = LIDAR_POS_OFFSET,
                return_world_frame = True,
                draw_debug         = (not headless),
            )
        )

        # ── Front camera (BEFORE build — Genesis requires this) ───────────
        # Visualization only — not used in RL at all.
        self.front_cam    = None
        self.cam_interval = 10
        self._cam_step    = 0

        if not headless:
            try:
                self.front_cam = self.scene.add_camera(
                    res    = (640, 360),
                    pos    = (0.3, 0.0, 0.1),
                    lookat = (1.0, 0.0, 0.1),
                    fov    = 60,
                    GUI    = False,
                )
                self._cam_frame   = None
                self._pip_ready   = False
                print(f"  ✅  Front camera created (640×360, FOV=60°)")
            except Exception as e:
                print(f"  ⚠️  Front camera failed: {e}")
                self.front_cam = None

        # ── Build ─────────────────────────────────────────────────────────
        self.scene.build(n_envs=n_envs)

        # ── Motors ────────────────────────────────────────────────────────
        self.motor_dofs = [
            self.robot.get_joint(n).dof_idx_local for n in MOTOR_JOINT_NAMES
        ]
        self.robot.set_dofs_kp([self.env_cfg["kp"]]*ACT_DIM, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]]*ACT_DIM, self.motor_dofs)
        self.default_dof_pos = torch.tensor(
            [DEFAULT_JOINT_ANGLES[n] for n in MOTOR_JOINT_NAMES],
            device=self.device, dtype=gs.tc_float)

        self.side_sign  = torch.tensor(LEG_SIDE_SIGN,  device=self.device, dtype=gs.tc_float)
        self.trot_phase = torch.tensor(TROT_PHASE,     device=self.device, dtype=gs.tc_float)

        # ── Contacts ──────────────────────────────────────────────────────
        self.contact_mode = "phase"
        try:
            self.foot_link_idx = [
                self.robot.get_link(n).idx_local for n in FOOT_LINK_NAMES
            ]
            _ = self.robot.get_links_net_contact_force()
            self.contact_mode = "force"
        except Exception as e:
            print(f"  [contacts] using phase proxy ({e})")

        # ── Kuramoto phi_star (precomputed, not recreated each step) ──────
        self.phi_star = torch.tensor([
            [0.0,       math.pi, math.pi, 0.0    ],
            [math.pi,   0.0,     0.0,     math.pi],
            [math.pi,   0.0,     0.0,     math.pi],
            [0.0,       math.pi, math.pi, 0.0    ],
        ], device=self.device, dtype=gs.tc_float)

        # ── IK self-test ──────────────────────────────────────────────────
        with torch.no_grad():
            qh0 = self.default_dof_pos[0::3].view(1, 4)
            qt0 = self.default_dof_pos[1::3].view(1, 4)
            qc0 = self.default_dof_pos[2::3].view(1, 4)
            fx, fy, fz = leg_fk(qh0, qt0, qc0, self.side_sign,
                                 L_HIP, L_THIGH, L_CALF)
            rh, rt, rc = leg_ik(fx, fy, fz, self.side_sign,
                                 L_HIP, L_THIGH, L_CALF)
            err = (torch.abs(rh-qh0)+torch.abs(rt-qt0)+torch.abs(rc-qc0)).max().item()

        print(f"\n{'='*60}")
        print(f"  Go2 Visual CPG-RL Navigation")
        print(f"{'='*60}")
        print(f"  Envs          : {n_envs}")
        print(f"  Control dt    : {dt}s ({1/dt:.0f} Hz) | CPG {self.n_cpg_substeps}x ({1/self.cpg_dt:.0f} Hz)")
        print(f"  Obs / Act     : {OBS_DIM} / {ACT_DIM}  "
              f"({PROP_DIM} prop + {N_LIDAR_SECTORS} LiDAR)")
        print(f"  LiDAR         : {N_LIDAR_SECTORS} sectors × {N_LIDAR_ELEV} elevations "
              f"= {N_LIDAR_SECTORS*N_LIDAR_ELEV} rays, FOV 55° downward-biased, "
              f"max {LIDAR_MAX_RANGE}m, update every {LIDAR_INTERVAL} steps")
        print(f"  Obstacles     : {N_OBSTACLES} total, "
              f"ring {OBSTACLE_RING_MIN}-{OBSTACLE_RING_MAX}m")
        print(f"  IK self-test  : max error {err:.2e} rad "
              f"[{'OK' if err < 1e-4 else 'WARN'}]")
        print(f"{'='*60}\n")

        # ── Reward registry ───────────────────────────────────────────────
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name] = torch.zeros(
                (n_envs,), device=self.device, dtype=gs.tc_float)

        # ── State buffers ─────────────────────────────────────────────────
        N, f = n_envs, gs.tc_float
        self.base_lin_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor(
            [0., 0., -1.], device=self.device, dtype=f).repeat(N, 1)
        self.base_pos          = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat         = torch.zeros((N, 4), device=self.device, dtype=f)
        self.base_euler        = torch.zeros((N, 3), device=self.device, dtype=f)

        self.dof_pos        = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.dof_vel        = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.last_dof_vel   = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.target_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(N, 1).clone()
        self.applied_torque = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)

        self.actions      = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)
        self.last_actions = torch.zeros((N, ACT_DIM), device=self.device, dtype=f)

        self.cpg_r     = torch.ones( (N, 4), device=self.device, dtype=f)
        self.cpg_rdot  = torch.zeros((N, 4), device=self.device, dtype=f)
        self.cpg_theta = torch.zeros((N, 4), device=self.device, dtype=f)
        self.cpg_phi   = torch.zeros((N, 4), device=self.device, dtype=f)
        self.foot_contacts = torch.zeros((N, 4), device=self.device, dtype=f)

        self.commands       = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"],
             self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"]],
            device=self.device, dtype=f)

        # LiDAR sector buffer — updated every LIDAR_INTERVAL steps
        self.lidar_sectors = torch.full(
            (N, N_LIDAR_SECTORS), fill_value=LIDAR_MAX_RANGE,
            device=self.device, dtype=f)
        self._lidar_step = 0
        self._cam_step   = 0

        self.obs_buf            = torch.zeros((N, OBS_DIM),  device=self.device, dtype=f)
        self.rew_buf            = torch.zeros((N,),           device=self.device, dtype=f)
        self.reset_buf          = torch.ones( (N,),           device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,),           device=self.device, dtype=gs.tc_int)
        self.extras = {}

    # ------------------------------------------------------------------
    # CPG
    # ------------------------------------------------------------------

    def _map_action(self, raw):
        t        = torch.tanh(raw)
        mu       = MU_MID  + MU_HALF  * t[:, 0:4]
        omega_hz = OMG_MID + OMG_HALF * t[:, 4:8]
        psi      = PSI_MAX * t[:, 8:12]
        return mu, omega_hz, psi

    def _integrate_cpg(self, mu, omega_hz, psi):
        """CPG integration with Kuramoto inter-leg coupling (trot pattern)."""
        omega = 2.0 * math.pi * omega_hz
        a, dt_c = A_CONV, self.cpg_dt
        COUPLING_W = 2.0

        for _ in range(self.n_cpg_substeps):
            theta_i  = self.cpg_theta.unsqueeze(2)   # [N,4,1]
            theta_j  = self.cpg_theta.unsqueeze(1)   # [N,1,4]
            coupling = COUPLING_W * torch.sum(
                torch.sin(theta_j - theta_i - self.phi_star), dim=2
            )
            r_ddot         = a * (0.25*a*(mu - self.cpg_r) - self.cpg_rdot)
            self.cpg_rdot  = self.cpg_rdot  + r_ddot          * dt_c
            self.cpg_r     = self.cpg_r     + self.cpg_rdot   * dt_c
            self.cpg_theta = self.cpg_theta + (omega+coupling) * dt_c
            self.cpg_phi   = self.cpg_phi   + psi              * dt_c

        self.cpg_theta = torch.remainder(self.cpg_theta, 2*math.pi)
        self.cpg_phi   = torch.remainder(self.cpg_phi,   2*math.pi)

    def _cpg_to_joint_targets(self):
        amp  = D_STEP * (self.cpg_r - 1.0)
        ct, st = torch.cos(self.cpg_theta), torch.sin(self.cpg_theta)
        cp, sp = torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi)
        px = -amp * ct * cp
        py = self.side_sign * L_HIP - amp * ct * sp
        z_clear = torch.where(st > 0, G_CLEAR*st, G_PEN*st)
        pz = -H_NOM + z_clear
        qh, qt, qc = leg_ik(px, py, pz, self.side_sign, L_HIP, L_THIGH, L_CALF)
        self.target_dof_pos[:, 0::3] = qh
        self.target_dof_pos[:, 1::3] = qt
        self.target_dof_pos[:, 2::3] = qc

    def _update_foot_contacts(self):
        if self.contact_mode == "force":
            try:
                ff = self.robot.get_links_net_contact_force()
                ff = ff[:, self.foot_link_idx, :]
                self.foot_contacts = (torch.norm(ff, dim=-1) > CONTACT_FORCE_THRESH).float()
                return
            except Exception:
                self.contact_mode = "phase"
        self.foot_contacts = (torch.sin(self.cpg_theta) < 0).float()

    # ------------------------------------------------------------------
    # LiDAR update
    # ------------------------------------------------------------------

    def _update_lidar(self):
        """
        Read LiDAR and aggregate into sector minimums.
        Multi-elevation: min across elevation layers per sector catches
        low objects (chair legs, balls, steps) missed by flat ring.
        Raw oscillation is intentionally preserved — the policy learns
        to handle it, matching real robot sensor behaviour.
        """
        raw_ = self.lidar.read().distances

        if raw_.dim() == 3:
            n_env, n_elev, n_horiz = raw_.shape
            if n_horiz == N_LIDAR_SECTORS:
                self.lidar_sectors[:] = raw_.min(dim=1).values
            else:
                raw = raw_.reshape(n_env, -1)
                n_raw = raw.shape[1]
                rps   = n_raw // N_LIDAR_SECTORS
                n_trim = rps * N_LIDAR_SECTORS
                self.lidar_sectors[:] = raw[:, :n_trim].view(
                    n_env, N_LIDAR_SECTORS, rps
                ).min(dim=2).values
        else:
            raw   = raw_.reshape(self.n_envs, -1)
            n_raw = raw.shape[1]
            if n_raw >= N_LIDAR_SECTORS:
                rps    = n_raw // N_LIDAR_SECTORS
                n_trim = rps * N_LIDAR_SECTORS
                self.lidar_sectors[:] = raw[:, :n_trim].view(
                    self.n_envs, N_LIDAR_SECTORS, rps
                ).min(dim=2).values
            else:
                pad = raw.repeat(1, math.ceil(N_LIDAR_SECTORS / n_raw))
                self.lidar_sectors[:] = pad[:, :N_LIDAR_SECTORS]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _compute_observation(self):
        lidar_norm = torch.clamp(
            self.lidar_sectors / LIDAR_MAX_RANGE, 0.0, 1.0
        )
        self.obs_buf = torch.cat([
            self.base_lin_vel * self.obs_scales["lin_vel"],                      # 3
            self.base_ang_vel * self.obs_scales["ang_vel"],                      # 3
            self.projected_gravity,                                              # 3
            self.commands * self.commands_scale,                                 # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"], # 12
            self.dof_vel * self.obs_scales["dof_vel"],                           # 12
            self.last_actions,                                                   # 12
            self.foot_contacts,                                                  # 4
            self.cpg_r, self.cpg_rdot,                                           # 8
            torch.cos(self.cpg_theta), torch.sin(self.cpg_theta),               # 8
            torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi),                 # 8
            lidar_norm,                                                          # 36
        ], dim=-1)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = actions
        mu, omega_hz, psi = self._map_action(self.actions)
        self._integrate_cpg(mu, omega_hz, psi)
        self._cpg_to_joint_targets()
        self.robot.control_dofs_position(self.target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self._lidar_step         += 1
        self._cam_step           += 1

        # Front camera render — head-mounted PiP via OpenCV
        if (self.front_cam is not None
                and self._cam_step % self.cam_interval == 0):
            try:
                import math as _math, cv2
                pos  = self.robot.get_pos()[0].cpu().numpy()
                quat = self.robot.get_quat()[0].cpu().numpy()
                yaw  = float(quat_to_xyz(
                    torch.tensor(quat).unsqueeze(0)
                )[0, 2].item())

                cam_x  = pos[0] + 0.25 * _math.cos(yaw)
                cam_y  = pos[1] + 0.25 * _math.sin(yaw)
                cam_z  = pos[2] + 0.15
                look_x = pos[0] + 2.0 * _math.cos(yaw)
                look_y = pos[1] + 2.0 * _math.sin(yaw)
                look_z = pos[2] + 0.05

                self.front_cam.set_pose(
                    pos=(cam_x, cam_y, cam_z),
                    lookat=(look_x, look_y, look_z),
                )
                rgb, _, _, _ = self.front_cam.render(rgb=True)
                frame = np.array(rgb, dtype=np.uint8)
                bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Resize + label
                pip = cv2.resize(bgr, (320, 180))
                pip = cv2.copyMakeBorder(
                    pip, 2, 2, 2, 2,
                    cv2.BORDER_CONSTANT, value=(0, 220, 0)
                )
                cv2.putText(pip, "Front Cam", (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0), 1, cv2.LINE_AA)

                if not self._pip_ready:
                    cv2.namedWindow("pip", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("pip", 324, 184)
                    cv2.moveWindow("pip", 950, 10)
                    self._pip_ready = True

                cv2.imshow("pip", pip)
                # pollKey() is truly non-blocking — does NOT consume
                # mouse events, safe for macOS main thread use
                cv2.pollKey()

                if not hasattr(self, '_cam_ok'):
                    print("  [cam] PiP active")
                    self._cam_ok = True

            except Exception as e:
                if not hasattr(self, '_cam_warn'):
                    print(f"  ⚠️  Camera: {e}")
                    self._cam_warn = True

        # State update
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler   = quat_to_xyz(transform_quat_by_quat(
            torch.ones_like(self.base_quat) * self.inv_base_init_quat,
            self.base_quat))
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.applied_torque = (
            self.env_cfg["kp"] * (self.target_dof_pos - self.dof_pos)
            - self.env_cfg["kd"] * self.dof_vel
        )
        self._update_foot_contacts()

        # LiDAR — update every LIDAR_INTERVAL steps only
        if self._lidar_step % LIDAR_INTERVAL == 0:
            self._update_lidar()

        # Command resample
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (
            self.episode_length_buf % resample_every == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)

        # Termination
        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_less_than"]
        # Collision termination — only after first LiDAR update (step >= LIDAR_INTERVAL)
        # Avoids false termination from stale max-range initialisation at spawn
        if self._lidar_step >= LIDAR_INTERVAL:
            self.reset_buf |= self.lidar_sectors.min(dim=1).values < LIDAR_COLLISION
        # Arena boundary
        self.reset_buf |= self.base_pos[:, 0].abs() > self.arena_term_dist
        self.reset_buf |= self.base_pos[:, 1].abs() > self.arena_term_dist

        time_out_idx = (
            self.episode_length_buf > self.max_episode_length
        ).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.curriculum_step_ctr += self.n_envs

        # Curriculum tracking scale — ramps 0→1 during training, fixed 1.0 at eval
        if self._use_curriculum:
            tracking_scale = min(
                self.curriculum_step_ctr / max(self.curriculum_steps, 1), 1.0
            )
        else:
            tracking_scale = 1.0

        # Reward
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            # Scale tracking rewards during curriculum phase 1
            if name.startswith("tracking_"):
                rew = rew * tracking_scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        self.rew_buf = torch.clamp(self.rew_buf, -10.0, 10.0)

        self._compute_observation()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_q = inv_quat(self.base_quat)
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_q)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_q)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_q)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self._update_lidar()
        self._update_foot_contacts()
        self._compute_observation()
        assert self.obs_buf.shape[-1] == OBS_DIM, \
            f"obs dim {self.obs_buf.shape[-1]} != OBS_DIM {OBS_DIM}"
        return self.obs_buf, None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Joints
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position       = self.dof_pos[envs_idx],
            dofs_idx_local = self.motor_dofs,
            zero_velocity  = True,
            envs_idx       = envs_idx,
        )

        # Base pose — fixed identity quat at spawn
        # Random yaw causes CPG instability when training from scratch.
        # Re-enable after policy is stable (>50M steps).
        n = len(envs_idx)
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.robot.set_pos( self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # CPG reset
        self.cpg_r[envs_idx]     = 1.0
        self.cpg_rdot[envs_idx]  = 0.0
        self.cpg_theta[envs_idx] = self.trot_phase
        self.cpg_phi[envs_idx]   = self.trot_phase * 0.1

        # Reset LiDAR sectors for reset envs
        self.lidar_sectors[envs_idx] = LIDAR_MAX_RANGE

        # Randomise obstacle positions for reset envs
        self._randomise_obstacles(envs_idx)

        # Buffers
        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_"+key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"])
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        # Curriculum: scale vx commands 0→max over curriculum_steps
        # Phase 1: slow commands → survival is main objective
        # Phase 2: full range → track velocity while avoiding
        tracking_scale = min(
            self.curriculum_step_ctr / max(self.curriculum_steps, 1), 1.0
        )
        vx_lo = self.command_cfg["lin_vel_x_range"][0]
        vx_hi = self.command_cfg["lin_vel_x_range"][1]
        vx_hi_curr = max(vx_lo, vx_hi * tracking_scale)
        self.commands[envs_idx, 0] = gs_rand_float(
            vx_lo, vx_hi_curr, (n,), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (n,), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"],   (n,), self.device)

    def _randomise_obstacles(self, envs_idx):
        """
        Place obstacles at random positions in the ring around spawn.
        Chairs: all 4 legs placed together around one random centre.
        Other types: single entity per random position.
        """
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        spawn = self.base_init_pos

        for obs_type, entity, height, meta in self.obstacles:
            angles = gs_rand_float(0.0, 2*math.pi, (n,), self.device)
            radii  = gs_rand_float(
                OBSTACLE_RING_MIN, OBSTACLE_RING_MAX, (n,), self.device)
            cx = spawn[0] + radii * torch.cos(angles)
            cy = spawn[1] + radii * torch.sin(angles)

            if obs_type == "chair":
                # entity is a list of 4 leg entities
                # meta is list of (dx, dy) offsets
                # Add random yaw rotation to the chair
                chair_yaw = gs_rand_float(0.0, 2*math.pi, (n,), self.device)
                for leg_e, (dx, dy) in zip(entity, meta):
                    # Rotate leg offset by chair yaw
                    lx = cx + dx * torch.cos(chair_yaw) - dy * torch.sin(chair_yaw)
                    ly = cy + dx * torch.sin(chair_yaw) + dy * torch.cos(chair_yaw)
                    lz = torch.full((n,), height/2, device=self.device)
                    leg_e.set_pos(
                        torch.stack([lx, ly, lz], dim=-1),
                        envs_idx=envs_idx
                    )
            else:
                z   = torch.full((n,), height/2, device=self.device)
                pos = torch.stack([cx, cy, z], dim=-1)
                entity.set_pos(pos, envs_idx=envs_idx)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        return torch.exp(
            -torch.square(self.commands[:,0] - self.base_lin_vel[:,0])
            / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        return torch.exp(
            -torch.square(self.commands[:,1] - self.base_lin_vel[:,1])
            / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        return torch.exp(
            -torch.square(self.commands[:,2] - self.base_ang_vel[:,2])
            / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_work(self):
        dqd = self.dof_vel - self.last_dof_vel
        return torch.abs(torch.sum(self.applied_torque * dqd, dim=1))

    def _reward_obstacle_avoidance(self):
        """
        4-zone penalty based on minimum LiDAR distance.

        ANTICIPATE (< 2.5m): very light penalty — encourages early steering
        CAUTION    (< 1.5m): linear penalty — start turning now
        DANGER     (< 0.6m): strong quadratic — urgent avoidance
        COLLISION  (< 0.25m): maximum penalty — robot is touching obstacle

        The wide anticipation zone is the key improvement: the policy gets
        a gradient signal far enough ahead to actually change direction.
        """
        min_d = self.lidar_sectors.min(dim=1).values
        min_d = torch.clamp(min_d, 0.0, LIDAR_MAX_RANGE)

        # Zone 1: anticipate
        anticipate_pen = torch.where(
            (min_d >= LIDAR_CAUTION) & (min_d < LIDAR_ANTICIPATE),
            (LIDAR_ANTICIPATE - min_d) * 0.05,
            torch.zeros_like(min_d),
        )
        # Zone 2: caution
        caution_pen = torch.where(
            (min_d >= LIDAR_DANGER) & (min_d < LIDAR_CAUTION),
            (LIDAR_CAUTION - min_d) * 0.5,
            torch.zeros_like(min_d),
        )
        # Zone 3: danger
        danger_pen = torch.where(
            (min_d >= LIDAR_COLLISION) & (min_d < LIDAR_DANGER),
            torch.square(LIDAR_DANGER - min_d) * 2.0,
            torch.zeros_like(min_d),
        )
        # Zone 4: collision — maximum flat penalty
        collision_pen = torch.where(
            min_d < LIDAR_COLLISION,
            torch.full_like(min_d, 1.0),
            torch.zeros_like(min_d),
        )
        return anticipate_pen + caution_pen + danger_pen + collision_pen

    def _reward_survival(self):
        """+1 every step the robot is alive — incentivises avoiding resets."""
        return torch.ones(self.n_envs, device=self.device, dtype=gs.tc_float)

    def _reward_forward_progress(self):
        """
        Reward actual forward velocity — directly breaks the standing still optimum.
        Uses a sqrt to avoid overwhelming the tracking reward at high speeds.
        Only rewards positive forward velocity (not backward drift).
        """
        vx = torch.clamp(self.base_lin_vel[:, 0], 0.0, None)
        return torch.sqrt(vx + 1e-6) - math.sqrt(1e-6)   # 0 at vx=0, grows with speed

    def _reward_stall_penalty(self):
        """
        Penalise near-zero velocity when an obstacle is close.
        This specifically targets the standing-still-near-obstacle behaviour:
        if min_lidar < LIDAR_CAUTION and speed < threshold → penalty.
        Forces the policy to keep moving (turn away) rather than freeze.
        """
        min_d     = self.lidar_sectors.min(dim=1).values
        near_obs  = (min_d < LIDAR_CAUTION).float()
        speed     = torch.norm(self.base_lin_vel[:, :2], dim=1)
        stalled   = (speed < 0.15).float()   # below 15cm/s = stalled
        return near_obs * stalled


# ==========================================================================
#  ActorCritic
# ==========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head  = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim))
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.00)

    def forward(self, obs):
        h     = self.trunk(obs)
        mean  = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        std   = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        action   = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def get_value(self, obs):
        return self.critic_head(self.trunk(obs)).squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
#  RolloutBuffer
# ==========================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = rollout_steps, n_envs, device, 0
        b = lambda *s: torch.zeros(*s, device=device)
        self.obs        = b(rollout_steps, n_envs, obs_dim)
        self.actions    = b(rollout_steps, n_envs, act_dim)
        self.log_probs  = b(rollout_steps, n_envs)
        self.values     = b(rollout_steps, n_envs)
        self.rewards    = b(rollout_steps, n_envs)
        self.dones      = b(rollout_steps, n_envs)
        self.advantages = b(rollout_steps, n_envs)
        self.returns    = b(rollout_steps, n_envs)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t], self.actions[t]   = obs.detach(), actions.detach()
        self.log_probs[t], self.values[t] = log_probs.detach(), values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t] = rewards.detach().float()
        self.dones[t]   = dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            nv    = last_value if t == self.T-1 else self.values[t+1]
            mask  = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma*nv*mask - self.values[t]
            gae   = delta + gamma*lam*mask*gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        T, N = self.T, self.N
        return (self.obs.view(T*N,-1), self.actions.view(T*N,-1),
                self.log_probs.view(T*N), self.advantages.view(T*N),
                self.returns.view(T*N),   self.values.view(T*N))


# ==========================================================================
#  PPO Trainer
# ==========================================================================

class PPOTrainer:

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = cfg["device"]

        self.env = Go2CPGNavEnv(
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
            device            = cfg["device"],
        )

        self.net = ActorCritic(OBS_DIM, ACT_DIM, cfg["hidden_size"]).to(self.device)
        print(f"  Network params: {self.net.num_parameters:,}\n")

        # Optional: warm-start from pretrained CPG locomotion checkpoint
        if cfg.get("pretrained"):
            self._load_pretrained(cfg["pretrained"])
            # Freeze trunk for first N steps so LiDAR weights can warm up
            # without corrupting the pretrained walking behaviour
            freeze_steps = cfg.get("freeze_trunk_steps", 5_000_000)
            if freeze_steps > 0:
                for param in self.net.trunk.parameters():
                    param.requires_grad = False
                print(f"  Trunk frozen for first {freeze_steps:,} steps\n")

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg["lr"], eps=1e-5)
        self.buf = RolloutBuffer(
            cfg["rollout_steps"], cfg["n_envs"],
            OBS_DIM, ACT_DIM, self.device)

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()
        self.ep_returns, self.ep_lengths = [], []
        self._env_ret = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len = torch.zeros(cfg["n_envs"], device=self.device,
                                    dtype=torch.int32)

    def _load_pretrained(self, path: str):
        """
        Warm-start from go2_cpg_rl.py checkpoint (OBS_DIM=76).
        Copy matching weights; leave the 36 new LiDAR input weights near zero.
        """
        print(f"\n  Loading pretrained CPG weights: {path}")
        ckpt      = torch.load(path, weights_only=False, map_location=self.device)
        src       = clean_state_dict(ckpt["model_state"])
        dst       = self.net.state_dict()
        n_loaded  = 0
        for key, dp in dst.items():
            if key not in src:
                continue
            sp = src[key]
            if sp.shape == dp.shape:
                dst[key] = sp.clone()
                n_loaded += 1
            elif key == "trunk.0.weight":
                # [hidden, 112] ← [hidden, 76]: copy prop cols, LiDAR cols stay ~0
                dp[:, :sp.shape[1]] = sp.clone()
                dst[key] = dp
                n_loaded += 1
        self.net.load_state_dict(dst)
        print(f"  Loaded {n_loaded}/{len(dst)} tensors from pretrained.\n")

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()

        print(f"{'='*60}")
        print(f"  Visual CPG-RL Nav | {cfg['total_steps']:,} steps | {self.device}")
        print(f"{'='*60}\n")

        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates         = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates+1):
            # Unfreeze trunk once enough steps have passed
            freeze_steps = cfg.get("freeze_trunk_steps", 0)
            if (freeze_steps > 0
                    and self.global_step >= freeze_steps
                    and not self.net.trunk[0].weight.requires_grad):
                for param in self.net.trunk.parameters():
                    param.requires_grad = True
                print(f"\n  Trunk unfrozen at step {self.global_step:,}\n")
            frac = max(1.0 - self.global_step / max(cfg["total_steps"],1),
                       cfg["lr_floor_frac"])
            lr = cfg["lr"] * frac
            for g in self.opt.param_groups:
                g["lr"] = lr

            obs     = self._collect_rollout(obs)
            m       = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                sps  = self.global_step / (time.time() - self.start_time)
                mret = float(np.mean(self.ep_returns[-50:])) if self.ep_returns else 0.0
                mlen = float(np.mean(self.ep_lengths[-50:])) if self.ep_lengths else 0.0
                print(f"  step {self.global_step:>11,} | "
                      f"ret {mret:>7.3f} | len {mlen:>5.0f} | "
                      f"ploss {m['policy_loss']:>7.4f} | "
                      f"vloss {m['value_loss']:>8.3f} | "
                      f"kl {m['approx_kl']:>6.4f} | "
                      f"clip {m['clip_frac']:>4.2f} | "
                      f"lr {lr:.1e} | {sps:>7,.0f} sps")
                scalars = {
                    "train/mean_return": mret, "train/mean_ep_len": mlen,
                    "loss/policy": m["policy_loss"], "loss/value": m["value_loss"],
                    "loss/entropy": m["entropy"], "train/approx_kl": m["approx_kl"],
                    "train/clip_fraction": m["clip_frac"], "train/lr": lr,
                    "train/sps": sps,
                }
                for k, v in scalars.items():
                    self.writer.add_scalar(k, v, self.global_step)

            if update % cfg["save_interval"] == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.writer.close()
        print("\nTraining complete.")

    def _collect_rollout(self, obs):
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg["rollout_steps"]):
                action, log_prob, value = self.net.get_action(obs)
                self.buf.store_step(obs, action, log_prob, value)
                next_obs, _, reward, reset_buf, _ = self.env.step(action)
                self.buf.store_outcome(reward, reset_buf.float())
                self._env_ret += reward
                self._env_len += 1
                done_idx = reset_buf.nonzero(as_tuple=False).flatten()
                for idx in done_idx:
                    self.ep_returns.append(float(self._env_ret[idx]))
                    self.ep_lengths.append(int(self._env_len[idx]))
                self._env_ret[done_idx] = 0.0
                self._env_len[done_idx] = 0
                obs = next_obs
            last_value = self.net.get_value(obs)
            self.buf.compute_gae(last_value, self.cfg["gamma"], self.cfg["lam"])
        return obs

    def _ppo_update(self):
        self.net.train()
        cfg = self.cfg
        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()

        if not (torch.isfinite(adv_f).all() and torch.isfinite(ret_f).all()):
            print("  [warn] non-finite rollout; skipping update.")
            return {"policy_loss":0., "value_loss":0.,
                    "entropy":0., "clip_frac":0., "approx_kl":0.}

        backup    = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        adv_f     = ((adv_f - adv_f.mean()) / (adv_f.std()+1e-8)).clamp(-10., 10.)
        total     = obs_f.shape[0]
        target_kl = cfg.get("target_kl", 0.02)
        vloss_cap = cfg.get("vloss_skip", 1e3)
        m         = {"policy_loss":[], "value_loss":[], "entropy":[],
                     "clip_frac":[], "approx_kl":[]}
        bad = False

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total, device=self.device)
            epoch_kl = []
            for start in range(0, total, cfg["minibatch_size"]):
                mb = idx[start:start+cfg["minibatch_size"]]
                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                logratio  = new_lp - lp_f[mb]
                ratio     = logratio.exp()
                with torch.no_grad():
                    akl = ((ratio-1.0) - logratio).mean().item()
                epoch_kl.append(akl); m["approx_kl"].append(akl)

                surr1 = ratio * adv_f[mb]
                surr2 = ratio.clamp(1-cfg["clip_eps"], 1+cfg["clip_eps"]) * adv_f[mb]
                pl    = -torch.min(surr1, surr2).mean()
                vc    = val_f[mb] + (value-val_f[mb]).clamp(-cfg["clip_eps"], cfg["clip_eps"])
                vl    = torch.max((value-ret_f[mb]).pow(2), (vc-ret_f[mb]).pow(2)).mean()
                loss  = pl + cfg["vf_coef"]*vl - cfg["ent_coef"]*entropy.mean()

                if (not torch.isfinite(loss)) or (vl.item() > vloss_cap):
                    bad = True; break
                self.opt.zero_grad(); loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg["max_grad_norm"])
                if not torch.isfinite(gnorm):
                    bad = True; break
                self.opt.step()
                m["policy_loss"].append(pl.item())
                m["value_loss"].append(vl.item())
                m["entropy"].append(entropy.mean().item())
                m["clip_frac"].append(
                    ((ratio-1.0).abs() > cfg["clip_eps"]).float().mean().item())
            if bad: break
            if np.mean(epoch_kl) > 1.5*target_kl: break

        if bad:
            self.net.load_state_dict(backup)
            self.opt.zero_grad(set_to_none=True)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in m.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_{tag}" if tag \
               else f"checkpoint_step_{self.global_step:09d}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step":        self.global_step,
            "model_state": clean_state_dict(self.net.state_dict()),
            "optim_state": self.opt.state_dict(),
            "config":      self.cfg,
            "obs_dim":     OBS_DIM,
            "act_dim":     ACT_DIM,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-50:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-50:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] {path}")

    @classmethod
    def load_checkpoint(cls, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(path, weights_only=False, map_location=device)
        trainer = cls(ckpt["config"])
        trainer.net.load_state_dict(clean_state_dict(ckpt["model_state"]))
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(f"  Resumed step={trainer.global_step:,}  "
              f"ret={ckpt['metrics']['mean_return']:.3f}")
        return trainer


# ==========================================================================
#  Evaluation
# ==========================================================================

def evaluate(checkpoint_path, n_episodes=3,
             command=(0.5, 0.0, 0.0), headless=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg    = ckpt["config"]
    sd     = clean_state_dict(ckpt["model_state"])

    obs_dim = sd["trunk.0.weight"].shape[1]
    act_dim = sd["log_std"].shape[0]
    print(f"  Checkpoint: obs_dim={obs_dim}  act_dim={act_dim}  "
          f"step={ckpt.get('step',0):,}")

    if obs_dim != OBS_DIM:
        print(f"  [warn] obs mismatch: ckpt={obs_dim} script={OBS_DIM}. Retrain.")
        return

    env = Go2CPGNavEnv(
        n_envs            = 1,
        headless          = headless,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
        device            = device,
    )
    net = ActorCritic(obs_dim, act_dim, cfg.get("hidden_size", 512))
    net.load_state_dict(sd)
    net.eval().to(device)

    cmd = torch.tensor([command], device=device, dtype=gs.tc_float)
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len, vx_err = 0.0, 0, []
        while not done[0]:
            env.commands[:] = cmd
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done    = reset_buf.bool()
            vx_err.append(abs(command[0] - env.base_lin_vel[0,0].item()))
            if ep_len % 50 == 0:
                r   = env.cpg_r[0].cpu().numpy()
                md  = env.lidar_sectors[0].min().item()
                print(f"    step {ep_len:4d}  "
                      f"vx={env.base_lin_vel[0,0].item():+.2f}/{command[0]:.2f}  "
                      f"wz={env.base_ang_vel[0,2].item():+.2f}  "
                      f"h={env.base_pos[0,2].item():.2f}  "
                      f"min_lidar={md:.2f}m  "
                      f"r=[{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}]")
        print(f"  Episode {ep+1} | return={ep_ret:7.2f} | "
              f"length={ep_len:4d} | "
              f"mean|vx_err|={np.mean(vx_err):.3f}\n")


# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer = args.n_envs * args.rollout_steps
    # Detect if we are fine-tuning from an early checkpoint for avoidance
    # Fine-tune mode: lower LR + higher entropy to preserve locomotion
    # while allowing avoidance corrections to emerge
    is_finetune = args.resume is not None
    return dict(
        n_envs             = args.n_envs,
        dt                 = 0.02,
        max_episode_steps  = 1000,
        headless           = args.headless,
        hidden_size        = 512,
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = max(total_buffer // 4, 256),
        n_epochs           = 5,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 5e-5 if is_finetune else 3e-4,   # 6× lower for finetune
        vf_coef            = 1.0,
        ent_coef           = 0.05 if is_finetune else 0.01,   # 5× higher entropy for exploration
        max_grad_norm      = 0.5 if is_finetune else 1.0,     # tighter grad clip
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
        target_kl          = 0.01 if is_finetune else 0.02,   # tighter KL — smaller updates
        vloss_skip         = 1e3,
        lr_floor_frac      = 0.1 if is_finetune else 0.05,
        pretrained         = args.pretrained,
        freeze_trunk_steps = 5_000_000 if args.pretrained else 0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs",        type=int,  default=4096)
    p.add_argument("--total-steps",   type=int,  default=200_000_000)
    p.add_argument("--rollout-steps", type=int,  default=24)
    p.add_argument("--device",        type=str,  default="cuda",
                   choices=["cpu","cuda","mps"])
    p.add_argument("--run-dir",       type=str,  default="../../runs/go2_cpg_nav")
    p.add_argument("--headless",      action="store_true", default=False)
    p.add_argument("--resume",        type=str,  default=None)
    p.add_argument("--eval",          type=str,  default=None)
    p.add_argument("--pretrained",    type=str,  default=None,
                   help="Warm-start from go2_cpg_rl.py checkpoint (OBS_DIM=76)")
    p.add_argument("--vx",   type=float, default=0.5)
    p.add_argument("--vy",   type=float, default=0.0)
    p.add_argument("--vyaw", type=float, default=0.0)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.eval:
        evaluate(args.eval, command=(args.vx, args.vy, args.vyaw))
        return

    trainer = (PPOTrainer.load_checkpoint(args.resume)
               if args.resume else PPOTrainer(get_config(args)))
    trainer.train()


if __name__ == "__main__":
    main()
