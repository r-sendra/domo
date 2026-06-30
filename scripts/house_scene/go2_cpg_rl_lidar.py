"""
go2_avoidance.py
================
Clean obstacle avoidance via a small velocity-correction network
layered on top of a frozen CPG locomotion policy.

Architecture:
    LiDAR (36 sectors) → AvoidanceNet → velocity correction (Δvx, Δvy, Δvyaw)
    Base command + correction → frozen CPG policy → joint targets

The CPG policy is completely frozen — it just tracks velocity commands.
The avoidance network only needs to learn: given LiDAR readings,
what velocity correction keeps the robot away from obstacles?

This decoupling solves the core problems of joint training:
  - No reward dominance: avoidance reward is the ONLY signal
  - No CPG symmetry breaking: locomotion is pre-solved
  - Clean local optima: the action space is just 3 scalars

Training:
    python go2_avoidance.py \\
        --cpg-checkpoint ../../runs/go2_cpg/checkpoint_final.pt \\
        --n-envs 4096 --device cuda --headless

Evaluation:
    python go2_avoidance.py \\
        --eval ../../runs/go2_avoidance/checkpoint_final.pt \\
        --cpg-checkpoint ../../runs/go2_cpg/checkpoint_final.pt
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

# Import frozen CPG policy from go2_cpg_rl.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_cpg_rl import Go2CPGEnv, ActorCritic as CPGActorCritic, clean_state_dict, OBS_DIM as CPG_OBS_DIM


# ==========================================================================
#  Constants
# ==========================================================================

N_LIDAR_SECTORS  = 36
N_LIDAR_ELEV     = 5
LIDAR_MAX_RANGE  = 4.0
LIDAR_INTERVAL   = 5
LIDAR_POS_OFFSET = (0.0, 0.0, 0.35)

# Reward thresholds
LIDAR_COLLISION  = 0.25
LIDAR_DANGER     = 0.60
LIDAR_CAUTION    = 1.50
LIDAR_ANTICIPATE = 2.50

# Avoidance network obs/act dims
AVOID_OBS_DIM = N_LIDAR_SECTORS   # just LiDAR — nothing else needed
AVOID_ACT_DIM = 3                 # (Δvx, Δvy, Δvyaw)

# Correction limits — how much the avoidance net can override the base command
DELTA_VX_MAX   = 0.8    # can slow down or reverse up to 0.8 m/s
DELTA_VY_MAX   = 0.5    # lateral correction
DELTA_VYAW_MAX = 1.5    # yaw correction — generous so it can turn sharply

# Arena
N_OBS_CHAIRS   = 3
N_OBS_SOFAS    = 2
N_OBS_PILLARS  = 3
N_OBS_STEPS    = 2
N_OBS_BALLS    = 2
N_OBSTACLES    = N_OBS_CHAIRS*4 + N_OBS_SOFAS + N_OBS_PILLARS + N_OBS_STEPS + N_OBS_BALLS

ARENA_HALF        = 4.0
ARENA_WALL_HEIGHT = 0.6
ARENA_WALL_THICK  = 0.15
OBSTACLE_RING_MIN = 2.0
OBSTACLE_RING_MAX = 3.5

CHAIR_LEG_RADIUS = 0.03
CHAIR_LEG_HEIGHT = 0.45
CHAIR_LEG_SPREAD = 0.25

SOFA_WIDTH_RANGE  = (0.8,  1.6)
SOFA_DEPTH_RANGE  = (0.3,  0.5)
SOFA_HEIGHT_RANGE = (0.35, 0.50)

PILLAR_RADIUS_RANGE = (0.04, 0.10)
PILLAR_HEIGHT_RANGE = (0.60, 1.20)

STEP_SIZE_RANGE   = (0.20, 0.50)
STEP_HEIGHT_RANGE = (0.04, 0.15)

SPHERE_RAD_RANGE  = (0.06, 0.16)

# Base forward command — fixed during avoidance training
BASE_VX   = 0.6    # m/s forward
BASE_VY   = 0.0
BASE_VYAW = 0.0


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def clean_sd(sd):
    return {k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()}


# ==========================================================================
#  Avoidance Network — small MLP, obs=36 LiDAR, act=3 corrections
# ==========================================================================

class AvoidanceNet(nn.Module):
    """
    Small network that maps LiDAR sectors → velocity corrections.
    Kept simple intentionally — the task is simple once locomotion is solved.
    """

    def __init__(self, obs_dim=AVOID_OBS_DIM, act_dim=AVOID_ACT_DIM, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head  = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ELU(),
            nn.Linear(hidden//2, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ELU(),
            nn.Linear(hidden//2, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        # Initialise to near-zero outputs — start with minimal corrections
        nn.init.zeros_(self.actor_head[-1].bias)

    def forward(self, obs):
        h     = self.trunk(obs)
        mean  = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        std   = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist   = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        lp     = dist.log_prob(action).sum(-1)
        return action, lp, value

    def get_value(self, obs):
        return self.critic_head(self.trunk(obs)).squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def action_to_correction(self, raw_action):
        """Map tanh-squashed action to velocity correction."""
        t     = torch.tanh(raw_action)
        dvx   = DELTA_VX_MAX   * t[:, 0]
        dvy   = DELTA_VY_MAX   * t[:, 1]
        dvyaw = DELTA_VYAW_MAX * t[:, 2]
        return dvx, dvy, dvyaw

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
#  Environment — CPG inner loop + avoidance outer loop
# ==========================================================================

class Go2AvoidanceEnv:
    """
    Wraps Go2CPGEnv with:
      - Frozen CPG policy tracking base velocity command
      - LiDAR sensor for obstacle detection
      - Arena with randomised obstacles
      - Observation: 36 normalised LiDAR sector distances
      - Action: 3 velocity corrections (Δvx, Δvy, Δvyaw)
    """

    def __init__(self, cpg_checkpoint, n_envs=4096, dt=0.02,
                 max_episode_steps=1000, headless=True, device="cuda"):
        self.n_envs    = n_envs
        self.num_envs  = n_envs
        self.dt        = dt
        self.device    = torch.device(device)
        self.max_episode_steps = max_episode_steps

        # ── Load frozen CPG policy ─────────────────────────────────────────
        print(f"\n  Loading frozen CPG policy: {cpg_checkpoint}")
        ckpt   = torch.load(cpg_checkpoint, weights_only=False, map_location=device)
        sd     = clean_sd(ckpt["model_state"])
        obs_d  = sd["trunk.0.weight"].shape[1]
        act_d  = sd["log_std"].shape[0]
        hidden = ckpt["config"].get("hidden_size", 512)

        self.cpg_net = CPGActorCritic(obs_d, act_d, hidden).to(self.device)
        self.cpg_net.load_state_dict(sd)
        self.cpg_net.eval()
        # Freeze all parameters — never updated
        for p in self.cpg_net.parameters():
            p.requires_grad = False
        print(f"  CPG policy frozen: obs={obs_d} act={act_d} params={sum(p.numel() for p in self.cpg_net.parameters()):,}")

        # ── Build Genesis env ──────────────────────────────────────────────
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
            renderer    = gs.renderers.Rasterizer(),
            show_viewer = not headless,
        )

        # ── Ground ────────────────────────────────────────────────────────
        self.scene.add_entity(gs.morphs.Plane())

        # ── Arena walls ───────────────────────────────────────────────────
        s, t, h = ARENA_HALF, ARENA_WALL_THICK, ARENA_WALL_HEIGHT
        for wx, wy, sx, sy in [
            ( 0.0,  s,  2*s+2*t, t),
            ( 0.0, -s,  2*s+2*t, t),
            ( s,   0.0, t,       2*s),
            (-s,   0.0, t,       2*s),
        ]:
            self.scene.add_entity(gs.morphs.Box(
                size=(sx, sy, h), pos=(wx, wy, h/2), fixed=True))
        self.arena_term_dist = s + 0.5

        # ── Obstacles ─────────────────────────────────────────────────────
        self.obstacles = []
        leg_offsets = [
            ( CHAIR_LEG_SPREAD,  CHAIR_LEG_SPREAD),
            ( CHAIR_LEG_SPREAD, -CHAIR_LEG_SPREAD),
            (-CHAIR_LEG_SPREAD,  CHAIR_LEG_SPREAD),
            (-CHAIR_LEG_SPREAD, -CHAIR_LEG_SPREAD),
        ]
        for _ in range(N_OBS_CHAIRS):
            legs = [self.scene.add_entity(gs.morphs.Cylinder(
                radius=CHAIR_LEG_RADIUS, height=CHAIR_LEG_HEIGHT,
                pos=(99.0, 99.0, CHAIR_LEG_HEIGHT/2), fixed=True))
                for _ in leg_offsets]
            self.obstacles.append(("chair", legs, CHAIR_LEG_HEIGHT, leg_offsets))

        for i in range(N_OBS_SOFAS):
            t2 = i/max(N_OBS_SOFAS-1,1)
            w  = SOFA_WIDTH_RANGE[0]  + (SOFA_WIDTH_RANGE[1]-SOFA_WIDTH_RANGE[0])*t2
            d  = SOFA_DEPTH_RANGE[0]  + (SOFA_DEPTH_RANGE[1]-SOFA_DEPTH_RANGE[0])*t2
            hh = SOFA_HEIGHT_RANGE[0] + (SOFA_HEIGHT_RANGE[1]-SOFA_HEIGHT_RANGE[0])*t2
            e  = self.scene.add_entity(gs.morphs.Box(
                size=(w,d,hh), pos=(99.0,99.0,hh/2), fixed=True))
            self.obstacles.append(("sofa", e, hh, None))

        for i in range(N_OBS_PILLARS):
            t2 = i/max(N_OBS_PILLARS-1,1)
            r  = PILLAR_RADIUS_RANGE[0]+(PILLAR_RADIUS_RANGE[1]-PILLAR_RADIUS_RANGE[0])*t2
            hh = PILLAR_HEIGHT_RANGE[0]+(PILLAR_HEIGHT_RANGE[1]-PILLAR_HEIGHT_RANGE[0])*t2
            e  = self.scene.add_entity(gs.morphs.Cylinder(
                radius=r, height=hh, pos=(99.0,99.0,hh/2), fixed=True))
            self.obstacles.append(("pillar", e, hh, None))

        for i in range(N_OBS_STEPS):
            t2 = i/max(N_OBS_STEPS-1,1)
            ss = STEP_SIZE_RANGE[0]  +(STEP_SIZE_RANGE[1]-STEP_SIZE_RANGE[0])*t2
            hh = STEP_HEIGHT_RANGE[0]+(STEP_HEIGHT_RANGE[1]-STEP_HEIGHT_RANGE[0])*t2
            e  = self.scene.add_entity(gs.morphs.Box(
                size=(ss,ss,hh), pos=(99.0,99.0,hh/2), fixed=True))
            self.obstacles.append(("step", e, hh, None))

        for i in range(N_OBS_BALLS):
            t2 = i/max(N_OBS_BALLS-1,1)
            r  = SPHERE_RAD_RANGE[0]+(SPHERE_RAD_RANGE[1]-SPHERE_RAD_RANGE[0])*t2
            e  = self.scene.add_entity(gs.morphs.Sphere(
                radius=r, pos=(99.0,99.0,r), fixed=True))
            self.obstacles.append(("ball", e, r*2, None))

        # ── Robot ─────────────────────────────────────────────────────────
        self.base_init_pos  = torch.tensor([0.0, 0.0, 0.42], device=self.device)
        self.base_init_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        MOTOR_JOINT_NAMES = [
            "FR_hip_joint","FR_thigh_joint","FR_calf_joint",
            "FL_hip_joint","FL_thigh_joint","FL_calf_joint",
            "RR_hip_joint","RR_thigh_joint","RR_calf_joint",
            "RL_hip_joint","RL_thigh_joint","RL_calf_joint",
        ]
        DEFAULT_JOINT_ANGLES = {
            "FR_hip_joint":0.0,"FR_thigh_joint":0.8,"FR_calf_joint":-1.5,
            "FL_hip_joint":0.0,"FL_thigh_joint":0.8,"FL_calf_joint":-1.5,
            "RR_hip_joint":0.0,"RR_thigh_joint":1.0,"RR_calf_joint":-1.5,
            "RL_hip_joint":0.0,"RL_thigh_joint":1.0,"RL_calf_joint":-1.5,
        }

        self.robot = self.scene.add_entity(gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos =self.base_init_pos.cpu().numpy(),
            quat=self.base_init_quat.cpu().numpy(),
        ))

        # ── LiDAR (BEFORE build) ──────────────────────────────────────────
        self.front_cam = None
        if not headless:
            try:
                self.front_cam = self.scene.add_camera(
                    res=(640,360), pos=(0.3,0.0,0.1),
                    lookat=(1.0,0.0,0.1), fov=60, GUI=False)
                self._cam_frame = None
                self._pip_ready = False
            except Exception as e:
                print(f"  ⚠️  Camera: {e}")

        self.lidar = self.scene.add_sensor(gs.sensors.Lidar(
            pattern    = gs.sensors.SphericalPattern(
                fov      = (360.0, 50.0),
                n_points = (N_LIDAR_SECTORS, N_LIDAR_ELEV),
            ),
            entity_idx         = self.robot.idx,
            pos_offset         = LIDAR_POS_OFFSET,
            return_world_frame = True,
            draw_debug         = (not headless),
        ))

        # ── Build ─────────────────────────────────────────────────────────
        self.scene.build(n_envs=n_envs)

        # ── Motor setup ───────────────────────────────────────────────────
        self.motor_dofs = [self.robot.get_joint(n).dof_idx_local
                           for n in MOTOR_JOINT_NAMES]
        self.robot.set_dofs_kp([100.0]*12, self.motor_dofs)
        self.robot.set_dofs_kv([2.0  ]*12, self.motor_dofs)
        self.default_dof_pos = torch.tensor(
            [DEFAULT_JOINT_ANGLES[n] for n in MOTOR_JOINT_NAMES],
            device=self.device, dtype=gs.tc_float)

        from go2_cpg_rl import LEG_SIDE_SIGN, TROT_PHASE
        self.side_sign  = torch.tensor(LEG_SIDE_SIGN, device=self.device, dtype=gs.tc_float)
        self.trot_phase = torch.tensor(TROT_PHASE,    device=self.device, dtype=gs.tc_float)

        from go2_cpg_rl import A_CONV, CPG_DT, MU_MID, MU_HALF, OMG_MID, OMG_HALF, PSI_MAX
        self.A_CONV, self.CPG_DT = A_CONV, dt / max(1, round(dt/CPG_DT))
        self.n_cpg_substeps = max(1, round(dt/CPG_DT))
        self.cpg_dt = dt / self.n_cpg_substeps
        self.MU_MID, self.MU_HALF   = MU_MID, MU_HALF
        self.OMG_MID, self.OMG_HALF = OMG_MID, OMG_HALF
        self.PSI_MAX = PSI_MAX

        # Precompute Kuramoto phi_star
        self.phi_star = torch.tensor([
            [0.0,     math.pi, math.pi, 0.0    ],
            [math.pi, 0.0,     0.0,     math.pi],
            [math.pi, 0.0,     0.0,     math.pi],
            [0.0,     math.pi, math.pi, 0.0    ],
        ], device=self.device, dtype=gs.tc_float)

        print(f"\n{'='*60}")
        print(f"  Go2 Avoidance (2-layer)")
        print(f"{'='*60}")
        print(f"  Envs         : {n_envs}")
        print(f"  CPG policy   : FROZEN ({sum(p.numel() for p in self.cpg_net.parameters()):,} params)")
        print(f"  Avoid obs    : {AVOID_OBS_DIM} (LiDAR sectors only)")
        print(f"  Avoid act    : {AVOID_ACT_DIM} (Δvx, Δvy, Δvyaw)")
        print(f"  Obstacles    : {N_OBSTACLES} total")
        print(f"  Arena        : {2*ARENA_HALF:.0f}×{2*ARENA_HALF:.0f}m")
        print(f"{'='*60}\n")

        # ── State buffers ─────────────────────────────────────────────────
        N, f = n_envs, gs.tc_float
        self.base_lin_vel      = torch.zeros((N,3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N,3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N,3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor([0.,0.,-1.], device=self.device, dtype=f).repeat(N,1)
        self.base_pos          = torch.zeros((N,3), device=self.device, dtype=f)
        self.base_quat         = torch.zeros((N,4), device=self.device, dtype=f)
        self.base_euler        = torch.zeros((N,3), device=self.device, dtype=f)
        self.dof_pos           = torch.zeros((N,12), device=self.device, dtype=f)
        self.dof_vel           = torch.zeros((N,12), device=self.device, dtype=f)
        self.last_dof_vel      = torch.zeros((N,12), device=self.device, dtype=f)
        self.target_dof_pos    = self.default_dof_pos.unsqueeze(0).repeat(N,1).clone()
        self.applied_torque    = torch.zeros((N,12), device=self.device, dtype=f)
        self.cpg_actions       = torch.zeros((N,12), device=self.device, dtype=f)
        self.last_cpg_actions  = torch.zeros((N,12), device=self.device, dtype=f)
        self.cpg_r             = torch.ones( (N,4),  device=self.device, dtype=f)
        self.cpg_rdot          = torch.zeros((N,4),  device=self.device, dtype=f)
        self.cpg_theta         = torch.zeros((N,4),  device=self.device, dtype=f)
        self.cpg_phi           = torch.zeros((N,4),  device=self.device, dtype=f)
        self.foot_contacts     = torch.zeros((N,4),  device=self.device, dtype=f)

        # CPG obs buffer — built each step for the frozen CPG policy
        self.cpg_obs = torch.zeros((N, CPG_OBS_DIM), device=self.device, dtype=f)

        # Velocity command sent to CPG (base + correction from avoidance net)
        self.commands   = torch.zeros((N,3), device=self.device, dtype=f)
        # Previous step's correction — used for smoothness penalty (REASAN-style)
        self.last_correction = torch.zeros((N,3), device=self.device, dtype=f)

        # LiDAR
        self.lidar_sectors      = torch.full((N,N_LIDAR_SECTORS), LIDAR_MAX_RANGE,
                                              device=self.device, dtype=f)
        self._lidar_step        = 0

        # Episode tracking
        self.obs_buf            = torch.zeros((N, AVOID_OBS_DIM), device=self.device, dtype=f)
        self.rew_buf            = torch.zeros((N,), device=self.device, dtype=f)
        self.reset_buf          = torch.ones( (N,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,), device=self.device, dtype=gs.tc_int)
        self.episode_sums       = {
            "avoidance":         torch.zeros((N,), device=self.device, dtype=f),
            "survival":          torch.zeros((N,), device=self.device, dtype=f),
            "command_tracking":  torch.zeros((N,), device=self.device, dtype=f),
        }
        self.extras = {}
        self._cam_step = 0

    # ------------------------------------------------------------------
    # CPG inner loop helpers
    # ------------------------------------------------------------------

    def _build_cpg_obs(self):
        """Build the 76-dim observation the frozen CPG policy expects."""
        obs_scales = {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
        commands_scale = torch.tensor(
            [obs_scales["lin_vel"], obs_scales["lin_vel"], 0.25],
            device=self.device, dtype=gs.tc_float)
        self.cpg_obs = torch.cat([
            self.base_lin_vel * obs_scales["lin_vel"],
            self.base_ang_vel * obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * commands_scale,
            (self.dof_pos - self.default_dof_pos) * obs_scales["dof_pos"],
            self.dof_vel * obs_scales["dof_vel"],
            self.last_cpg_actions,
            self.foot_contacts,
            self.cpg_r, self.cpg_rdot,
            torch.cos(self.cpg_theta), torch.sin(self.cpg_theta),
            torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi),
        ], dim=-1)

    def _integrate_cpg(self, mu, omega_hz, psi):
        omega = 2.0 * math.pi * omega_hz
        a, dt_c = self.A_CONV, self.cpg_dt
        COUPLING_W = 2.0
        for _ in range(self.n_cpg_substeps):
            theta_i  = self.cpg_theta.unsqueeze(2)
            theta_j  = self.cpg_theta.unsqueeze(1)
            coupling = COUPLING_W * torch.sum(
                torch.sin(theta_j - theta_i - self.phi_star), dim=2)
            r_ddot        = a * (0.25*a*(mu - self.cpg_r) - self.cpg_rdot)
            self.cpg_rdot = self.cpg_rdot + r_ddot * dt_c
            self.cpg_r    = self.cpg_r    + self.cpg_rdot * dt_c
            self.cpg_theta = self.cpg_theta + (omega + coupling) * dt_c
            self.cpg_phi   = self.cpg_phi   + psi * dt_c
        self.cpg_theta = torch.remainder(self.cpg_theta, 2*math.pi)
        self.cpg_phi   = torch.remainder(self.cpg_phi,   2*math.pi)

    def _cpg_action_to_joints(self, raw_action):
        from go2_cpg_rl import (D_STEP, H_NOM, G_CLEAR, G_PEN)
        t        = torch.tanh(raw_action)
        mu       = self.MU_MID  + self.MU_HALF  * t[:, 0:4]
        omega_hz = self.OMG_MID + self.OMG_HALF * t[:, 4:8]
        psi      = self.PSI_MAX * t[:, 8:12]
        self._integrate_cpg(mu, omega_hz, psi)

        from go2_cpg_rl import leg_ik
        amp  = D_STEP * (self.cpg_r - 1.0)
        ct, st = torch.cos(self.cpg_theta), torch.sin(self.cpg_theta)
        cp, sp = torch.cos(self.cpg_phi),   torch.sin(self.cpg_phi)
        px = -amp * ct * cp
        py = self.side_sign * 0.0955 - amp * ct * sp
        z_clear = torch.where(st > 0, G_CLEAR*st, G_PEN*st)
        pz = -H_NOM + z_clear
        qh, qt, qc = leg_ik(px, py, pz, self.side_sign, 0.0955, 0.213, 0.213)
        self.target_dof_pos[:, 0::3] = qh
        self.target_dof_pos[:, 1::3] = qt
        self.target_dof_pos[:, 2::3] = qc

    def _update_foot_contacts(self):
        self.foot_contacts = (torch.sin(self.cpg_theta) < 0).float()

    def _update_lidar(self):
        raw_ = self.lidar.read().distances
        if raw_.dim() == 3:
            n_env, n_elev, n_horiz = raw_.shape
            if n_horiz == N_LIDAR_SECTORS:
                self.lidar_sectors[:] = raw_.min(dim=1).values
            else:
                raw = raw_.reshape(n_env, -1)
                n_raw = raw.shape[1]
                rps = n_raw // N_LIDAR_SECTORS
                n_trim = rps * N_LIDAR_SECTORS
                self.lidar_sectors[:] = raw[:,:n_trim].view(
                    n_env, N_LIDAR_SECTORS, rps).min(dim=2).values
        else:
            raw = raw_.reshape(self.n_envs, -1)
            n_raw = raw.shape[1]
            if n_raw >= N_LIDAR_SECTORS:
                rps = n_raw // N_LIDAR_SECTORS
                n_trim = rps * N_LIDAR_SECTORS
                self.lidar_sectors[:] = raw[:,:n_trim].view(
                    self.n_envs, N_LIDAR_SECTORS, rps).min(dim=2).values

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, avoid_action):
        """
        avoid_action: [N, 3] raw output from AvoidanceNet
        Converts to velocity correction, adds to base command,
        runs frozen CPG policy one step, steps physics.
        """
        # 1. Avoidance correction → final command
        t     = torch.tanh(avoid_action)
        dvx   = DELTA_VX_MAX   * t[:, 0]
        dvy   = DELTA_VY_MAX   * t[:, 1]
        dvyaw = DELTA_VYAW_MAX * t[:, 2]
        correction = torch.stack([dvx, dvy, dvyaw], dim=-1)   # [N,3]

        # Commands: base forward + avoidance correction
        self.commands[:, 0] = torch.clamp(BASE_VX   + dvx,   -1.0, 2.0)
        self.commands[:, 1] = torch.clamp(BASE_VY   + dvy,   -0.5, 0.5)
        self.commands[:, 2] = torch.clamp(BASE_VYAW + dvyaw, -1.5, 1.5)

        # 2. Build CPG obs and run frozen CPG policy
        self._build_cpg_obs()
        with torch.no_grad():
            cpg_action, _, _ = self.cpg_net.get_action(
                self.cpg_obs, deterministic=True)
        self.cpg_actions = cpg_action

        # 3. CPG action → joint targets
        self._cpg_action_to_joints(cpg_action)
        self.robot.control_dofs_position(self.target_dof_pos, self.motor_dofs)
        self.scene.step()

        self.episode_length_buf += 1
        self._lidar_step        += 1
        self._cam_step          += 1

        # 4. Update state
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
        self.applied_torque = (100.0*(self.target_dof_pos - self.dof_pos)
                               - 2.0*self.dof_vel)
        self._update_foot_contacts()
        self.last_cpg_actions[:] = self.cpg_actions

        # 5. LiDAR
        if self._lidar_step % LIDAR_INTERVAL == 0:
            self._update_lidar()

        # 6. Termination
        self.reset_buf  = self.episode_length_buf > self.max_episode_steps
        self.reset_buf |= torch.abs(self.base_euler[:,1]) > 1.0
        self.reset_buf |= torch.abs(self.base_euler[:,0]) > 1.0
        self.reset_buf |= self.base_pos[:,2] < 0.18
        self.reset_buf |= self.base_pos[:,0].abs() > self.arena_term_dist
        self.reset_buf |= self.base_pos[:,1].abs() > self.arena_term_dist
        # Collision termination (after first LiDAR update)
        if self._lidar_step >= LIDAR_INTERVAL:
            self.reset_buf |= self.lidar_sectors.min(dim=1).values < LIDAR_COLLISION

        self.extras["time_outs"] = torch.zeros_like(self.reset_buf,
                                                     dtype=gs.tc_float)
        self.extras["time_outs"][
            self.episode_length_buf > self.max_episode_steps] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 7. Reward — avoidance only, clean signal
        min_d = torch.clamp(
            self.lidar_sectors.min(dim=1).values, 0.0, LIDAR_MAX_RANGE)

        # Survival: +1 every step alive
        survival = torch.ones(self.n_envs, device=self.device, dtype=gs.tc_float)

        # Avoidance: 4-zone penalty
        anticipate = torch.where(
            (min_d >= LIDAR_CAUTION) & (min_d < LIDAR_ANTICIPATE),
            (LIDAR_ANTICIPATE - min_d) * 0.05, torch.zeros_like(min_d))
        caution = torch.where(
            (min_d >= LIDAR_DANGER) & (min_d < LIDAR_CAUTION),
            (LIDAR_CAUTION - min_d) * 0.5,    torch.zeros_like(min_d))
        danger = torch.where(
            (min_d >= LIDAR_COLLISION) & (min_d < LIDAR_DANGER),
            torch.square(LIDAR_DANGER - min_d) * 3.0, torch.zeros_like(min_d))
        collision = torch.where(
            min_d < LIDAR_COLLISION,
            torch.full_like(min_d, 2.0), torch.zeros_like(min_d))

        avoidance_penalty = -(anticipate + caution + danger + collision)

        # Smoothness penalty (REASAN-style) — discourages oscillating
        # corrections between consecutive steps.
        correction_rate = torch.sum(
            torch.square(correction - self.last_correction), dim=1)
        smoothness_penalty = -correction_rate * 0.05
        self.last_correction = correction.detach()

        # Command-tracking reward — rewards the net for outputting ZERO
        # correction (i.e. trusting the base command) when clear of
        # obstacles. This is what was missing: without it, any constant
        # nonzero correction satisfies the smoothness term just as well
        # as zero, so the net had no reason to ever return to baseline.
        #
        # Gated by clearance via `safety_margin` (1 when far from anything,
        # fading to 0 inside the caution zone) so this term never fights
        # the avoidance penalty when a turn is actually needed.
        safety_margin = torch.clamp(
            (min_d - LIDAR_CAUTION) / (LIDAR_ANTICIPATE - LIDAR_CAUTION),
            0.0, 1.0
        )
        correction_mag = torch.sum(torch.square(correction), dim=1)
        # exp(-mag) is 1.0 at zero correction, decays as correction grows
        command_tracking = safety_margin * torch.exp(-correction_mag * 2.0)

        self.rew_buf = (
            survival          * self.dt
            + avoidance_penalty  * self.dt * 5.0
            + smoothness_penalty * self.dt
            + command_tracking   * self.dt * 2.0   # strong — this is the fix
        )

        self.episode_sums["avoidance"]        += avoidance_penalty
        self.episode_sums["survival"]         += survival
        self.episode_sums["command_tracking"] += command_tracking

        # 8. Observation — normalised LiDAR sectors only
        self.obs_buf = torch.clamp(
            self.lidar_sectors / LIDAR_MAX_RANGE, 0.0, 1.0)

        # 9. Camera PiP
        if (self.front_cam is not None
                and self._cam_step % 10 == 0):
            self._render_pip()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def _render_pip(self):
        try:
            import cv2, math as _m
            pos  = self.robot.get_pos()[0].cpu().numpy()
            quat = self.robot.get_quat()[0].cpu().numpy()
            yaw  = float(quat_to_xyz(
                torch.tensor(quat).unsqueeze(0))[0,2].item())
            self.front_cam.set_pose(
                pos   =(pos[0]+0.25*_m.cos(yaw), pos[1]+0.25*_m.sin(yaw), pos[2]+0.15),
                lookat=(pos[0]+2.0*_m.cos(yaw),  pos[1]+2.0*_m.sin(yaw),  pos[2]+0.05))
            rgb, _, _, _ = self.front_cam.render(rgb=True)
            frame = np.array(rgb, dtype=np.uint8)
            bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pip   = cv2.resize(bgr, (320,180))
            pip   = cv2.copyMakeBorder(pip,2,2,2,2,
                                       cv2.BORDER_CONSTANT,value=(0,220,0))
            if not hasattr(self,'_pip_ready'):
                cv2.namedWindow("pip", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("pip", 324, 184)
                cv2.moveWindow("pip", 950, 10)
                self._pip_ready = True
            cv2.imshow("pip", pip)
            cv2.pollKey()
        except Exception:
            pass

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True, envs_idx=envs_idx)

        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1,-1)
        self.robot.set_pos( self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.cpg_r[envs_idx]     = 1.0
        self.cpg_rdot[envs_idx]  = 0.0
        self.cpg_theta[envs_idx] = self.trot_phase
        self.cpg_phi[envs_idx]   = self.trot_phase * 0.1

        self.lidar_sectors[envs_idx] = LIDAR_MAX_RANGE
        self._randomise_obstacles(envs_idx)

        self.last_cpg_actions[envs_idx]   = 0.0
        self.last_correction[envs_idx]    = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        for k in self.episode_sums:
            self.episode_sums[k][envs_idx] = 0.0

    def _randomise_obstacles(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        spawn = self.base_init_pos
        for obs_type, entity, height, meta in self.obstacles:
            angles = gs_rand_float(0.0, 2*math.pi, (n,), self.device)
            radii  = gs_rand_float(OBSTACLE_RING_MIN, OBSTACLE_RING_MAX,
                                   (n,), self.device)
            cx = spawn[0] + radii * torch.cos(angles)
            cy = spawn[1] + radii * torch.sin(angles)
            if obs_type == "chair":
                yaw = gs_rand_float(0.0, 2*math.pi, (n,), self.device)
                for leg_e, (dx, dy) in zip(entity, meta):
                    lx = cx + dx*torch.cos(yaw) - dy*torch.sin(yaw)
                    ly = cy + dx*torch.sin(yaw) + dy*torch.cos(yaw)
                    lz = torch.full((n,), height/2, device=self.device)
                    leg_e.set_pos(torch.stack([lx,ly,lz],dim=-1),
                                  envs_idx=envs_idx)
            else:
                z   = torch.full((n,), height/2, device=self.device)
                pos = torch.stack([cx,cy,z],dim=-1)
                entity.set_pos(pos, envs_idx=envs_idx)


# ==========================================================================
#  RolloutBuffer (minimal)
# ==========================================================================

class RolloutBuffer:
    def __init__(self, T, N, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = T, N, device, 0
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs        = z(T,N,obs_dim)
        self.actions    = z(T,N,act_dim)
        self.log_probs  = z(T,N)
        self.values     = z(T,N)
        self.rewards    = z(T,N)
        self.dones      = z(T,N)
        self.advantages = z(T,N)
        self.returns    = z(T,N)

    def store_step(self, obs, actions, lp, val):
        t = self.ptr
        self.obs[t], self.actions[t] = obs.detach(), actions.detach()
        self.log_probs[t], self.values[t] = lp.detach(), val.detach()

    def store_outcome(self, rew, done):
        t = self.ptr
        self.rewards[t] = rew.detach().float()
        self.dones[t]   = done.detach().float()
        self.ptr += 1

    def compute_gae(self, last_val, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            nv    = last_val if t == self.T-1 else self.values[t+1]
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

        self.env = Go2AvoidanceEnv(
            cpg_checkpoint    = cfg["cpg_checkpoint"],
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
            device            = cfg["device"],
        )

        self.net = AvoidanceNet(
            obs_dim = AVOID_OBS_DIM,
            act_dim = AVOID_ACT_DIM,
            hidden  = cfg["hidden_size"],
        ).to(self.device)
        print(f"  AvoidanceNet params: {self.net.num_parameters:,}\n")

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg["lr"], eps=1e-5)
        self.buf = RolloutBuffer(
            cfg["rollout_steps"], cfg["n_envs"],
            AVOID_OBS_DIM, AVOID_ACT_DIM, self.device)

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()
        self.ep_returns, self.ep_lengths = [], []
        self._env_ret = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len = torch.zeros(cfg["n_envs"], device=self.device,
                                    dtype=torch.int32)

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()

        print(f"{'='*55}")
        print(f"  Avoidance PPO | {cfg['total_steps']:,} steps | {self.device}")
        print(f"{'='*55}\n")

        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates         = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates+1):
            frac = max(1.0 - self.global_step/max(cfg["total_steps"],1),
                       cfg["lr_floor_frac"])
            lr = cfg["lr"] * frac
            for g in self.opt.param_groups:
                g["lr"] = lr

            obs     = self._collect_rollout(obs)
            m       = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                sps  = self.global_step / (time.time()-self.start_time)
                mret = float(np.mean(self.ep_returns[-50:])) if self.ep_returns else 0.0
                mlen = float(np.mean(self.ep_lengths[-50:])) if self.ep_lengths else 0.0
                print(f"  step {self.global_step:>10,} | "
                      f"ret {mret:>7.3f} | len {mlen:>5.0f} | "
                      f"ploss {m['policy_loss']:>7.4f} | "
                      f"vloss {m['value_loss']:>7.4f} | "
                      f"clip {m['clip_frac']:>4.2f} | "
                      f"lr {lr:.1e} | {sps:>7,.0f} sps")
                for k, v in {
                    "train/mean_return": mret, "train/mean_ep_len": mlen,
                    "loss/policy": m["policy_loss"], "loss/value": m["value_loss"],
                    "loss/entropy": m["entropy"], "train/sps": sps,
                }.items():
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
                action, lp, val = self.net.get_action(obs)
                self.buf.store_step(obs, action, lp, val)
                next_obs, _, rew, reset_buf, _ = self.env.step(action)
                self.buf.store_outcome(rew, reset_buf.float())
                self._env_ret += rew
                self._env_len += 1
                done_idx = reset_buf.nonzero(as_tuple=False).flatten()
                for idx in done_idx:
                    self.ep_returns.append(float(self._env_ret[idx]))
                    self.ep_lengths.append(int(self._env_len[idx]))
                self._env_ret[done_idx] = 0.0
                self._env_len[done_idx] = 0
                obs = next_obs
            last_val = self.net.get_value(obs)
            self.buf.compute_gae(last_val,
                                 self.cfg["gamma"], self.cfg["lam"])
        return obs

    def _ppo_update(self):
        self.net.train()
        cfg = self.cfg
        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()

        if not (torch.isfinite(adv_f).all() and torch.isfinite(ret_f).all()):
            return {"policy_loss":0.,"value_loss":0.,
                    "entropy":0.,"clip_frac":0.}

        backup = {k:v.detach().clone() for k,v in self.net.state_dict().items()}
        adv_f  = ((adv_f-adv_f.mean())/(adv_f.std()+1e-8)).clamp(-10.,10.)
        total  = obs_f.shape[0]
        m      = {"policy_loss":[],"value_loss":[],"entropy":[],"clip_frac":[]}
        bad    = False

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total, device=self.device)
            epoch_kl = []
            for start in range(0, total, cfg["minibatch_size"]):
                mb = idx[start:start+cfg["minibatch_size"]]
                new_lp, ent, val = self.net.evaluate(obs_f[mb], act_f[mb])
                logratio = new_lp - lp_f[mb]
                ratio    = logratio.exp()
                with torch.no_grad():
                    akl = ((ratio-1.)-logratio).mean().item()
                epoch_kl.append(akl)

                surr1 = ratio * adv_f[mb]
                surr2 = ratio.clamp(1-cfg["clip_eps"],1+cfg["clip_eps"])*adv_f[mb]
                pl    = -torch.min(surr1, surr2).mean()
                vc    = val_f[mb]+(val-val_f[mb]).clamp(-cfg["clip_eps"],cfg["clip_eps"])
                vl    = torch.max((val-ret_f[mb]).pow(2),(vc-ret_f[mb]).pow(2)).mean()
                loss  = pl + cfg["vf_coef"]*vl - cfg["ent_coef"]*ent.mean()

                if not torch.isfinite(loss):
                    bad = True; break
                self.opt.zero_grad(); loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg["max_grad_norm"])
                if not torch.isfinite(gnorm):
                    bad = True; break
                self.opt.step()
                m["policy_loss"].append(pl.item())
                m["value_loss"].append(vl.item())
                m["entropy"].append(ent.mean().item())
                m["clip_frac"].append(
                    ((ratio-1.).abs()>cfg["clip_eps"]).float().mean().item())
            if bad: break
            if np.mean(epoch_kl) > 1.5*cfg.get("target_kl",0.02): break

        if bad:
            self.net.load_state_dict(backup)
        return {k: float(np.mean(v)) if v else 0. for k,v in m.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_{tag}" if tag \
               else f"checkpoint_step_{self.global_step:09d}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step":        self.global_step,
            "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(),
            "config":      self.cfg,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-50:]) if self.ep_returns else 0.,
                "mean_length": np.mean(self.ep_lengths[-50:]) if self.ep_lengths else 0.,
            },
        }, path)
        print(f"  [ckpt] {path}")

    @classmethod
    def load_checkpoint(cls, path, cpg_override=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(path, weights_only=False, map_location=device)
        cfg    = ckpt["config"]
        if cpg_override:
            cfg["cpg_checkpoint"] = cpg_override
        trainer = cls(cfg)
        trainer.net.load_state_dict(ckpt["model_state"])
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(f"  Resumed step={trainer.global_step:,}")
        return trainer


# ==========================================================================
#  Evaluation
# ==========================================================================

def evaluate(checkpoint_path, cpg_checkpoint, n_episodes=5, command_vx=0.6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg    = ckpt["config"]
    if cpg_checkpoint:
        cfg["cpg_checkpoint"] = cpg_checkpoint

    env = Go2AvoidanceEnv(
        cpg_checkpoint    = cfg["cpg_checkpoint"],
        n_envs            = 1,
        headless          = False,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
        device            = device,
    )
    # Override base forward speed for eval — use globals() so this works
    # regardless of what filename this script is saved/renamed as
    globals()["BASE_VX"] = command_vx

    net = AvoidanceNet()
    net.load_state_dict(ckpt["model_state"])
    net.eval().to(device)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len = 0.0, 0
        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, rew, reset_buf, _ = env.step(act)
            ep_ret += rew[0].item()
            ep_len += 1
            done    = reset_buf.bool()
            if ep_len % 50 == 0:
                min_d = env.lidar_sectors[0].min().item()
                cmd   = env.commands[0].cpu().numpy()
                print(f"    step {ep_len:4d}  "
                      f"min_lidar={min_d:.2f}m  "
                      f"vx={env.base_lin_vel[0,0].item():+.2f}  "
                      f"cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})")
        print(f"  Episode {ep+1} | return={ep_ret:.2f} | length={ep_len}\n")


# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer = args.n_envs * args.rollout_steps
    return dict(
        cpg_checkpoint     = args.cpg_checkpoint,
        n_envs             = args.n_envs,
        dt                 = 0.02,
        max_episode_steps  = 1000,
        headless           = args.headless,
        hidden_size        = 128,
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = max(total_buffer // 4, 256),
        n_epochs           = 5,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 3e-4,
        vf_coef            = 1.0,
        ent_coef           = 0.02,
        max_grad_norm      = 1.0,
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
        target_kl          = 0.02,
        lr_floor_frac      = 0.05,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpg-checkpoint", required=True,
                   help="Path to trained CPG locomotion checkpoint")
    p.add_argument("--n-envs",        type=int,   default=4096)
    p.add_argument("--total-steps",   type=int,   default=100_000_000)
    p.add_argument("--rollout-steps", type=int,   default=24)
    p.add_argument("--device",        type=str,   default="cuda",
                   choices=["cpu","cuda","mps"])
    p.add_argument("--run-dir",       type=str,   default="../../runs/go2_avoidance")
    p.add_argument("--headless",      action="store_true", default=False)
    p.add_argument("--resume",        type=str,   default=None)
    p.add_argument("--eval",          type=str,   default=None)
    p.add_argument("--vx",            type=float, default=0.6)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.eval:
        evaluate(args.eval, args.cpg_checkpoint,
                 command_vx=args.vx)
        return

    if args.resume:
        trainer = PPOTrainer.load_checkpoint(
            args.resume, cpg_override=args.cpg_checkpoint)
    else:
        trainer = PPOTrainer(get_config(args))

    trainer.train()


if __name__ == "__main__":
    main()
