"""
go2_lidar_nav.py
================
Self-contained RL experiment: Go2 locomotion with LiDAR-based
obstacle avoidance on flat terrain with random obstacles.

Extends the working Go2WalkEnv (genesis official example style) with:
  1. LiDAR sensor    — gs.sensors.Lidar with default SphericalPattern
  2. Sector obs      — 8192 raw rays → 36 sector minimums (81-dim total)
  3. Obstacles       — random cylinders placed in the scene each episode
  4. Extra reward    — penalty for approaching obstacles too closely

LiDAR pre-processing (sector aggregation):
  8192 raw rays → min per 36 horizontal sectors → 36 values → [0,1]
  This is the approach used in Omni-Perception (HKUST 2025) and is
  the standard for locomotion + obstacle avoidance RL.

Everything else (PPO, RolloutBuffer, ActorCritic, checkpointing) is
identical to main.py so checkpoints are cross-compatible.

Usage:
    # Train on H200
    python go2_lidar_nav.py --n-envs 4096 --device cuda --headless

    # Evaluate on Mac (loads checkpoint, opens viewer)
    python go2_lidar_nav.py --eval ../../runs/go2_lidar/checkpoint_final.pt

    # Resume training
    python go2_lidar_nav.py --resume ../../runs/go2_lidar/checkpoint_step_050000000.pt
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
#  Constants
# ==========================================================================

MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint":   0.0, "FR_thigh_joint":  0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint":   0.0, "FL_thigh_joint":  0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint":   0.0, "RR_thigh_joint":  1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint":   0.0, "RL_thigh_joint":  1.0, "RL_calf_joint": -1.5,
}

BASE_INIT_POS  = [3.0, -2.0, 0.42]
BASE_INIT_QUAT = [0.0, 0.0, 0.0, 1.0]   # [x,y,z,w] identity

# LiDAR — default SphericalPattern (n_points=(64,128) = 8192 rays on GPU)
# On CPU (Mac) this is slow; training on H200 GPU is fast
LIDAR_MAX_RANGE    = 5.0    # metres
LIDAR_DANGER_ZONE  = 0.8    # metres — penalise if closer than this
LIDAR_CAUTION_ZONE = 1.5    # metres — smaller penalty

# Sector aggregation: 8192 rays → 36 sector minimums
# Each sector covers 360°/36 = 10° horizontal slice
# Min distance per sector is the relevant signal for avoidance
N_SECTORS = 36
PROP_DIM  = 45               # proprioception dims (unchanged from walking env)
OBS_DIM   = PROP_DIM + N_SECTORS   # 45 + 36 = 81

# Obstacle configuration
N_OBSTACLES        = 8      # cylinders per environment
OBSTACLE_RING_MIN  = 1.0    # metres from spawn — inner exclusion zone
OBSTACLE_RING_MAX  = 4.0    # metres from spawn — outer boundary
OBSTACLE_HEIGHT    = 1.2    # metres
OBSTACLE_RADIUS    = 0.25   # metres


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# ==========================================================================
#  Environment
# ==========================================================================

class Go2LidarNavEnv:
    """
    Go2 locomotion + LiDAR obstacle avoidance environment.

    Observation: [proprioception (45)] + [LiDAR distances (N_lidar)]
    Action:      joint position offsets (12)
    Reward:      velocity tracking + collision avoidance penalty
    """

    ACT_DIM = 12

    def __init__(
        self,
        n_envs:            int   = 4096,
        dt:                float = 0.02,
        max_episode_steps: int   = 1000,
        headless:          bool  = True,
        device:            str   = "cuda",
    ):
        self.n_envs            = n_envs
        self.num_envs          = n_envs
        self.dt                = dt
        self.max_episode_steps = max_episode_steps
        self.device            = torch.device(device)

        self.simulate_action_latency = True

        # ── Config ────────────────────────────────────────────────────────
        self.env_cfg = {
            "num_actions":   12,
            "base_init_pos":  BASE_INIT_POS,
            "base_init_quat": BASE_INIT_QUAT,
            "episode_length_s": max_episode_steps * dt,
            "resampling_time_s": 4.0,
            "action_scale":  0.25,
            "clip_actions":  100.0,
            "kp": 20.0,
            "kd":  0.5,
            "dof_names": MOTOR_JOINT_NAMES,
            "default_joint_angles": DEFAULT_JOINT_ANGLES,
            "termination_if_pitch_greater_than": 1.0,
            "termination_if_roll_greater_than":  1.0,
        }
        self.obs_cfg = {
            "obs_scales": {
                "lin_vel": 2.0, "ang_vel": 0.25,
                "dof_pos": 1.0, "dof_vel": 0.05,
            },
        }
        self.reward_cfg = {
            "tracking_sigma":     0.25,
            "base_height_target": 0.34,
            "reward_scales": {
                "tracking_lin_vel":   1.0,
                "tracking_ang_vel":   0.2,
                "lin_vel_z":         -1.0,
                "base_height":       -50.0,
                "action_rate":       -0.005,
                "similar_to_default":-0.1,
                "obstacle_avoidance":-2.0,   # NEW: collision avoidance
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.5, 0.5],
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range":   [0.0, 0.0],
        }

        self.obs_scales     = self.obs_cfg["obs_scales"]
        self.reward_scales  = {
            k: v * dt
            for k, v in self.reward_cfg["reward_scales"].items()
        }
        self.max_episode_length = math.ceil(
            self.env_cfg["episode_length_s"] / self.dt
        )
        self.num_commands = self.command_cfg["num_commands"]

        # ── Genesis init ──────────────────────────────────────────────────
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(
            backend      = backend,
            precision    = "32",
            logging_level= "warning",
            performance_mode = True,
        )

        self.scene = gs.Scene(
            sim_options    = gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options = gs.options.ViewerOptions(
                max_FPS      = int(0.5 / self.dt),
                camera_pos   = (3.0, 0.0, 3.0),
                camera_lookat= (0.0, 0.0, 0.5),
                camera_fov   = 50,
            ),
            vis_options    = gs.options.VisOptions(n_rendered_envs=1),
            rigid_options  = gs.options.RigidOptions(
                dt                = self.dt,
                constraint_solver = gs.constraint_solver.Newton,
                enable_collision  = True,
                enable_joint_limit= True,
            ),
            show_viewer = not headless,
        )

        # ── Ground ────────────────────────────────────────────────────────
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # ── Obstacles ─────────────────────────────────────────────────────
        # Static cylinders placed in the scene.
        # Position is randomised at reset via set_pos().
        self.obstacles = []
        for _ in range(N_OBSTACLES):
            obs_entity = self.scene.add_entity(
                gs.morphs.Cylinder(
                    height = OBSTACLE_HEIGHT,
                    radius = OBSTACLE_RADIUS,
                    pos    = (5.0, 5.0, OBSTACLE_HEIGHT / 2),  # off-scene initially
                    fixed  = True,
                )
            )
            self.obstacles.append(obs_entity)

        # ── Robot ─────────────────────────────────────────────────────────
        self.base_init_pos  = torch.tensor(BASE_INIT_POS,  device=self.device)
        self.base_init_quat = torch.tensor(BASE_INIT_QUAT, device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = "urdf/go2/urdf/go2.urdf",
                pos  = self.base_init_pos.cpu().numpy(),
                quat = self.base_init_quat.cpu().numpy(),
            )
        )

        # ── LiDAR sensor ──────────────────────────────────────────────────
        # Default SphericalPattern — let Genesis choose the resolution.
        # On GPU (H200) this is vectorised and fast.
        # On CPU (Mac) it is slower — reduce n_points if needed.
        self.lidar = self.scene.add_sensor(
            gs.sensors.Lidar(
                pattern = gs.sensors.SphericalPattern(
                    # Default: n_points=(64, 128) = 8192 rays
                    # Reduce for CPU: n_points=(16, 32) = 512 rays
                ),
                entity_idx         = self.robot.idx,
                pos_offset         = (0.0, 0.0, 0.15),
                return_world_frame = False,   # body frame for RL
                draw_debug         = (not headless),
            )
        )

        # ── Build ─────────────────────────────────────────────────────────
        self.scene.build(n_envs=n_envs)

        # Determine LiDAR output dimension from a dummy read after build
        # SphericalPattern default: 64 horizontal × 128 vertical = 8192
        _dummy           = self.lidar.read()
        _dummy_flat      = _dummy.distances.reshape(_dummy.distances.shape[0], -1)
        self.n_lidar_raw     = _dummy_flat.shape[-1]
        self.rays_per_sector = self.n_lidar_raw // N_SECTORS
        self.n_trim          = self.rays_per_sector * N_SECTORS
        self.OBS_DIM         = OBS_DIM   # 45 + 36 = 81
        print(f"  LiDAR raw rays : {self.n_lidar_raw}")
        print(f"  Rays/sector    : {self.rays_per_sector}  (trimmed to {self.n_trim})")
        print(f"  LiDAR sectors  : {N_SECTORS}  (min per sector)")
        print(f"  Total obs dim  : {self.OBS_DIM}")

        # ── Motor DOFs ────────────────────────────────────────────────────
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.ACT_DIM, self.motor_dofs
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.ACT_DIM, self.motor_dofs
        )

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][n]
             for n in self.env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        # ── Reward registry ───────────────────────────────────────────────
        self.reward_functions = {}
        self.episode_sums     = {}
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name]     = torch.zeros(
                (n_envs,), device=self.device, dtype=gs.tc_float
            )

        # ── State buffers ─────────────────────────────────────────────────
        N, f = n_envs, gs.tc_float
        self.base_lin_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=f
        ).repeat(N, 1)

        self.obs_buf            = torch.zeros((N, self.OBS_DIM), device=self.device, dtype=f)
        self.rew_buf            = torch.zeros((N,),              device=self.device, dtype=f)
        self.reset_buf          = torch.ones( (N,),              device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,),              device=self.device, dtype=gs.tc_int)

        self.commands       = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"],
             self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"]],
            device=self.device, dtype=f,
        )

        self.actions      = torch.zeros((N, self.ACT_DIM), device=self.device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos     = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat    = torch.zeros((N, 4), device=self.device, dtype=f)

        # LiDAR sector buffer — shape [N, N_SECTORS]
        # Aggregated from raw rays: min distance per 36 horizontal sectors
        self.lidar_sectors = torch.zeros(
            (N, N_SECTORS), device=self.device, dtype=f
        )

        self.extras = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
             self.env_cfg["clip_actions"],
        )

        exec_actions   = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # ── Update state ──────────────────────────────────────────────────
        self.episode_length_buf += 1
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler   = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat              = inv_quat(self.base_quat)
        self.base_lin_vel[:]       = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]       = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:]  = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:]            = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:]            = self.robot.get_dofs_velocity(self.motor_dofs)

        # ── LiDAR read + sector aggregation ──────────────────────────────
        # scene.step() already computed all rays — this is just a memory read
        lidar_data = self.lidar.read()
        raw_       = lidar_data.distances   # [N, channels, horizontal] or [N, n_rays]

        # One-time debug print to show actual runtime shape
        if not hasattr(self, '_lidar_shape_printed'):
            print(f"  [LiDAR runtime] distances.shape = {raw_.shape}")
            self._lidar_shape_printed = True

        # Flatten to [N, total_rays] regardless of whether Genesis returns
        # a 2D [N, rays] or 3D [N, channels, horizontal] tensor
        raw = raw_.reshape(raw_.shape[0], -1)   # [N, total_rays]

        # Aggregate raw rays into N_SECTORS sector minimums.
        # Handles any ray count gracefully — no assumption about divisibility.
        batch, n_raw = raw.shape
        if n_raw >= N_SECTORS:
            # Enough rays — reshape into N_SECTORS groups and take min
            rays_ps = n_raw // N_SECTORS
            n_trim  = rays_ps * N_SECTORS
            self.lidar_sectors[:] = raw[
                :, :n_trim
            ].view(batch, N_SECTORS, rays_ps).min(dim=2).values
        else:
            # Fewer rays than sectors — repeat-pad to fill sectors
            # Each sector gets the same small set of rays
            padded = raw.repeat(1, math.ceil(N_SECTORS / n_raw))[:, :N_SECTORS]
            self.lidar_sectors[:] = padded

        # ── Resample commands ─────────────────────────────────────────────
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (
            (self.episode_length_buf % resample_every == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)

        # ── Termination ───────────────────────────────────────────────────
        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False).flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # ── Reward ────────────────────────────────────────────────────────
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ── Observation ───────────────────────────────────────────────────
        # Proprioception (45) + sector minimums normalised to [0,1] (36)
        lidar_norm = torch.clamp(
            self.lidar_sectors / LIDAR_MAX_RANGE, 0.0, 1.0
        )
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],                  # 3
                self.projected_gravity,                                            # 3
                self.commands * self.commands_scale,                               # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                         # 12
                self.actions,                                                      # 12
                lidar_norm,                                                        # 36 sectors
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # Reset joints
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position      = self.dof_pos[envs_idx],
            dofs_idx_local= self.motor_dofs,
            zero_velocity = True,
            envs_idx      = envs_idx,
        )

        # Reset base pose
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx],  zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Randomise obstacle positions for reset envs
        self._randomise_obstacles(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (n,), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (n,), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"],   (n,), self.device
        )

    def _randomise_obstacles(self, envs_idx):
        """
        Place obstacles at random positions around each reset env's spawn.
        Each env in envs_idx gets a new random obstacle layout.
        """
        if len(envs_idx) == 0:
            return
        for obs_entity in self.obstacles:
            # Random angle and radius in the annulus
            angles = torch.rand(len(envs_idx), device=self.device) * 2 * math.pi
            radii  = (
                OBSTACLE_RING_MIN
                + torch.rand(len(envs_idx), device=self.device)
                * (OBSTACLE_RING_MAX - OBSTACLE_RING_MIN)
            )
            x = radii * torch.cos(angles)
            y = radii * torch.sin(angles)
            z = torch.full((len(envs_idx),), OBSTACLE_HEIGHT / 2, device=self.device)
            pos = torch.stack([x, y, z], dim=-1)
            obs_entity.set_pos(pos, envs_idx=envs_idx)

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        return torch.square(
            self.base_pos[:, 2] - self.reward_cfg["base_height_target"]
        )

    def _reward_action_rate(self):
        return torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )

    def _reward_similar_to_default(self):
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        )

    def _reward_obstacle_avoidance(self):
        """
        Penalty based on minimum LiDAR distance.
        In danger zone (<LIDAR_DANGER_ZONE): strong quadratic penalty
        In caution zone (<LIDAR_CAUTION_ZONE): weaker linear penalty
        Beyond caution zone: zero penalty
        """
        min_dist = self.lidar_sectors.min(dim=1).values   # [N]
        min_dist = torch.clamp(min_dist, 0.0, LIDAR_MAX_RANGE)

        # Danger zone penalty (quadratic — grows sharply near obstacles)
        danger_pen = torch.where(
            min_dist < LIDAR_DANGER_ZONE,
            torch.square(min_dist - LIDAR_DANGER_ZONE),
            torch.zeros_like(min_dist),
        )

        # Caution zone penalty (linear — gentle warning further away)
        caution_pen = torch.where(
            (min_dist >= LIDAR_DANGER_ZONE) & (min_dist < LIDAR_CAUTION_ZONE),
            (LIDAR_CAUTION_ZONE - min_dist) * 0.2,
            torch.zeros_like(min_dist),
        )

        return danger_pen + caution_pen


# ==========================================================================
#  ActorCritic (identical to main.py — cross-compatible checkpoints)
# ==========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
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
#  RolloutBuffer (identical to main.py)
# ==========================================================================

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = rollout_steps, n_envs, device, 0

        def buf(*shape):
            return torch.zeros(*shape, device=device)

        self.obs        = buf(rollout_steps, n_envs, obs_dim)
        self.actions    = buf(rollout_steps, n_envs, act_dim)
        self.log_probs  = buf(rollout_steps, n_envs)
        self.values     = buf(rollout_steps, n_envs)
        self.rewards    = buf(rollout_steps, n_envs)
        self.dones      = buf(rollout_steps, n_envs)
        self.advantages = buf(rollout_steps, n_envs)
        self.returns    = buf(rollout_steps, n_envs)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t]       = obs.detach()
        self.actions[t]   = actions.detach()
        self.log_probs[t] = log_probs.detach()
        self.values[t]    = values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t] = rewards.detach().float()
        self.dones[t]   = dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask     = 1.0 - self.dones[t]
            delta    = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae      = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        T, N = self.T, self.N
        return (
            self.obs.view(T*N, -1),
            self.actions.view(T*N, -1),
            self.log_probs.view(T*N),
            self.advantages.view(T*N),
            self.returns.view(T*N),
            self.values.view(T*N),
        )


# ==========================================================================
#  PPO Trainer
# ==========================================================================

class PPOTrainer:

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = cfg["device"]

        self.env = Go2LidarNavEnv(
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
            device            = cfg["device"],
        )

        obs_dim = self.env.OBS_DIM
        self.net = ActorCritic(
            obs_dim = obs_dim,
            act_dim = Go2LidarNavEnv.ACT_DIM,
            hidden  = cfg["hidden_size"],
        ).to(self.device)
        print(f"  Network params : {self.net.num_parameters:,}")
        print(f"  Obs dim        : {obs_dim}  (45 proprioception + {N_SECTORS} LiDAR sectors)")

        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=cfg["lr"], eps=1e-5
        )
        self.buf = RolloutBuffer(
            rollout_steps = cfg["rollout_steps"],
            n_envs        = cfg["n_envs"],
            obs_dim       = obs_dim,
            act_dim       = Go2LidarNavEnv.ACT_DIM,
            device        = self.device,
        )

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()
        self.ep_returns  = []
        self.ep_lengths  = []
        self._env_ret    = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len    = torch.zeros(cfg["n_envs"], device=self.device, dtype=torch.int32)

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()

        print(f"\n{'='*55}")
        print(f"  Go2 LiDAR Navigation PPO Training")
        print(f"{'='*55}")
        print(f"  Envs          : {cfg['n_envs']}")
        print(f"  Obstacles/env : {N_OBSTACLES}")
        print(f"  Total steps   : {cfg['total_steps']:,}")
        print(f"  Device        : {self.device}")
        print(f"  Run dir       : {cfg['run_dir']}")
        print(f"{'='*55}\n")

        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates         = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates + 1):
            obs     = self._collect_rollout(obs)
            metrics = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                elapsed   = time.time() - self.start_time
                sps       = self.global_step / elapsed
                mean_ret  = float(np.mean(self.ep_returns[-20:])) if self.ep_returns else 0.0
                mean_len  = float(np.mean(self.ep_lengths[-20:])) if self.ep_lengths else 0.0

                print(
                    f"  step {self.global_step:>10,} | "
                    f"ret {mean_ret:>7.3f} | "
                    f"len {mean_len:>5.0f} | "
                    f"ploss {metrics['policy_loss']:>7.4f} | "
                    f"vloss {metrics['value_loss']:>7.4f} | "
                    f"clip {metrics['clip_frac']:>4.2f} | "
                    f"{sps:>7,.0f} sps"
                )
                self.writer.add_scalar("train/mean_return",   mean_ret,               self.global_step)
                self.writer.add_scalar("train/mean_ep_len",   mean_len,               self.global_step)
                self.writer.add_scalar("loss/policy",         metrics["policy_loss"], self.global_step)
                self.writer.add_scalar("loss/value",          metrics["value_loss"],  self.global_step)
                self.writer.add_scalar("loss/entropy",        metrics["entropy"],     self.global_step)
                self.writer.add_scalar("train/clip_fraction", metrics["clip_frac"],   self.global_step)
                self.writer.add_scalar("train/sps",           sps,                    self.global_step)

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

                next_obs, _, reward, reset_buf, extras = self.env.step(action)
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
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        total   = obs_f.shape[0]
        metrics = {"policy_loss":[], "value_loss":[], "entropy":[], "clip_frac":[]}

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total, device=self.device)
            for start in range(0, total, cfg["minibatch_size"]):
                mb         = idx[start : start + cfg["minibatch_size"]]
                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                ratio      = (new_lp - lp_f[mb]).exp()
                surr1      = ratio * adv_f[mb]
                surr2      = ratio.clamp(1-cfg["clip_eps"], 1+cfg["clip_eps"]) * adv_f[mb]
                policy_loss= -torch.min(surr1, surr2).mean()

                vclip      = val_f[mb] + (value - val_f[mb]).clamp(-cfg["clip_eps"], cfg["clip_eps"])
                value_loss = torch.max(
                    (value - ret_f[mb]).pow(2),
                    (vclip  - ret_f[mb]).pow(2),
                ).mean()

                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["clip_frac"].append(clip_frac.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_step_{self.global_step:09d}" if not tag else f"checkpoint_{tag}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step":        self.global_step,
            "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(),
            "config":      self.cfg,
            "obs_dim":     self.env.OBS_DIM,
            "n_sectors":   N_SECTORS,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-20:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-20:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] Saved {path}")

    @classmethod
    def load_checkpoint(cls, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt   = torch.load(path, weights_only=False, map_location=device)
        trainer= cls(ckpt["config"])
        trainer.net.load_state_dict(ckpt["model_state"])
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(f"  Resumed step={trainer.global_step:,}  "
              f"ret={ckpt['metrics']['mean_return']:.3f}")
        return trainer


# ==========================================================================
#  Evaluation
# ==========================================================================

def evaluate(checkpoint_path, n_episodes=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg    = ckpt["config"]

    env = Go2LidarNavEnv(
        n_envs            = 1,
        headless          = False,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
        device            = device,
    )
    net = ActorCritic(env.OBS_DIM, Go2LidarNavEnv.ACT_DIM, cfg["hidden_size"])
    net.load_state_dict(ckpt["model_state"])
    net.eval().to(device)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len = 0.0, 0
        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done    = reset_buf.bool()
        print(f"  Episode {ep+1} | return={ep_ret:.2f} | length={ep_len}")


# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer   = args.n_envs * args.rollout_steps
    minibatch_size = max(total_buffer // 4, 256)
    return dict(
        n_envs             = args.n_envs,
        dt                 = 0.02,
        max_episode_steps  = 1000,
        headless           = args.headless,
        hidden_size        = 512,
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = minibatch_size,
        n_epochs           = 5,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 3e-4,
        vf_coef            = 1.0,
        ent_coef           = 0.01,
        max_grad_norm      = 1.0,
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",        type=int,  default=4096)
    parser.add_argument("--total-steps",   type=int,  default=200_000_000)
    parser.add_argument("--rollout-steps", type=int,  default=24)
    parser.add_argument("--device",        type=str,  default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-dir",       type=str,  default="../../runs/go2_lidar")
    parser.add_argument("--headless",      action="store_true", default=True)
    parser.add_argument("--resume",        type=str,  default=None)
    parser.add_argument("--eval",          type=str,  default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    if args.resume:
        trainer = PPOTrainer.load_checkpoint(args.resume)
    else:
        cfg     = get_config(args)
        trainer = PPOTrainer(cfg)

    trainer.train()


if __name__ == "__main__":
    main()
