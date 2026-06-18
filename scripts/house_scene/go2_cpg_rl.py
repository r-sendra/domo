"""
go2_cpg_rl.py
================
Self-contained RL experiment: Go2 locomotion on flat terrain
using a Central Pattern Generator (CPG) framework.

Features:
- Batched Analytical Inverse Kinematics
- Sim-to-Real Observation Noise Injection
- Adaptive KL-divergence Learning Rate Schedule
- Automatic Velocity Command Curriculum
- Paper Parity Hyperparameters (100Hz Control, [512, 256, 128] MLP)

Usage:
    # Train locally/HPC (defaults to 150M steps)
    python go2_cpg_rl.py --n-envs 4096 --device cuda --headless

    # Evaluate (Automatically falls back to CPU/Metal on Mac)
    python go2_cpg_rl.py --eval runs/go2_cpg/checkpoint_final.pt
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
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

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

BASE_INIT_POS  = [0.0, 0.0, 0.42]
BASE_INIT_QUAT = [0.0, 0.0, 0.0, 1.0]

# ==========================================================================
#  Analytical Inverse Kinematics for Unitree Go2
# ==========================================================================

def compute_go2_ik(x, y, z, is_left_leg):
    """
    Batched Analytical Inverse Kinematics for Unitree Go2.
    """
    L_HIP = 0.0955
    L_THIGH = 0.213
    L_CALF = 0.213

    # Hip Roll (q0)
    d_yz = torch.sqrt(y**2 + z**2)
    l_hip_eff = L_HIP * torch.where(is_left_leg, 1.0, -1.0)
    arg_hip = torch.clamp(l_hip_eff / d_yz, -1.0, 1.0)
    q0 = torch.atan2(y, -z) - torch.asin(arg_hip)

    # 2D Planar Projection for Thigh and Calf
    z_prime = -torch.sqrt(torch.clamp(y**2 + z**2 - L_HIP**2, min=1e-6))
    d_xz = torch.sqrt(x**2 + z_prime**2)

    # Calf Pitch (q2)
    arg_calf = (x**2 + z_prime**2 - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    arg_calf = torch.clamp(arg_calf, -1.0, 1.0)
    q2 = -torch.acos(arg_calf)

    # Thigh Pitch (q1)
    alpha = torch.atan2(x, -z_prime)
    arg_thigh = (d_xz**2 + L_THIGH**2 - L_CALF**2) / (2 * d_xz * L_THIGH)
    arg_thigh = torch.clamp(arg_thigh, -1.0, 1.0)
    q1 = alpha + torch.acos(arg_thigh)

    return q0, q1, q2

# ==========================================================================
#  Environment
# ==========================================================================

class Go2CPGEnv:
    ACT_DIM = 12

    def __init__(self, n_envs=4096, dt=0.01, max_episode_steps=2000, headless=True, device="cuda"):
        self.n_envs = n_envs
        self.num_envs = n_envs
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.device = torch.device(device)
        self.simulate_action_latency = True

        # ── Config ────────────────────────────────────────────────────────
        self.env_cfg = {
            "num_actions": 12,
            "base_init_pos": BASE_INIT_POS,
            "base_init_quat": BASE_INIT_QUAT,
            "episode_length_s": max_episode_steps * dt,
            "resampling_time_s": 4.0,
            "clip_actions": 1.0,
            "kp": 100.0, # Paper parity
            "kd": 2.0,   # Paper parity
            "dof_names": MOTOR_JOINT_NAMES,
            "default_joint_angles": DEFAULT_JOINT_ANGLES,
            "termination_if_pitch_greater_than": 1.0,
            "termination_if_roll_greater_than": 1.0,
        }
        self.obs_cfg = {
            "obs_scales": {
                "lin_vel": 2.0, "ang_vel": 0.25,
                "dof_pos": 1.0, "dof_vel": 0.05,
            },
            # Sim-to-Real Observation Noise scales
            "noise_scales": {
                "ang_vel": 0.05,
                "gravity": 0.02,
                "dof_pos": 0.01,
                "dof_vel": 1.5
            }
        }
        self.reward_cfg = {
            "tracking_sigma": 0.25,
            "reward_scales": {
                "tracking_lin_vel_x": 0.75,
                "tracking_lin_vel_y": 0.75,
                "tracking_ang_vel_z": 0.5,
                "lin_vel_z_penalty": -2.0,
                "ang_vel_xy_penalty": -0.05,
                "work_penalty": -0.001,
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.0, 1.0],
            "lin_vel_y_range": [-0.5, 0.5],
            "ang_vel_range":   [-0.5, 0.5],
        }

        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = {k: v * dt for k, v in self.reward_cfg["reward_scales"].items()}
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.num_commands = self.command_cfg["num_commands"]

        # ── Genesis init ──────────────────────────────────────────────────
        backend = gs.cuda if torch.cuda.is_available() and "cuda" in device else gs.cpu
        gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(BASE_INIT_POS[0] + 2.0, BASE_INIT_POS[1] - 2.0, 1.5),
                camera_lookat=(BASE_INIT_POS[0], BASE_INIT_POS[1], 0.4),
                camera_fov=50,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=not headless,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos = torch.tensor(BASE_INIT_POS, device=self.device)
        self.base_init_quat = torch.tensor(BASE_INIT_QUAT, device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        self.scene.build(n_envs=n_envs)

        # ── CPG and IK Setup ──────────────────────────────────────────────
        self.a_cpg = 150.0
        self.cpg_dt = 0.001
        self.n_cpg_substeps = int(self.dt / self.cpg_dt)

        self.cpg_r = torch.ones((n_envs, 4), device=self.device)
        self.cpg_dr = torch.zeros((n_envs, 4), device=self.device)
        self.cpg_theta = torch.zeros((n_envs, 4), device=self.device)

        # FR (0), FL (1), RR (2), RL (3)
        self.is_left = torch.tensor([False, True, False, True], device=self.device).expand(self.n_envs, 4)

        self.OBS_DIM = 45 + 20 # Proprioception + 20 CPG states
        print(f"  Total obs dim  : {self.OBS_DIM}")

        # ── Motor DOFs ────────────────────────────────────────────────────
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.ACT_DIM, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.ACT_DIM, self.motor_dofs)

        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][n] for n in self.env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        # ── Reward registry & State Buffers ───────────────────────────────
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_scales}
        self.episode_sums = {name: torch.zeros((n_envs,), device=self.device) for name in self.reward_scales}

        N, f = n_envs, gs.tc_float
        self.base_lin_vel = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=f).repeat(N, 1)

        self.obs_buf = torch.zeros((N, self.OBS_DIM), device=self.device, dtype=f)
        self.rew_buf = torch.zeros((N,), device=self.device, dtype=f)
        self.reset_buf = torch.ones((N,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((N,), device=self.device, dtype=gs.tc_int)

        self.commands = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device, dtype=f,
        )

        self.actions = torch.zeros((N, self.ACT_DIM), device=self.device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat = torch.zeros((N, 4), device=self.device, dtype=f)
        self.extras = {}

    def update_curriculum(self, global_step):
        """Gradually widen the command distribution to stabilize early learning."""
        if global_step < 20_000_000:
            self.command_cfg["lin_vel_x_range"] = [0.3, 0.5]
            self.command_cfg["lin_vel_y_range"] = [-0.1, 0.1]
            self.command_cfg["ang_vel_range"] = [-0.1, 0.1]
        elif global_step < 50_000_000:
            self.command_cfg["lin_vel_x_range"] = [0.0, 0.8]
            self.command_cfg["lin_vel_y_range"] = [-0.3, 0.3]
            self.command_cfg["ang_vel_range"] = [-0.3, 0.3]
        else:
            self.command_cfg["lin_vel_x_range"] = [0.0, 1.0]
            self.command_cfg["lin_vel_y_range"] = [-0.5, 0.5]
            self.command_cfg["ang_vel_range"] = [-0.5, 0.5]

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # 1. Map RL actions to CPG Parameters
        mu = 1.5 + 0.5 * exec_actions[:, 0::3]                 # Amplitude target: [1, 2]
        omega_hz = 2.25 * (exec_actions[:, 1::3] + 1.0)        # Freq target: [0, 4.5 Hz]
        omega = 2 * math.pi * omega_hz                         # Rad/s
        psi = 1.5 * exec_actions[:, 2::3]                      # Phase bias: [-1.5, 1.5]

        # 2. Integrate CPG Euler Sub-steps
        for _ in range(self.n_cpg_substeps):
            ddr = self.a_cpg * ((self.a_cpg / 4.0) * (mu - self.cpg_r) - self.cpg_dr)
            self.cpg_dr += ddr * self.cpg_dt
            self.cpg_r += self.cpg_dr * self.cpg_dt
            self.cpg_theta += omega * self.cpg_dt

        self.cpg_theta = torch.fmod(self.cpg_theta, 2 * math.pi)

        # 3. Map to Cartesian trajectories in Leg Frame
        d_step = 0.10
        h_robot = 0.32
        g_c = 0.12  # Clearance
        g_p = 0.02  # Penetration

        r_minus_1 = self.cpg_r - 1.0
        x_foot = -d_step * r_minus_1 * torch.cos(self.cpg_theta) * torch.cos(psi)
        y_foot = -d_step * r_minus_1 * torch.cos(self.cpg_theta) * torch.sin(psi)

        sin_theta = torch.sin(self.cpg_theta)
        z_foot = torch.where(sin_theta > 0, -h_robot + g_c * sin_theta, -h_robot + g_p * sin_theta)

        # 4. Compute Analytical IK
        q0, q1, q2 = compute_go2_ik(x_foot, y_foot, z_foot, self.is_left)
        
        target_dof_pos = torch.empty((self.n_envs, 12), device=self.device)
        target_dof_pos[:, 0::3] = q0
        target_dof_pos[:, 1::3] = q1
        target_dof_pos[:, 2::3] = q2

        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # ── Update state ──────────────────────────────────────────────────
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat))
        inv_base_quat = inv_quat(self.base_quat)
        
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = ((self.episode_length_buf % resample_every == 0).nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        # ── Termination ───────────────────────────────────────────────────
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # ── Reward ────────────────────────────────────────────────────────
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # ── Observation Noise Injection (Sim-to-Real Prep) ────────────────
        ns = self.obs_cfg["noise_scales"]
        noise_ang_vel = torch.randn_like(self.base_ang_vel) * ns["ang_vel"]
        noise_gravity = torch.randn_like(self.projected_gravity) * ns["gravity"]
        noise_dof_pos = torch.randn_like(self.dof_pos) * ns["dof_pos"]
        noise_dof_vel = torch.randn_like(self.dof_vel) * ns["dof_vel"]

        cpg_obs = torch.cat([self.cpg_r, self.cpg_dr, self.cpg_theta, omega, psi], dim=-1)
        
        self.obs_buf = torch.cat([
            (self.base_ang_vel + noise_ang_vel) * self.obs_scales["ang_vel"],
            (self.projected_gravity + noise_gravity),
            self.commands * self.commands_scale,
            ((self.dof_pos + noise_dof_pos) - self.default_dof_pos) * self.obs_scales["dof_pos"],
            (self.dof_vel + noise_dof_vel) * self.obs_scales["dof_vel"],
            self.actions,
            cpg_obs
        ], dim=-1)

        self.last_actions[:] = self.actions[:]
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0: return

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(self.dof_pos[envs_idx], self.motor_dofs, zero_velocity=True, envs_idx=envs_idx)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset CPG: Initialize to a nominal trot to speed up initial learning
        self.cpg_r[envs_idx] = 1.0
        self.cpg_dr[envs_idx] = 0.0
        self.cpg_theta[envs_idx, 0] = 0.0       # FR
        self.cpg_theta[envs_idx, 1] = math.pi   # FL
        self.cpg_theta[envs_idx, 2] = math.pi   # RR
        self.cpg_theta[envs_idx, 3] = 0.0       # RL

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"])
            self.episode_sums[key][envs_idx] = 0.0
        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0: return
        n = len(envs_idx)
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (n,), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (n,), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (n,), self.device)

    # ── Paper 1:1 Rewards ──────────────────────────────────────────────
    def _reward_tracking_lin_vel_x(self):
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel_z(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z_penalty(self):
        # Penalizes bouncing up and down (replaces static height target)
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy_penalty(self):
        # Penalizes body roll and pitch rates to keep the chassis flat
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_work_penalty(self):
        # Work ζ : -|tau * (q_dot_t - q_dot_t-1)|
        # Extract torques calculated by the physics engine in the previous step
        torques = self.robot.get_dofs_force(self.motor_dofs)
        dof_vel_delta = self.dof_vel - self.last_dof_vel
        work = torch.sum(torch.abs(torques * dof_vel_delta), dim=1)
        return work

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

# ==========================================================================
#  ActorCritic & Rollout Buffer
# ==========================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Matches paper: [512, 256, 128] asymmetric MLP
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU()
        )
        self.actor_head = nn.Linear(128, act_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.00)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.actor_head(h)
        return mean, self.log_std.exp().expand_as(mean), self.critic_head(h).squeeze(-1)

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        return action, dist.log_prob(action).sum(-1), value

    def get_value(self, obs): return self.critic_head(self.trunk(obs)).squeeze(-1)

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    @property
    def num_parameters(self): return sum(p.numel() for p in self.parameters())

class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, obs_dim, act_dim, device):
        self.T, self.N, self.device, self.ptr = rollout_steps, n_envs, device, 0
        buf = lambda *s: torch.zeros(*s, device=device)
        self.obs, self.actions = buf(self.T, self.N, obs_dim), buf(self.T, self.N, act_dim)
        self.log_probs, self.values = buf(self.T, self.N), buf(self.T, self.N)
        self.rewards, self.dones = buf(self.T, self.N), buf(self.T, self.N)
        self.advantages, self.returns = buf(self.T, self.N), buf(self.T, self.N)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t], self.actions[t] = obs.detach(), actions.detach()
        self.log_probs[t], self.values[t] = log_probs.detach(), values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t], self.dones[t] = rewards.detach().float(), dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        T, N = self.T, self.N
        return (self.obs.view(T*N, -1), self.actions.view(T*N, -1), self.log_probs.view(T*N),
                self.advantages.view(T*N), self.returns.view(T*N), self.values.view(T*N))

# ==========================================================================
#  PPO Trainer
# ==========================================================================

class PPOTrainer:
    def __init__(self, cfg):
        self.cfg, self.device = cfg, cfg["device"]
        self.env = Go2CPGEnv(
            n_envs=cfg["n_envs"], dt=cfg["dt"], max_episode_steps=cfg["max_episode_steps"],
            headless=cfg["headless"], device=cfg["device"]
        )

        self.net = ActorCritic(self.env.OBS_DIM, self.env.ACT_DIM).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], eps=1e-5)
        self.buf = RolloutBuffer(cfg["rollout_steps"], cfg["n_envs"], self.env.OBS_DIM, self.env.ACT_DIM, self.device)

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer = SummaryWriter(cfg["run_dir"])
        self.global_step, self.start_time = 0, time.time()
        self.ep_returns, self.ep_lengths = [], []
        self._env_ret = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_len = torch.zeros(cfg["n_envs"], device=self.device, dtype=torch.int32)

    def train(self):
        steps_per_rollout = self.cfg["rollout_steps"] * self.cfg["n_envs"]
        n_updates = self.cfg["total_steps"] // steps_per_rollout
        obs, _ = self.env.reset()

        print(f"\n=======================================================")
        print(f"  Go2 CPG-RL Training (Paper Parity Mode)")
        print(f"=======================================================")

        for update in range(1, n_updates + 1):
            self.env.update_curriculum(self.global_step)
            obs = self._collect_rollout(obs)
            metrics = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % self.cfg["log_interval"] == 0:
                sps = self.global_step / (time.time() - self.start_time)
                mean_ret = float(np.mean(self.ep_returns[-20:])) if self.ep_returns else 0.0
                mean_len = float(np.mean(self.ep_lengths[-20:])) if self.ep_lengths else 0.0
                current_lr = self.opt.param_groups[0]['lr']

                print(f"  step {self.global_step:>10,} | ret {mean_ret:>7.3f} | len {mean_len:>5.0f} | "
                      f"ploss {metrics['policy_loss']:>7.4f} | vloss {metrics['value_loss']:>7.4f} | lr {current_lr:.1e} | {sps:>7,.0f} sps")
                
                self.writer.add_scalar("train/mean_return", mean_ret, self.global_step)
                self.writer.add_scalar("train/mean_ep_len", mean_len, self.global_step)
                self.writer.add_scalar("loss/policy", metrics["policy_loss"], self.global_step)
                self.writer.add_scalar("loss/value", metrics["value_loss"], self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)

            if update % self.cfg["save_interval"] == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.writer.close()

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
                for idx in reset_buf.nonzero(as_tuple=False).flatten():
                    self.ep_returns.append(float(self._env_ret[idx]))
                    self.ep_lengths.append(int(self._env_len[idx]))
                self._env_ret[reset_buf] = 0.0
                self._env_len[reset_buf] = 0
                obs = next_obs

            last_value = self.net.get_value(obs)
            self.buf.compute_gae(last_value, self.cfg["gamma"], self.cfg["lam"])
        return obs

    def _ppo_update(self):
        self.net.train()
        cfg = self.cfg
        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        metrics = {"policy_loss": [], "value_loss": []}
        total = obs_f.shape[0]

        target_kl = 0.01 # KL Divergence Target

        for _ in range(cfg["n_epochs"]):
            # Early stopping and adaptive LR based on KL divergence
            with torch.no_grad():
                new_lp, _, _ = self.net.evaluate(obs_f, act_f)
                kl_div = (lp_f - new_lp).mean().item()

            if kl_div > target_kl * 1.5:
                self.opt.param_groups[0]['lr'] = max(self.opt.param_groups[0]['lr'] / 1.5, 1e-5)
            elif kl_div < target_kl / 2.0:
                self.opt.param_groups[0]['lr'] = min(self.opt.param_groups[0]['lr'] * 1.5, 1e-3)

            idx = torch.randperm(total, device=self.device)
            for start in range(0, total, cfg["minibatch_size"]):
                mb = idx[start : start + cfg["minibatch_size"]]
                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                ratio = (new_lp - lp_f[mb]).exp()
                surr1 = ratio * adv_f[mb]
                surr2 = ratio.clamp(1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv_f[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                vclip = val_f[mb] + (value - val_f[mb]).clamp(-cfg["clip_eps"], cfg["clip_eps"])
                value_loss = torch.max((value - ret_f[mb]).pow(2), (vclip - ret_f[mb]).pow(2)).mean()

                loss = policy_loss + cfg["vf_coef"] * value_loss - cfg["ent_coef"] * entropy.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def _save_checkpoint(self, tag=None):
        name = f"checkpoint_step_{self.global_step:09d}" if not tag else f"checkpoint_{tag}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step": self.global_step, "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(), "config": self.cfg,
        }, path)
        print(f"  [ckpt] Saved {path}")

# ==========================================================================
#  Evaluation
# ==========================================================================

def evaluate(checkpoint_path, n_episodes=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    
    obs_dim = ckpt["model_state"]["trunk.0.weight"].shape[1]
    act_dim = ckpt["model_state"]["actor_head.weight"].shape[0]

    env = Go2CPGEnv(n_envs=1, headless=False, max_episode_steps=cfg["max_episode_steps"], dt=cfg["dt"], device=device)
    net = ActorCritic(obs_dim, act_dim)
    net.load_state_dict(ckpt["model_state"])
    net.eval().to(device)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len = 0.0, 0

        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=True)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done = reset_buf.bool()

        print(f"  Episode {ep+1} | return={ep_ret:.2f} | length={ep_len}")

# ==========================================================================
#  Entry point
# ==========================================================================

def get_config(args):
    total_buffer = args.n_envs * args.rollout_steps
    return dict(
        n_envs=args.n_envs, dt=0.01, max_episode_steps=2000, headless=args.headless,
        total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256), n_epochs=5, gamma=0.99, lam=0.95,
        clip_eps=0.2, lr=3e-4, vf_coef=1.0, ent_coef=0.01, max_grad_norm=1.0,
        device=args.device, run_dir=args.run_dir, log_interval=10, save_interval=100,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4096)
    parser.add_argument("--total-steps", type=int, default=150_000_000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--run-dir", type=str, default="runs/go2_cpg")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--eval", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.eval: evaluate(args.eval)
    else: PPOTrainer(get_config(args)).train()

if __name__ == "__main__":
    main()
