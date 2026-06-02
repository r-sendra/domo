"""
go2.py — Go2 Walking Environment (Genesis)
===========================================
Task: velocity-tracking locomotion on flat terrain.

This version closely follows the official Genesis locomotion example
(Genesis-Embodied-AI/Genesis/examples/locomotion/go2_env.py) while
preserving our class name (Go2WalkEnv) and import structure.

Key differences from previous version that prevented walking:
  1. Everything stays on GPU as torch tensors — no .numpy() in the
     hot path. The previous version converted to numpy every step,
     breaking GPU-native computation and causing subtle dtype issues.

  2. simulate_action_latency = True — the real Go2 has a 1-step
     delay between command and execution. Without this the policy
     learns to exploit instantaneous control which does not transfer.

  3. base_init_quat = [0,0,0,1] — Genesis uses [x,y,z,w] quaternion
     convention internally. The previous [1,0,0,0] was wrong for
     Genesis and caused the robot to spawn with a twisted orientation.

  4. RigidOptions with Newton constraint solver — matches the official
     example and gives more stable contact simulation.

  5. Termination uses base_euler (roll/pitch angles) from quat_to_xyz,
     not projected gravity thresholds. More numerically stable.

  6. Reward scales multiplied by dt at init — makes weights
     timestep-independent (standard legged_gym practice).

  7. Observation scaling: lin_vel*2.0, ang_vel*0.25, dof_vel*0.05.
     Without scaling the obs values are in very different ranges which
     hurts network learning.

  8. performance_mode=True in gs.init for maximum GPU throughput.

Observation space (45 values per env):
  [0:3]   base angular velocity   (body frame, scaled *0.25)
  [3:6]   projected gravity       (body frame)
  [6:9]   velocity command        (scaled)
  [9:21]  joint pos relative to default (scaled *1.0)
  [21:33] joint velocities        (scaled *0.05)
  [33:45] previous action

Action space (12 values per env):
  Target joint position offsets from default stance * action_scale.

Reward (official Genesis 6-term set):
  + 1.0  * exp(-||cmd_xy - lin_vel_xy||^2 / sigma)  lin vel tracking
  + 0.2  * exp(-(cmd_yaw - ang_vel_z)^2  / sigma)   ang vel tracking
  - 1.0  * lin_vel_z^2                               no bouncing
  - 50.0 * (base_height - 0.34)^2                   stay at height
  - 0.005* ||action - last_action||^2               smoothness
  - 0.1  * ||dof_pos - default||                    near default pose

Done conditions:
  - episode_length > max_episode_length  (timeout)
  - |pitch| > termination_if_pitch_greater_than (1.0 rad)
  - |roll|  > termination_if_roll_greater_than  (1.0 rad)
"""

import math
import torch
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTOR_JOINT_NAMES = [
    "FR_hip_joint",   "FR_thigh_joint",  "FR_calf_joint",
    "FL_hip_joint",   "FL_thigh_joint",  "FL_calf_joint",
    "RR_hip_joint",   "RR_thigh_joint",  "RR_calf_joint",
    "RL_hip_joint",   "RL_thigh_joint",  "RL_calf_joint",
]

# Default joint angles in MOTOR_JOINT_NAMES order
DEFAULT_JOINT_ANGLES = {
    "FR_hip_joint":    0.0,
    "FR_thigh_joint":  0.8,
    "FR_calf_joint":  -1.5,
    "FL_hip_joint":    0.0,
    "FL_thigh_joint":  0.8,
    "FL_calf_joint":  -1.5,
    "RR_hip_joint":    0.0,
    "RR_thigh_joint":  1.0,
    "RR_calf_joint":  -1.5,
    "RL_hip_joint":    0.0,
    "RL_thigh_joint":  1.0,
    "RL_calf_joint":  -1.5,
}

# NOTE: Genesis uses [x, y, z, w] quaternion convention
BASE_INIT_POS  = [0.0, 0.0, 0.42]
BASE_INIT_QUAT = [0.0, 0.0, 0.0, 1.0]   # identity in [x,y,z,w]


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Go2WalkEnv:
    """
    Vectorised Genesis environment for Go2 velocity-tracking locomotion.
    Follows the official Genesis go2_env.py example closely.
    All internal state is kept as torch tensors on the simulation device.
    """

    OBS_DIM = 45
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
        self.num_envs          = n_envs   # alias used by some callers
        self.dt                = dt
        self.max_episode_steps = max_episode_steps
        self.device            = torch.device(device)

        self.simulate_action_latency = True   # matches real Go2 hardware

        # ---- configs (mirrors official get_cfgs structure) ----
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
            "num_obs": 45,
            "obs_scales": {
                "lin_vel":  2.0,
                "ang_vel":  0.25,
                "dof_pos":  1.0,
                "dof_vel":  0.05,
            },
        }
        self.reward_cfg = {
            "tracking_sigma":     0.25,
            "base_height_target": 0.34,
            "reward_scales": {
                "tracking_lin_vel": 1.0,
                "tracking_ang_vel": 0.2,
                "lin_vel_z":       -1.0,
                "base_height":    -50.0,
                "action_rate":    -0.005,
                "similar_to_default": -0.1,
            },
        }
        self.command_cfg = {
            "num_commands": 3,
            "lin_vel_x_range": [0.5, 0.5],   # fixed forward command (official)
            "lin_vel_y_range": [0.0, 0.0],
            "ang_vel_range":   [0.0, 0.0],
        }

        self.num_obs     = self.obs_cfg["num_obs"]
        self.num_actions = self.env_cfg["num_actions"]
        self.num_commands= self.command_cfg["num_commands"]
        self.obs_scales  = self.obs_cfg["obs_scales"]
        self.reward_scales = {
            k: v * dt
            for k, v in self.reward_cfg["reward_scales"].items()
        }
        self.max_episode_length = math.ceil(
            self.env_cfg["episode_length_s"] / self.dt
        )

        # ---- Genesis init ----
        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        gs.init(
            backend=backend,
            precision="32",
            logging_level="warning",
            performance_mode=True,
        )

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
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

        # Ground
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # Robot
        self.base_init_pos  = torch.tensor(
            self.env_cfg["base_init_pos"],  device=self.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        self.scene.build(n_envs=n_envs)

        # Motor DOF indices
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]

        # PD gains
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.num_actions, self.motor_dofs
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.num_actions, self.motor_dofs
        )

        # Default joint positions tensor
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][n]
             for n in self.env_cfg["dof_names"]],
            device=self.device, dtype=gs.tc_float,
        )

        # Reward function registry
        self.reward_functions = {}
        self.episode_sums     = {}
        for name in self.reward_scales:
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name] = torch.zeros(
                (n_envs,), device=self.device, dtype=gs.tc_float
            )

        # State buffers — all torch tensors on device
        N = n_envs
        f = gs.tc_float
        self.base_lin_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_ang_vel      = torch.zeros((N, 3), device=self.device, dtype=f)
        self.projected_gravity = torch.zeros((N, 3), device=self.device, dtype=f)
        self.global_gravity    = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=f
        ).repeat(N, 1)

        self.obs_buf           = torch.zeros((N, self.num_obs), device=self.device, dtype=f)
        self.rew_buf           = torch.zeros((N,),              device=self.device, dtype=f)
        self.reset_buf         = torch.ones( (N,),              device=self.device, dtype=gs.tc_int)
        self.episode_length_buf= torch.zeros((N,),              device=self.device, dtype=gs.tc_int)

        self.commands          = torch.zeros((N, self.num_commands), device=self.device, dtype=f)
        self.commands_scale    = torch.tensor(
            [self.obs_scales["lin_vel"],
             self.obs_scales["lin_vel"],
             self.obs_scales["ang_vel"]],
            device=self.device, dtype=f,
        )

        self.actions      = torch.zeros((N, self.num_actions), device=self.device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos     = torch.zeros((N, 3), device=self.device, dtype=f)
        self.base_quat    = torch.zeros((N, 4), device=self.device, dtype=f)
        self.extras       = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, actions):
        """
        actions : Tensor [N, 12]  (torch, on device)
        returns : obs_buf, None, rew_buf, reset_buf, extras
        """
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
             self.env_cfg["clip_actions"],
        )

        # 1-step action latency: execute last step's actions
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # Update state buffers
        self.episode_length_buf += 1
        self.base_pos[:]  = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler   = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:]      = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:]      = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity[:] = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # Resample commands periodically
        resample_every = int(self.env_cfg["resampling_time_s"] / self.dt)
        envs_idx = (
            (self.episode_length_buf % resample_every == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(envs_idx)

        # Termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False).flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Reward
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Observation
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],            # 3
                self.projected_gravity,                                     # 3
                self.commands * self.commands_scale,                        # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                  # 12
                self.actions,                                               # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

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
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Reset base pose
        self.base_pos[envs_idx]  = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx]       = 0.0
        self.last_dof_vel[envs_idx]       = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx]          = True

        # Log episode sums
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
            *self.command_cfg["ang_vel_range"], (n,), self.device
        )

    # ------------------------------------------------------------------
    # Reward functions (official Genesis 6-term set)
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
