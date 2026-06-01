"""
go2_env.py — Go2 Stand-Up Environment (Genesis)  [fixed]
=========================================================
Fixes from v1:
  - set_euler does not exist → replaced with set_quat() using euler_to_quat()
  - base_lin_vel now correctly obtained via robot.get_vel() (world frame)
    then rotated to body frame with quat_rotate_inverse()
  - base_ang_vel now correctly obtained via robot.get_ang() (world frame, 0.3.x API)
  - Robot spawns upright (not lying down) — Genesis resets are always to a
    known-good pose; the RL learns to recover from perturbations, not from
    arbitrary initial conditions (matching the official Genesis example)
  - set_dofs_kp / set_dofs_kv added for proper PD control (matching real robot)

Observation space (48 values per env):
  [0:3]   base linear velocity    (body frame)
  [3:6]   base angular velocity   (body frame)
  [6:9]   projected gravity       (body frame)
  [9:12]  velocity command        [vx, vy, yaw_rate]
  [12:24] joint pos (relative)    relative to default stand pose
  [24:36] joint vel
  [36:48] previous action

Action space (12 values per env):
  Target joint position offsets from default stand pose, scaled by ACTION_SCALE.

Reward (per step):
  +1.0  * exp(-20 * (height - 0.34)^2)      height near target
  +0.5  * dot(proj_gravity, [0,0,-1])        robot upright
  -0.01 * sum(joint_vel^2)                   stay still
  -0.005 * sum(action^2)                     energy efficiency
  -0.005 * sum((action - prev_action)^2)     smooth actions

Done conditions:
  - base height < 0.15 m         (fallen)
  - proj_gravity Z in body > 0.5 (flipped)
  - step count >= max_episode_steps (timeout)

Usage:
    from go2_env import Go2StandEnv
    env = Go2StandEnv(n_envs=64, headless=True)
    obs = env.reset()
    obs, reward, done, info = env.step(action)  # action: (N, 12) in [-1, 1]
"""

import math
import numpy as np
import torch
import genesis as gs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOTOR_DOFS = list(range(6, 18))   # skip free-base DOFs 0-5

DEFAULT_JOINT_POS = np.array([
    0.0,  0.8, -1.5,   # FL  hip, thigh, calf
    0.0,  0.8, -1.5,   # FR
    0.0,  1.0, -1.5,   # RL
    0.0,  1.0, -1.5,   # RR
], dtype=np.float32)

# Upright spawn: position and quaternion [w, x, y, z]
BASE_INIT_POS  = np.array([0.0, 0.0, 0.42], dtype=np.float32)
BASE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity = upright

TARGET_HEIGHT = 0.34    # Go2 nominal standing height [m]
ACTION_SCALE  = 0.25    # rad — joint offset scale
OBS_CLIP      = 5.0     # clip obs to prevent outliers

# PD gains matching the real Unitree Go2
KP = 20.0   # position stiffness
KV = 0.5    # velocity damping


# ---------------------------------------------------------------------------
# Geometry helpers (no Genesis dependency — pure numpy)
# ---------------------------------------------------------------------------

def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Vectorised Euler (rad) → quaternion [w, x, y, z].
    Inputs can be scalars or (N,) arrays.
    """
    cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate world-frame vectors v into body frame using quaternion q.
    Equivalent to R(q)^T @ v.

    q : (N, 4)  [w, x, y, z]
    v : (N, 3)
    returns (N, 3)
    """
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    bx = (1 - 2*(y*y + z*z)) * vx + 2*(x*y + w*z) * vy + 2*(x*z - w*y) * vz
    by = 2*(x*y - w*z) * vx + (1 - 2*(x*x + z*z)) * vy + 2*(y*z + w*x) * vz
    bz = 2*(x*z + w*y) * vx + 2*(y*z - w*x) * vy + (1 - 2*(x*x + y*y)) * vz
    return np.concatenate([bx, by, bz], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class Go2StandEnv:
    """
    Vectorised Genesis environment for the Go2 stand-up task.

    All state tensors live on CPU (Genesis CPU backend).
    obs, reward, done are returned as numpy arrays.
    """

    OBS_DIM = 48
    ACT_DIM = 12

    def __init__(
        self,
        n_envs:            int   = 64,
        dt:                float = 0.02,   # 50 Hz — matches real robot
        substeps:          int   = 2,
        max_episode_steps: int   = 500,
        headless:          bool  = True,
        device:            str   = "cpu",
    ):
        self.n_envs            = n_envs
        self.dt                = dt
        self.max_episode_steps = max_episode_steps
        self.device            = device

        # --- Genesis initialisation ---
        gs.init(backend=gs.cpu, logging_level="warning")

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos    = (3.0, -2.0, 2.0),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
                max_FPS       = 60,
            ),
            sim_options=gs.options.SimOptions(
                dt       = dt,
                substeps = substeps,
            ),
            show_viewer=not headless,
        )

        # Ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Go2 — spawns upright at standing height
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file  = "urdf/go2/urdf/go2.urdf",
                pos   = BASE_INIT_POS.tolist(),
                quat  = BASE_INIT_QUAT.tolist(),   # [w, x, y, z]
            )
        )

        # Build with n_envs parallel instances
        self.scene.build(n_envs=n_envs)

        # --- PD controller gains (match real robot) ---
        self.robot.set_dofs_kp(
            [KP] * self.ACT_DIM,
            dofs_idx_local=MOTOR_DOFS,
        )
        self.robot.set_dofs_kv(
            [KV] * self.ACT_DIM,
            dofs_idx_local=MOTOR_DOFS,
        )

        # --- Internal state buffers ---
        self._step_count  = np.zeros(n_envs, dtype=np.int32)
        self._prev_action = np.zeros((n_envs, self.ACT_DIM), dtype=np.float32)
        self._ep_return   = np.zeros(n_envs, dtype=np.float32)
        self._ep_length   = np.zeros(n_envs, dtype=np.int32)
        self._commands    = np.zeros((n_envs, 3), dtype=np.float32)  # [vx, vy, yaw]

        # Gravity vector in world frame
        self._gravity_world = np.tile([0.0, 0.0, -1.0], (n_envs, 1)).astype(np.float32)

        # Resample commands every 2s (was 4s) — prevents standing still exploitation
        self._resample_every = int(2.0 / self.dt)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset all envs and return initial observations."""
        self._reset_envs(np.arange(self.n_envs))
        return self._get_obs()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        action : (n_envs, 12)  float32, values in [-1, 1]
        returns: obs (N,48), reward (N,), done (N,), info dict
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_pos = DEFAULT_JOINT_POS + ACTION_SCALE * action   # (N, 12)

        # Apply position command (PD controller runs inside Genesis)
        self.robot.control_dofs_position(
            target_pos,
            dofs_idx_local=self.motor_dofs,
        )
        self.scene.step()

        obs    = self._get_obs()
        reward = self._compute_reward(obs, action)
        done   = self._compute_done(obs)

        # Track episode stats
        self._ep_return += reward
        self._ep_length += 1
        self._step_count += 1
        self._prev_action = action.copy()

        # Resample commands periodically (every 2s) — prevents standing exploitation
        resample_ids = np.where(self._step_count % self._resample_every == 0)[0]
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Collect completed episode stats before reset
        info = {}
        finished = np.where(done)[0]
        if len(finished) > 0:
            info["episode"] = {
                "return": float(self._ep_return[finished].mean()),
                "length": float(self._ep_length[finished].mean()),
            }
            self._reset_envs(finished)

        return obs, reward, done, info

    def close(self):
        pass   # Genesis cleans up on process exit

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_envs(self, env_ids: np.ndarray):
        """
        Reset specific envs to upright standing pose with small random perturbations.
        Uses set_pos() + set_quat() — the correct Genesis API.
        """
        n = len(env_ids)
        if n == 0:
            return

        # --- Base position: standing height + small xy noise ---
        pos = np.tile(BASE_INIT_POS, (n, 1))
        pos[:, :2] += np.random.uniform(-0.05, 0.05, (n, 2)).astype(np.float32)
        pos[:, 2]  += np.random.uniform(-0.02, 0.02, n).astype(np.float32)
        pos[:, 2]   = np.clip(pos[:, 2], 0.35, 0.50)

        # --- Base orientation: upright + random yaw ---
        yaw = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
        quat = euler_to_quat(
            np.zeros(n, np.float32),
            np.zeros(n, np.float32),
            yaw,
        )   # (n, 4)  [w, x, y, z]

        # --- Joint positions: default + small noise ---
        jpos = (DEFAULT_JOINT_POS
                + np.random.uniform(-0.05, 0.05, (n, self.ACT_DIM)).astype(np.float32))

        # --- Apply to Genesis ---
        self.robot.set_pos(pos,  envs_idx=env_ids)
        self.robot.set_quat(quat, envs_idx=env_ids)
        self.robot.set_dofs_position(
            jpos,
            dofs_idx_local=self.motor_dofs,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        # --- Reset internal buffers ---
        self._step_count[env_ids]  = 0
        self._prev_action[env_ids] = 0.0
        self._ep_return[env_ids]   = 0.0
        self._ep_length[env_ids]   = 0
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: np.ndarray):
        """Sample new random velocity commands for selected envs."""
        n = len(env_ids)
        self._commands[env_ids, 0] = np.random.uniform(*CMD_LIN_VEL_X, n).astype(np.float32)
        self._commands[env_ids, 1] = np.random.uniform(*CMD_LIN_VEL_Y, n).astype(np.float32)
        self._commands[env_ids, 2] = np.random.uniform(*CMD_ANG_VEL_Z, n).astype(np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build (n_envs, 48) observation from Genesis state.

        Genesis API used:
          robot.get_pos()            -> (N, 3)  world frame position
          robot.get_quat()           -> (N, 4)  [w, x, y, z] world orientation
          robot.get_vel()            -> (N, 3)  world frame linear velocity
          robot.get_ang()            -> (N, 3)  world frame angular velocity  (0.3.x API)
          robot.get_dofs_position()  -> (N, 18) all DOF positions
          robot.get_dofs_velocity()  -> (N, 18) all DOF velocities
        """
        # Get raw Genesis state (returns torch tensors, convert to numpy)
        base_pos      = self.robot.get_pos().numpy()           # (N, 3)
        base_quat     = self.robot.get_quat().numpy()          # (N, 4)  w,x,y,z
        vel_world     = self.robot.get_vel().numpy()           # (N, 3)  world frame
        angv_world    = self.robot.get_ang().numpy()           # (N, 3)  world frame
        dof_pos_all   = self.robot.get_dofs_position().numpy() # (N, 18)
        dof_vel_all   = self.robot.get_dofs_velocity().numpy() # (N, 18)

        # Rotate world-frame velocities to body frame
        base_lin_vel  = quat_rotate_inverse(base_quat, vel_world)    # (N, 3)
        base_ang_vel  = quat_rotate_inverse(base_quat, angv_world)   # (N, 3)

        # Projected gravity: gravity world vector rotated to body frame
        proj_gravity  = quat_rotate_inverse(base_quat, self._gravity_world)  # (N, 3)

        # Motor joints only
        motor_pos = dof_pos_all[:, MOTOR_DOFS]          # (N, 12)
        motor_vel = dof_vel_all[:, MOTOR_DOFS]          # (N, 12)
        motor_pos_rel = motor_pos - DEFAULT_JOINT_POS   # (N, 12) relative

        # Assemble (N, 48)
        obs = np.concatenate([
            base_lin_vel,           # [0:3]
            base_ang_vel,           # [3:6]
            proj_gravity,           # [6:9]
            self._commands,         # [9:12]
            motor_pos_rel,          # [12:24]
            motor_vel,              # [24:36]
            self._prev_action,      # [36:48]
        ], axis=1).astype(np.float32)

        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, obs: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Walking Reward Function based on standard Legged Locomotion RL.
        """
        # --- 1. Extract states from observation ---
        height        = self.robot.get_pos().cpu().numpy()[:, 2]   # (N,)
        lin_vel       = obs[:, 0:3]    # (N, 3) body frame
        ang_vel       = obs[:, 3:6]    # (N, 3) body frame
        proj_grav     = obs[:, 6:9]    # (N, 3) projected gravity
        commands      = obs[:, 9:12]   # (N, 3) [vx, vy, yaw_rate]
        motor_pos_rel = obs[:, 12:24]  # (N, 12) relative to nominal
        motor_vel     = obs[:, 24:36]  # (N, 12)
        prev_action   = obs[:, 36:48]  # (N, 12)

        cmd_mag = np.linalg.norm(commands[:, :2], axis=1)   # (N,)

        # --- 2. Task Rewards (Tracking) ---
        lin_vel_error = np.sum(np.square(commands[:, :2] - lin_vel[:, :2]), axis=1)
        r_track_lin = np.exp(-lin_vel_error / 0.25)

        ang_vel_error = np.square(commands[:, 2] - ang_vel[:, 2])
        r_track_ang = np.exp(-ang_vel_error / 0.25)

        # Penalise standing still when a non-zero command is given.
        # vel_mag near zero + large cmd → large negative reward.
        # vel_mag growing → exp shrinks → penalty disappears naturally.
        vel_mag       = np.linalg.norm(lin_vel[:, :2], axis=1)
        r_stand_still = -np.square(cmd_mag) * np.exp(-vel_mag / 0.2)

        # --- 3. Base Motion Penalties ---
        r_lin_vel_z  = -np.square(lin_vel[:, 2])
        r_ang_vel_xy = -np.sum(np.square(ang_vel[:, :2]), axis=1)

        # --- 4. Posture Penalties ---
        # Orientation: strong weight to stay upright (prevents belly-flopping)
        r_orientation = -np.sum(np.square(proj_grav[:, :2]), axis=1)

        # Height: strong weight to stay at nominal standing height
        r_height = -np.square(height - 0.34)

        # Joint deviation: only penalise when standing still
        r_dof_pos  = -np.sum(np.square(motor_pos_rel), axis=1)
        r_dof_pos *= (cmd_mag < 0.1).astype(np.float32)

        # --- 5. Energy and Smoothness Penalties ---
        r_action_mag  = -np.sum(np.square(action), axis=1)
        r_action_rate = -np.sum(np.square(action - prev_action), axis=1)
        r_joint_vel   = -np.sum(np.square(motor_vel), axis=1)

        # --- 6. Combine ---
        reward = (
             1.0   * r_track_lin
           + 0.2   * r_track_ang
           + 2.0   * r_stand_still   # break standing local optimum
           + 2.0   * r_lin_vel_z
           + 0.05  * r_ang_vel_xy
           + 2.5   * r_orientation   # was 0.2 — strong upright signal
           + 50.0  * r_height        # was 2.0 — prevents belly-flopping
           + 0.1   * r_dof_pos       # command-masked
           + 0.005 * r_action_mag
           + 0.005 * r_action_rate
           + 0.001 * r_joint_vel
        )
        
        return reward.astype(np.float32)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _compute_done(self, obs: np.ndarray) -> np.ndarray:
            height     = self.robot.get_pos().numpy()[:, 2]
            proj_grav  = obs[:, 6:9]

            # Existing conditions
            done_fallen  = height < 0.28
            done_flipped = proj_grav[:, 2] > 0.5 
            
            # NEW: Terminate if the robot pitches or rolls more than ~45 degrees.
            # proj_grav x and y components exceed 0.7 when tilted severely.
            done_pitch   = np.abs(proj_grav[:, 0]) > 0.7
            done_roll    = np.abs(proj_grav[:, 1]) > 0.7

            done_timeout = self._step_count >= self.max_episode_steps

            return (done_fallen | done_flipped | done_pitch | done_roll | done_timeout).astype(bool)
