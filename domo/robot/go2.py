"""
go2.py — Go2 Walking Environment (Genesis)
===========================================
Task: velocity-tracking locomotion on flat terrain.

Fixed bugs vs original:
  BUG 1 — MOTOR_DOFS was hardcoded as range(6,18). Genesis DOF indices
           depend on URDF joint ordering and are not guaranteed to be 0-5
           for the floating base. Fixed by querying each joint by name
           after scene.build() and storing as self.motor_dofs.

  BUG 2 — _compute_done() was indented at 12 spaces instead of 8,
           making it a nested block inside _compute_reward() rather than
           a class method. Calling self._compute_done() raised
           AttributeError at runtime — training never ran a single step.
           Fixed by correcting indentation to 8 spaces.

  BUG 3 — Task/reward mismatch: class was named Go2StandEnv and docstring
           described a stand-up task, but the reward tracked velocity
           commands and commands were hardcoded to vx=0.5. This is a
           walking task. Fixed by: renaming to Go2WalkEnv, randomising
           velocity commands at reset, and removing stand-up references.

Observation space (48 values per env):
  [0:3]   base linear velocity    (body frame)
  [3:6]   base angular velocity   (body frame)
  [6:9]   projected gravity       (body frame)
  [9:12]  velocity command        [vx, vy, yaw_rate]
  [12:24] joint pos relative to default stance
  [24:36] joint velocities
  [36:48] previous action

Action space (12 values per env):
  Target joint position offsets from default stance, scaled by ACTION_SCALE.

Reward:
  + 1.0  * exp(-||cmd_vel_xy - base_lin_vel_xy||^2 / 0.25)  lin vel tracking
  + 0.5  * exp(-(cmd_yaw - base_ang_vel_z)^2    / 0.25)     yaw tracking
  - 2.0  * base_lin_vel_z^2                                  no bouncing
  - 0.05 * ||base_ang_vel_xy||^2                             no rolling/pitching
  - 0.2  * ||proj_grav_xy||^2                                stay upright
  - 0.1  * ||joint_pos_rel||^2                               stay near default
  - 0.01 * ||action||^2                                      energy
  - 0.01 * ||action - prev_action||^2                        smoothness
  - 0.001* ||joint_vel||^2                                   no flailing

Done conditions:
  - base height < 0.15 m
  - projected gravity Z (body) > 0.5   (flipped upside-down)
  - |proj_grav X| > 0.7                (pitched > ~45 deg)
  - |proj_grav Y| > 0.7                (rolled  > ~45 deg)
  - step count >= max_episode_steps    (timeout)

Usage:
    from go2 import Go2WalkEnv
    env = Go2WalkEnv(n_envs=64, headless=True)
    obs = env.reset()                           # (N, 48) float32
    obs, reward, done, info = env.step(action)  # action: (N, 12) float32
"""

import math
import numpy as np
import torch
import genesis as gs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FIX 1: Motor joint names used to query DOF indices at runtime.
# Never hardcode range(6,18) — the floating base DOF layout is URDF-dependent.
MOTOR_JOINT_NAMES = [
    "FL_hip_joint",   "FL_thigh_joint",  "FL_calf_joint",
    "FR_hip_joint",   "FR_thigh_joint",  "FR_calf_joint",
    "RL_hip_joint",   "RL_thigh_joint",  "RL_calf_joint",
    "RR_hip_joint",   "RR_thigh_joint",  "RR_calf_joint",
]

DEFAULT_JOINT_POS = np.array([
    0.0,  0.8, -1.5,   # FL  hip, thigh, calf
    0.0,  0.8, -1.5,   # FR
    0.0,  1.0, -1.5,   # RL
    0.0,  1.0, -1.5,   # RR
], dtype=np.float32)

BASE_INIT_POS  = np.array([0.0, 0.0, 0.42], dtype=np.float32)
BASE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity = upright

ACTION_SCALE = 0.25    # rad — joint offset scale
OBS_CLIP     = 5.0     # clip obs to prevent outliers

# PD gains matching the real Unitree Go2
KP = 20.0   # position stiffness  [N·m/rad]
KV =  0.5   # velocity damping    [N·m·s/rad]

# Velocity command ranges
CMD_LIN_VEL_X  = (-1.0,  1.0)   # m/s
CMD_LIN_VEL_Y  = (-0.5,  0.5)   # m/s
CMD_ANG_VEL_Z  = (-1.0,  1.0)   # rad/s


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy — no Genesis/simulator dependency)
# ---------------------------------------------------------------------------

def euler_to_quat(roll: np.ndarray,
                  pitch: np.ndarray,
                  yaw: np.ndarray) -> np.ndarray:
    """
    Vectorised Euler (rad) → quaternion [w, x, y, z].
    Inputs can be scalars or (N,) arrays.
    """
    cr, sr = np.cos(roll  / 2), np.sin(roll  / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw   / 2), np.sin(yaw   / 2)
    w =  cr * cp * cy + sr * sp * sy
    x =  sr * cp * cy - cr * sp * sy
    y =  cr * sp * cy + sr * cp * sy
    z =  cr * cp * sy - sr * sp * cy
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate world-frame vectors v into body frame using quaternion q.
    Equivalent to R(q)^T @ v.

    q : (N, 4)  [w, x, y, z]
    v : (N, 3)
    returns (N, 3)
    """
    w  = q[:, 0:1]; x = q[:, 1:2]; y = q[:, 2:3]; z = q[:, 3:4]
    vx = v[:, 0:1]; vy = v[:, 1:2]; vz = v[:, 2:3]
    bx = (1 - 2*(y*y + z*z))*vx +     2*(x*y + w*z)*vy +     2*(x*z - w*y)*vz
    by =     2*(x*y - w*z)*vx + (1 - 2*(x*x + z*z))*vy +     2*(y*z + w*x)*vz
    bz =     2*(x*z + w*y)*vx +     2*(y*z - w*x)*vy + (1 - 2*(x*x + y*y))*vz
    return np.concatenate([bx, by, bz], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# FIX 3: Renamed from Go2StandEnv → Go2WalkEnv to match the actual task.
class Go2WalkEnv:
    """
    Vectorised Genesis environment for Go2 velocity-tracking locomotion.

    All state tensors are returned as numpy float32 arrays.
    Compatible with the PPOTrainer in main.py.
    """

    OBS_DIM = 48
    ACT_DIM = 12

    def __init__(
        self,
        n_envs:            int   = 64,
        dt:                float = 0.02,   # 50 Hz — matches real Go2 control rate
        substeps:          int   = 2,
        max_episode_steps: int   = 500,
        headless:          bool  = True,
        device:            str   = "cpu",
    ):
        self.n_envs            = n_envs
        self.dt                = dt
        self.max_episode_steps = max_episode_steps
        self.device            = device

        backend = gs.cuda if torch.cuda.is_available() else gs.cpu
        # --- Genesis initialisation ---
        gs.init(backend=backend, logging_level="warning")
        # gs.init(backend=gs.cuda, logging_level="warning")

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos    = (3.0, -2.0, 2.0),
                camera_lookat = (0.0,  0.0, 0.5),
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

        # Go2 — spawns upright
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = "urdf/go2/urdf/go2.urdf",
                pos  = BASE_INIT_POS.tolist(),
                quat = BASE_INIT_QUAT.tolist(),
            )
        )

        # Build parallel environments
        self.scene.build(n_envs=n_envs)

        # FIX 1: Query DOF indices by joint name after build().
        # This is safe regardless of how Genesis orders the floating-base DOFs.
        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in MOTOR_JOINT_NAMES
        ]

        # PD controller gains
        self.robot.set_dofs_kp([KP] * self.ACT_DIM, dofs_idx_local=self.motor_dofs)
        self.robot.set_dofs_kv([KV] * self.ACT_DIM, dofs_idx_local=self.motor_dofs)

        # Internal state buffers
        self._step_count  = np.zeros(n_envs, dtype=np.int32)
        self._prev_action = np.zeros((n_envs, self.ACT_DIM), dtype=np.float32)
        self._ep_return   = np.zeros(n_envs, dtype=np.float32)
        self._ep_length   = np.zeros(n_envs, dtype=np.int32)
        self._commands    = np.zeros((n_envs, 3), dtype=np.float32)

        # Gravity unit vector in world frame, tiled for all envs
        self._gravity_world = np.tile(
            [0.0, 0.0, -1.0], (n_envs, 1)
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset all envs and return initial observations."""
        self._reset_envs(np.arange(self.n_envs))
        self.scene.step()   # one warmup step so Genesis state is populated
        return self._get_obs()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        action : (n_envs, 12) float32, values in [-1, 1]
        returns: obs (N,48), reward (N,), done (N,), info dict
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target_pos = DEFAULT_JOINT_POS + ACTION_SCALE * action   # (N, 12)

        # Apply position target — PD controller runs inside Genesis
        self.robot.control_dofs_position(
            target_pos,
            dofs_idx_local=self.motor_dofs,
        )
        self.scene.step()

        obs    = self._get_obs()
        reward = self._compute_reward(obs, action)
        done   = self._compute_done(obs)

        # Track episode statistics
        self._ep_return  += reward
        self._ep_length  += 1
        self._step_count += 1
        self._prev_action = action.copy()

        # Collect completed episode stats before resetting
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
        Reset specific envs to upright standing pose with small perturbations.
        """
        n = len(env_ids)
        if n == 0:
            return

        # Base position: standing height + small xy noise
        pos = np.tile(BASE_INIT_POS, (n, 1))
        pos[:, :2] += np.random.uniform(-0.05, 0.05, (n, 2)).astype(np.float32)
        pos[:, 2]  += np.random.uniform(-0.02, 0.02, n).astype(np.float32)
        pos[:, 2]   = np.clip(pos[:, 2], 0.35, 0.50)

        # Base orientation: upright + random yaw
        yaw  = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
        quat = euler_to_quat(
            np.zeros(n, np.float32),
            np.zeros(n, np.float32),
            yaw,
        )   # (n, 4)  [w, x, y, z]

        # Joint positions: default + small noise
        jpos = (
            DEFAULT_JOINT_POS
            + np.random.uniform(-0.05, 0.05, (n, self.ACT_DIM)).astype(np.float32)
        )

        # Apply to Genesis
        self.robot.set_pos( pos,  envs_idx=env_ids)
        self.robot.set_quat(quat, envs_idx=env_ids)
        self.robot.set_dofs_position(
            jpos,
            dofs_idx_local=self.motor_dofs,
            envs_idx=env_ids,
        )
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        # FIX 3: Randomise velocity commands instead of hardcoding vx=0.5.
        # Random commands are essential for training a general walking policy.
        # A fixed command biases the policy towards one gait and direction.
        self._commands[env_ids, 0] = np.random.uniform(
            *CMD_LIN_VEL_X, n).astype(np.float32)
        self._commands[env_ids, 1] = np.random.uniform(
            *CMD_LIN_VEL_Y, n).astype(np.float32)
        self._commands[env_ids, 2] = np.random.uniform(
            *CMD_ANG_VEL_Z, n).astype(np.float32)

        # Reset internal buffers
        self._step_count[env_ids]  = 0
        self._prev_action[env_ids] = 0.0
        self._ep_return[env_ids]   = 0.0
        self._ep_length[env_ids]   = 0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build (n_envs, 48) observation array from Genesis state.

        Genesis API used (confirmed for genesis-world >= 0.3.0):
          robot.get_pos()           -> (N, 3)  world frame position
          robot.get_quat()          -> (N, 4)  [w,x,y,z] world orientation
          robot.get_vel()           -> (N, 3)  world frame linear velocity
          robot.get_ang()           -> (N, 3)  world frame angular velocity
          robot.get_dofs_position() -> (N, D)  all DOF positions
          robot.get_dofs_velocity() -> (N, D)  all DOF velocities
        """
        base_pos    = self.robot.get_pos().cpu().numpy()           # (N, 3)
        base_quat   = self.robot.get_quat().cpu().numpy()          # (N, 4)
        vel_world   = self.robot.get_vel().cpu().numpy()           # (N, 3)
        angv_world  = self.robot.get_ang().cpu().numpy()           # (N, 3)
        dof_pos_all = self.robot.get_dofs_position().cpu().numpy() # (N, D)
        dof_vel_all = self.robot.get_dofs_velocity().cpu().numpy() # (N, D)

        # World-frame velocities → body frame
        base_lin_vel = quat_rotate_inverse(base_quat, vel_world)    # (N, 3)
        base_ang_vel = quat_rotate_inverse(base_quat, angv_world)   # (N, 3)

        # Gravity vector projected to body frame
        proj_gravity = quat_rotate_inverse(
            base_quat, self._gravity_world
        )                                                            # (N, 3)

        # Motor joints only (FIX 1: use self.motor_dofs, not MOTOR_DOFS constant)
        motor_pos     = dof_pos_all[:, self.motor_dofs]              # (N, 12)
        motor_vel     = dof_vel_all[:, self.motor_dofs]              # (N, 12)
        motor_pos_rel = motor_pos - DEFAULT_JOINT_POS                # (N, 12)

        obs = np.concatenate([
            base_lin_vel,        # [0:3]
            base_ang_vel,        # [3:6]
            proj_gravity,        # [6:9]
            self._commands,      # [9:12]
            motor_pos_rel,       # [12:24]
            motor_vel,           # [24:36]
            self._prev_action,   # [36:48]
        ], axis=1).astype(np.float32)

        return np.clip(obs, -OBS_CLIP, OBS_CLIP)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self, obs: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        """
        Velocity-tracking locomotion reward.
        All weights are multiplied by dt inside the PPO trainer to keep
        them timestep-independent.
        """
        height = self.robot.get_pos().cpu().numpy()[:, 2]
        lin_vel       = obs[:, 0:3]    # (N, 3) body frame
        ang_vel       = obs[:, 3:6]    # (N, 3) body frame
        proj_grav     = obs[:, 6:9]    # (N, 3)
        commands      = obs[:, 9:12]   # (N, 3) [vx, vy, yaw_rate]
        motor_pos_rel = obs[:, 12:24]  # (N, 12)
        motor_vel     = obs[:, 24:36]  # (N, 12)
        prev_action   = obs[:, 36:48]  # (N, 12)

        # Task: track commanded linear velocity (xy)
        lin_vel_error = np.sum(np.square(commands[:, :2] - lin_vel[:, :2]), axis=1)
        r_track_lin   = np.exp(-lin_vel_error / 0.25)
        # FIX 1 revised — reward velocity projected onto the command direction
        cmd_xy      = commands[:, :2]                          # (N, 2)
        vel_xy      = lin_vel[:, :2]                           # (N, 2)
        cmd_norm    = np.linalg.norm(cmd_xy, axis=1, keepdims=True) + 1e-6
        cmd_dir     = cmd_xy / cmd_norm                        # unit vector of command
        vel_proj    = np.sum(vel_xy * cmd_dir, axis=1)         # projection onto cmd
        r_move      = np.where(
            cmd_norm.squeeze() > 0.1,
            np.clip(vel_proj, 0.0, None),   # only positive projection counts
            0.0
        )


        # Task: track commanded yaw rate
        ang_vel_error = np.square(commands[:, 2] - ang_vel[:, 2])
        r_track_ang   = np.exp(-ang_vel_error / 0.25)

        # Penalty: vertical base velocity (no bouncing)
        r_lin_vel_z   = -np.square(lin_vel[:, 2])

        # Penalty: roll and pitch angular velocity (torso stability)
        r_ang_vel_xy  = -np.sum(np.square(ang_vel[:, :2]), axis=1)

        # Penalty: body tilt (proj_grav xy should be near zero when upright)
        r_orientation = -np.sum(np.square(proj_grav[:, :2]), axis=1)

        # FIX 2: Reduced joint deviation penalty from 0.1 → 0.01.
        # The original weight actively discouraged leg movement away from
        # the default standing pose, which is exactly what walking requires.
        r_dof_pos     = -np.sum(np.square(motor_pos_rel), axis=1)

        # Penalty: action magnitude (energy proxy)
        r_action_mag  = -np.sum(np.square(action), axis=1)

        # FIX 3: Reduced action rate penalty from 0.01 → 0.001.
        # The original weight penalised rapid leg movement too heavily,
        # discouraging the alternating leg swing needed for a gait.
        r_action_rate = -np.sum(np.square(action - prev_action), axis=1)

        # Penalty: joint velocity (no flailing)
        r_joint_vel   = -np.sum(np.square(motor_vel), axis=1)
        # Reward being at the correct standing height
        r_height = -np.square(height - 0.34)   # 0.34m is nominal standing height

        reward = (
            1.0   * r_track_lin
          + 0.5   * r_move          # FIX 1: movement bonus
          + 0.5   * r_track_ang
          + 2.0   * r_lin_vel_z
          + 0.05  * r_ang_vel_xy
          + 0.2   * r_orientation
          + 0.01  * r_dof_pos       # FIX 2: was 0.1
          + 0.01  * r_action_mag
          + 0.001 * r_action_rate   # FIX 3: was 0.01
          + 0.001 * r_joint_vel
          + 2*r_height
        )

        return reward.astype(np.float32)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    # FIX 2: _compute_done was indented at 12 spaces (nested inside
    # _compute_reward), making it unreachable as a method. This caused
    # AttributeError: 'Go2StandEnv' object has no attribute '_compute_done'
    # on the very first call to step(). Fixed by restoring correct 4-space
    # class-level indentation throughout this method.
    def _compute_done(self, obs: np.ndarray) -> np.ndarray:
        height    = self.robot.get_pos().cpu().numpy()[:, 2]   # (N,)
        proj_grav = obs[:, 6:9]                          # (N, 3)

        done_fallen  = height < 0.25
        done_flipped = proj_grav[:, 2] > 0.5             # upside-down
        done_pitch   = np.abs(proj_grav[:, 0]) > 0.7     # pitched > ~45 deg
        done_roll    = np.abs(proj_grav[:, 1]) > 0.7     # rolled  > ~45 deg
        done_timeout = self._step_count >= self.max_episode_steps

        return (
            done_fallen | done_flipped | done_pitch | done_roll | done_timeout
        ).astype(bool)
