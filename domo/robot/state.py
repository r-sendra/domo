"""
RobotState: the per-step snapshot of robot kinematic state.

Filled in by the robot's sensors on `Robot.refresh()` and read by
controllers, observation builders, and reward functions. All tensors are
[n_envs, ...] on the sim device and are updated IN PLACE (`tensor[:] = ...`),
so a consumer may keep a reference to a field across steps. On the real
robot the same structure is filled from the state-estimation stack instead
of the simulator — consumers never know the difference.

Frames: `base_*_world` fields are world frame; the others are body frame.
`base_euler` uses the intrinsic x-y-z convention of
`domo.utils.rotations.quat_to_euler_xyz` (matches Genesis' default), NOT
aerospace roll-pitch-yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

__all__ = ["RobotState"]


@dataclass
class RobotState:
    # World frame (privileged in sim; from state estimator / mocap on real).
    base_pos: torch.Tensor          # [N, 3]
    base_quat: torch.Tensor         # [N, 4] wxyz
    base_lin_vel_world: torch.Tensor  # [N, 3]

    # Body frame (available on the real robot from IMU + estimator).
    base_lin_vel: torch.Tensor      # [N, 3]
    base_ang_vel: torch.Tensor      # [N, 3]
    projected_gravity: torch.Tensor  # [N, 3] unit gravity in body frame
    base_euler: torch.Tensor        # [N, 3] roll/pitch/yaw (intrinsic xyz)

    # Joints (from encoders), canonical joint order.
    dof_pos: torch.Tensor           # [N, D]
    dof_vel: torch.Tensor           # [N, D]

    # Feet (force-based if the backend supports it, else task-provided proxy).
    foot_contacts: torch.Tensor     # [N, n_feet] float 0/1

    @classmethod
    def zeros(cls, n_envs: int, n_dofs: int, n_feet: int,
              device: torch.device, dtype=torch.float32) -> RobotState:
        """Allocate an all-zero state with identity quaternions."""
        def z(*shape):
            return torch.zeros((n_envs, *shape), device=device, dtype=dtype)
        state = cls(
            base_pos=z(3), base_quat=z(4), base_lin_vel_world=z(3),
            base_lin_vel=z(3), base_ang_vel=z(3),
            projected_gravity=z(3), base_euler=z(3),
            dof_pos=z(n_dofs), dof_vel=z(n_dofs),
            foot_contacts=z(n_feet),
        )
        state.base_quat[:, 0] = 1.0
        return state

    def zero_idx(self, envs_idx: torch.Tensor) -> None:
        """Zero all fields for the given envs (identity quaternion)."""
        for f in fields(self):
            getattr(self, f.name)[envs_idx] = 0.0
        self.base_quat[envs_idx, 0] = 1.0
