"""
Sensors: the only components (besides actuators) that talk to the physics
backend for robot state.

Two families:

* `StateSensor.update(state)` — writes its measurements into the shared
  `RobotState`. `Robot.refresh()` calls them in a fixed order (IMU first,
  since body-frame quantities need the base quaternion).
* `ExteroceptiveSensor.read()` — returns its own tensor (lidar, cameras...).

Sim implementations query a `domo.sim.Articulation` handle. Real-robot
implementations of the same interfaces will read DDS/ROS topics instead;
tasks and controllers cannot tell them apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from domo.sim.base import Articulation, LidarSensorHandle
from domo.utils.rotations import quat_apply_inverse, quat_to_euler_xyz

from .state import RobotState

__all__ = [
    "StateSensor",
    "ExteroceptiveSensor",
    "SimIMU",
    "SimJointEncoders",
    "SimBaseStateSensor",
    "SimContactSensor",
    "SectorLidar",
]


class StateSensor(ABC):
    @abstractmethod
    def update(self, state: RobotState) -> None: ...


class ExteroceptiveSensor(ABC):
    @abstractmethod
    def read(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Simulated proprioception
# ---------------------------------------------------------------------------

class SimIMU(StateSensor):
    """
    Base orientation, body-frame angular velocity, projected gravity and
    Euler angles — the quantities a real IMU (+ orientation filter) provides.
    """

    def __init__(self, articulation: Articulation, device: torch.device):
        self._art = articulation
        self._gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=device)

    def update(self, state: RobotState) -> None:
        quat = self._art.get_base_quaternion()
        state.base_quat[:] = quat
        state.base_ang_vel[:] = quat_apply_inverse(
            quat, self._art.get_base_angular_velocity())
        state.projected_gravity[:] = quat_apply_inverse(
            quat, self._gravity_dir.expand_as(state.projected_gravity))
        state.base_euler[:] = quat_to_euler_xyz(quat)


class SimJointEncoders(StateSensor):
    """Joint positions and velocities in the spec's canonical joint order."""

    def __init__(self, articulation: Articulation, dof_idx: Sequence[int]):
        self._art = articulation
        self._dof_idx = dof_idx

    def update(self, state: RobotState) -> None:
        state.dof_pos[:] = self._art.get_joint_positions(self._dof_idx)
        state.dof_vel[:] = self._art.get_joint_velocities(self._dof_idx)


class SimBaseStateSensor(StateSensor):
    """
    Privileged base state: world position and linear velocity (world and
    body frame). On the real robot this comes from the state estimator and
    is less reliable — tasks intended for transfer should prefer IMU/encoder
    observations. Must run AFTER SimIMU (uses state.base_quat).
    """

    def __init__(self, articulation: Articulation):
        self._art = articulation

    def update(self, state: RobotState) -> None:
        state.base_pos[:] = self._art.get_base_position()
        state.base_lin_vel_world[:] = self._art.get_base_linear_velocity()
        state.base_lin_vel[:] = quat_apply_inverse(
            state.base_quat, state.base_lin_vel_world)


class SimContactSensor(StateSensor):
    """
    Force-based foot contact flags. If the backend does not expose contact
    forces, `available` is False and the sensor writes nothing — a task may
    then substitute a proxy (e.g. CPG stance phase).
    """

    def __init__(self, articulation: Articulation, foot_link_idx: Sequence[int],
                 force_threshold: float = 1.0):
        self._art = articulation
        self._foot_link_idx = list(foot_link_idx)
        self._threshold = force_threshold
        try:
            self._art.get_link_contact_forces()
            self.available = True
        except Exception:
            self.available = False

    def update(self, state: RobotState) -> None:
        if not self.available:
            return
        forces = self._art.get_link_contact_forces()[:, self._foot_link_idx]
        state.foot_contacts[:] = (forces.norm(dim=-1) > self._threshold).float()


# ---------------------------------------------------------------------------
# Exteroception
# ---------------------------------------------------------------------------

class SectorLidar(ExteroceptiveSensor):
    """
    2D sector lidar: [N, n_sectors] min distances around the base.
    `update_interval` emulates a sensor slower than the control loop —
    reads between refreshes return the cached scan (as on real hardware).
    """

    def __init__(self, handle: LidarSensorHandle, n_envs: int,
                 device: torch.device, update_interval: int = 1):
        self._handle = handle
        self._interval = max(1, update_interval)
        self._step_count = 0
        self.max_range = handle.config.max_range
        self._cache = torch.full(
            (n_envs, handle.config.n_horizontal), self.max_range, device=device)

    def tick(self) -> None:
        """Advance one control step; refresh the scan when due."""
        self._step_count += 1
        if self._step_count % self._interval == 0:
            self._cache = self._handle.read_sector_distances()

    @property
    def has_scan(self) -> bool:
        return self._step_count >= self._interval

    def read(self) -> torch.Tensor:
        return self._cache

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self._cache[envs_idx] = self.max_range
