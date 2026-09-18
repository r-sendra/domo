"""
Actuators: the write-side counterpart of sensors.

All actuation commands to the physics backend go through an Actuator;
controllers and tasks never call the engine directly (`Robot.set_joint_targets`
is the only caller). A real-robot PD actuator will implement the same
interface on top of the Unitree low-level command topic, so the seam between
sim and hardware is exactly this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from domo.sim.base import Articulation

__all__ = ["Actuator", "PDJointPositionActuator"]


class Actuator(ABC):
    """One control-step command sink."""

    @abstractmethod
    def apply(self, command: torch.Tensor) -> None:
        """Send one control-step command (shape [n_envs, n_dofs])."""


class PDJointPositionActuator(Actuator):
    """
    Joint-space PD position control (gains live in the engine/firmware).
    `apply(targets)` sets desired joint positions in canonical joint order.

    The gains are written to the engine once at construction (shared by all
    envs); per-env variation is a domain-randomization capability, see
    `Articulation.set_pd_gains_scaled`. `kp`/`kd` stay readable so DR can
    scale the nominal values.
    """

    def __init__(self, articulation: Articulation, dof_idx: Sequence[int],
                 kp: float, kd: float):
        self._art = articulation
        self._dof_idx = dof_idx
        self.kp = kp
        self.kd = kd
        n = len(dof_idx)
        self._art.set_pd_gains([kp] * n, [kd] * n, dof_idx)

    def apply(self, command: torch.Tensor) -> None:
        self._art.set_joint_position_targets(command, self._dof_idx)
