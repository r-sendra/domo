"""
Robot specifications: static, engine-independent descriptions of a robot.

A `RobotSpec` carries everything needed to instantiate the robot in any
simulator (or to sanity-check the real one): asset path, joint ordering,
default configuration, nominal PD gains, and initial pose. Kinematic
parameters used by analytic controllers live in `QuadrupedGeometry`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["QuadrupedGeometry", "RobotSpec"]


@dataclass(frozen=True)
class QuadrupedGeometry:
    """Per-leg kinematics for a 3-DOF-per-leg quadruped (hip/thigh/calf)."""
    l_hip: float
    l_thigh: float
    l_calf: float
    # Lateral sign per leg, ordered like the leg blocks in joint_names
    # (+1 left legs, -1 right legs).
    side_sign: tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf_path: str
    # Actuated joints in canonical order; every dof-indexed tensor in DOMO
    # follows this ordering.
    joint_names: tuple[str, ...]
    default_joint_angles: dict[str, float]
    foot_link_names: tuple[str, ...] = ()
    # Initial base pose (wxyz quaternion).
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.4)
    base_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Nominal joint PD gains (a task may override via its ActuatorConfig).
    kp: float = 20.0
    kd: float = 0.5
    geometry: QuadrupedGeometry | None = None

    @property
    def num_dofs(self) -> int:
        return len(self.joint_names)

    @property
    def default_dof_angles(self) -> tuple[float, ...]:
        """Default joint angles in joint_names order."""
        return tuple(self.default_joint_angles[n] for n in self.joint_names)
