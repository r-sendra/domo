"""
Robot specifications: static, engine-independent descriptions of a robot.

A `RobotSpec` carries everything needed to instantiate the robot in any
simulator (or to sanity-check the real one): asset path, joint ordering,
default configuration, nominal PD gains, and initial pose. Kinematic
parameters used by analytic controllers live in `QuadrupedGeometry`.

Both dataclasses are frozen: a spec is an immutable constant (see
`domo.robot.go2.GO2`), and per-run overrides (gains, spawn pose) are
passed to `Robot` instead of mutating the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["QuadrupedGeometry", "RobotSpec"]


@dataclass(frozen=True)
class QuadrupedGeometry:
    """
    Per-leg kinematics for a 3-DOF-per-leg quadruped (hip/thigh/calf).

    Lengths in metres. `side_sign` is the lateral sign per leg, ordered like
    the leg blocks in `RobotSpec.joint_names` (+1 left legs, −1 right legs);
    the analytic IK/FK in `domo.control.kinematics` mirrors the hip offset
    with it.
    """
    l_hip: float
    l_thigh: float
    l_calf: float
    side_sign: tuple[float, ...] = (-1.0, +1.0, -1.0, +1.0)


@dataclass(frozen=True)
class RobotSpec:
    """
    Static robot description.

    Attributes:
        name: short identifier (checkpoint metadata, logging).
        urdf_path: asset path as understood by the backend (Genesis resolves
            relative paths against its bundled assets).
        joint_names: actuated joints in canonical order; every dof-indexed
            tensor in DOMO follows this ordering.
        default_joint_angles: nominal standing pose (rad) keyed by joint name.
        foot_link_names: links probed for force-based contact; may be empty or
            absent from the asset (see `Robot.bind`).
        base_init_pos / base_init_quat: spawn pose (m, wxyz quaternion).
        kp / kd: nominal joint PD gains (a task may override via its config).
        geometry: leg kinematics for analytic controllers, if any.
    """
    name: str
    urdf_path: str
    joint_names: tuple[str, ...]
    default_joint_angles: dict[str, float]
    foot_link_names: tuple[str, ...] = ()
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.4)
    base_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
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
