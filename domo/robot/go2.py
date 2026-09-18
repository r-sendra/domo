"""
Unitree Go2 robot specification.

Static description only — the walking task lives in `domo.tasks.go2_walk`.

Joint order is [FR, FL, RR, RL] × [hip, thigh, calf] and is the canonical
ordering for every dof-indexed tensor in the library.

Note: base_init_quat is the true wxyz identity (1,0,0,0). The legacy env
used (0,0,0,1) believing Genesis was xyzw; in wxyz that is a 180° yaw flip —
harmless (all observations are body-frame) but corrected here.

Note: the go2 URDF bundled with Genesis merges the fixed foot links, so the
`foot_link_names` below do not resolve there — `Robot.bind` then leaves
`contact_sensor` None and tasks fall back to a stance-phase proxy.
"""

from __future__ import annotations

from .spec import QuadrupedGeometry, RobotSpec

__all__ = ["GO2", "GO2_GEOMETRY"]

# Leg geometry from the go2_description URDF: hip→thigh lateral offset
# 0.0955 m, thigh and calf links 0.213 m. side_sign: right legs −y, left +y,
# per-leg order [FR, FL, RR, RL].
GO2_GEOMETRY = QuadrupedGeometry(
    l_hip=0.0955,
    l_thigh=0.213,
    l_calf=0.213,
    side_sign=(-1.0, +1.0, -1.0, +1.0),
)

GO2 = RobotSpec(
    name="go2",
    urdf_path="urdf/go2/urdf/go2.urdf",   # bundled with Genesis assets
    joint_names=(
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ),
    # Nominal standing pose (rad); rear thighs sit slightly higher.
    default_joint_angles={
        "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    },
    foot_link_names=("FR_foot", "FL_foot", "RR_foot", "RL_foot"),
    base_init_pos=(0.0, 0.0, 0.42),
    base_init_quat=(1.0, 0.0, 0.0, 0.0),
    kp=20.0,
    kd=0.5,
    geometry=GO2_GEOMETRY,
)
