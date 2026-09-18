"""
Analytic leg kinematics for 3-DOF quadruped legs (hip-abduction / thigh /
calf), validated against the Go2 URDF convention (round-trips to machine
epsilon — see tests/test_kinematics.py).

Frames: hip frame origin at the abduction joint, x forward, y left, z up.
Abduction rotates about +x; thigh and calf about +y. Angles returned and
consumed ARE the URDF joint values (no sign flips).

Pure torch — usable in sim, in training, and on the real robot.
Ported from scripts/house_scene/go2_cpg_rl.py.
"""

from __future__ import annotations

import torch

from domo.robot.spec import QuadrupedGeometry

__all__ = ["LegKinematics", "leg_fk", "leg_ik"]

# Floor on the squared in-plane leg length in IK: keeps sqrt finite when the
# foot target sits exactly on the abduction axis.
_LENGTH_SQ_EPS = 1e-8


def leg_fk(qh: torch.Tensor, qt: torch.Tensor, qc: torch.Tensor,
           side_sign: torch.Tensor, l1: float, l2: float, l3: float
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward kinematics: joint angles → foot position in the hip frame.

    Args:
        qh, qt, qc: hip-abduction, thigh, calf angles [N, 4] (rad).
        side_sign: [4] +1 for left legs, −1 for right (hip offset direction).
        l1, l2, l3: hip offset, thigh length, calf length (m).

    Returns:
        (px, py, pz) foot positions [N, 4] in each leg's hip frame.
    """
    s = side_sign
    z_s = -l2 * torch.cos(qt) - l3 * torch.cos(qt + qc)
    px = -l2 * torch.sin(qt) - l3 * torch.sin(qt + qc)
    py = s * l1 * torch.cos(qh) - z_s * torch.sin(qh)
    pz = s * l1 * torch.sin(qh) + z_s * torch.cos(qh)
    return px, py, pz


def leg_ik(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor,
           side_sign: torch.Tensor, l1: float, l2: float, l3: float
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form inverse kinematics: foot position in hip frame → angles.

    Inverse of `leg_fk` (same argument conventions, foot targets [N, 4] in).
    The knee is always the "backward" (negative qc) solution, as on the Go2;
    unreachable targets are clamped to the workspace boundary.

    Returns:
        (qh, qt, qc) joint angles [N, 4] (rad).
    """
    l1_s = side_sign * l1
    L = torch.sqrt(torch.clamp(py * py + pz * pz - l1 * l1, min=_LENGTH_SQ_EPS))
    z_s = -L
    qh = torch.atan2(L * py + l1_s * pz, l1_s * py - L * pz)
    D2 = px * px + L * L
    cos_knee = torch.clamp((D2 - l2 * l2 - l3 * l3) / (2.0 * l2 * l3), -1.0, 1.0)
    qc = -torch.acos(cos_knee)
    A = l2 + l3 * cos_knee
    B = l3 * torch.sin(qc)
    qt = torch.atan2(B * z_s - A * px, -B * px - A * z_s)
    return qh, qt, qc


class LegKinematics:
    """Geometry-bound convenience wrapper around leg_fk / leg_ik (all 4 legs)."""

    def __init__(self, geometry: QuadrupedGeometry, device: torch.device):
        self.geo = geometry
        self.side_sign = torch.tensor(geometry.side_sign, device=device,
                                      dtype=torch.float32)

    def fk(self, qh, qt, qc):
        """Joint angles [N, 4] ×3 → foot positions (px, py, pz) [N, 4]."""
        return leg_fk(qh, qt, qc, self.side_sign,
                      self.geo.l_hip, self.geo.l_thigh, self.geo.l_calf)

    def ik(self, px, py, pz):
        """Foot positions [N, 4] ×3 → joint angles (qh, qt, qc) [N, 4]."""
        return leg_ik(px, py, pz, self.side_sign,
                      self.geo.l_hip, self.geo.l_thigh, self.geo.l_calf)

    def self_test(self, default_dof_pos: torch.Tensor) -> float:
        """IK(FK(q)) round-trip error [rad] for a [12] joint vector."""
        qh = default_dof_pos[0::3].view(1, 4)
        qt = default_dof_pos[1::3].view(1, 4)
        qc = default_dof_pos[2::3].view(1, 4)
        fx, fy, fz = self.fk(qh, qt, qc)
        rh, rt, rc = self.ik(fx, fy, fz)
        return (torch.abs(rh - qh) + torch.abs(rt - qt)
                + torch.abs(rc - qc)).max().item()
