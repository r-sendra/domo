"""
Pure-torch quaternion / rotation utilities.

DOMO canonical convention: quaternions are **scalar-first [w, x, y, z]**,
unit-norm, batched on the leading dimensions. All functions are pure torch
(no physics-engine imports) so they can run in simulation, in training code,
or on the real robot identically.

Note: Genesis also uses the wxyz convention, so the Genesis backend can pass
quaternions through without conversion. Any future backend that uses xyzw
(e.g. MuJoCo bindings via some wrappers, Isaac Gym) must convert inside the
backend — the rest of the library never sees a non-wxyz quaternion.

Two Euler conventions are provided on purpose; do not mix them:
  * `quat_to_euler_xyz` — intrinsic X-Y-Z, matches Genesis
    `quat_to_xyz(..., rpy=False)` (its default), used by the locomotion
    tasks and `RobotState.base_euler`.
  * `quat_to_rpy` — aerospace roll-pitch-yaw (intrinsic Z-Y-X).
Both agree for pure single-axis rotations and differ in the cross terms.
All of these are verified against `genesis.utils.geom` in tests/test_rotations.py.
"""

from __future__ import annotations

import torch

__all__ = [
    "identity_quat",
    "normalize_quat",
    "quat_apply",
    "quat_apply_inverse",
    "quat_conjugate",
    "quat_inverse",
    "quat_mul",
    "quat_to_euler_xyz",
    "quat_to_rpy",
]


def identity_quat(n: int, device: torch.device | str | None = None,
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return [n, 4] identity quaternions (w=1)."""
    q = torch.zeros((n, 4), device=device, dtype=dtype)
    q[:, 0] = 1.0
    return q


def normalize_quat(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Unit-normalise along the last dim; `eps` guards against zero vectors."""
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """[w, -x, -y, -z]. Equals the inverse for unit quaternions."""
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion (== conjugate)."""
    return quat_conjugate(q)


def quat_mul(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product u ⊗ v (both wxyz). Composition: applying (u ⊗ v)
    rotates first by v, then by u — i.e. R_u @ R_v.
    """
    w1, x1, y1, z1 = u.unbind(-1)
    w2, x2, y2, z2 = v.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector(s) v by quaternion(s) q: v' = R(q) v.
    q: [..., 4] wxyz, v: [..., 3]. Broadcasts on leading dims.
    """
    # Rodrigues-style form: v + 2w (qv × v) + 2 qv × (qv × v), no matrix built.
    qw = q[..., 0:1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q (world → body for a body-orientation q)."""
    return quat_apply(quat_conjugate(q), v)


def quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """
    Convert wxyz quaternion(s) to **intrinsic x-y-z** Euler angles
    (R = Rx(roll) · Ry(pitch) · Rz(yaw)), in radians. Returns [..., 3].

    This matches Genesis `quat_to_xyz(..., rpy=False)` (its default), which
    the locomotion tasks use for roll/pitch termination checks. It is NOT the
    aerospace yaw-pitch-roll (ZYX) convention — see `quat_to_rpy` for that.
    """
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x - y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y + x * z)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    siny_cosp = 2.0 * (w * z - x * y)
    # atan2 form (instead of asin) stays well-conditioned near ±90° pitch.
    pitch = torch.atan2(sinp, torch.sqrt(cosy_cosp**2 + siny_cosp**2))
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack((roll, pitch, yaw), dim=-1)


def quat_to_rpy(q: torch.Tensor) -> torch.Tensor:
    """
    Convert wxyz quaternion(s) to aerospace roll-pitch-yaw (intrinsic Z-Y-X:
    R = Rz(yaw) · Ry(pitch) · Rx(roll)), in radians. Returns [..., 3].
    """
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)
