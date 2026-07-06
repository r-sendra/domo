"""Pure-math checks for domo.utils.rotations (wxyz convention)."""

import math

import pytest
import torch

from domo.utils.rotations import (
    identity_quat,
    normalize_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_to_euler_xyz,
    quat_to_rpy,
)

torch.manual_seed(0)


def _rand_quat(n=256):
    return normalize_quat(torch.randn(n, 4, dtype=torch.float64))


def test_identity_apply():
    q = identity_quat(8, dtype=torch.float64)
    v = torch.randn(8, 3, dtype=torch.float64)
    assert torch.allclose(quat_apply(q, v), v)


def test_apply_inverse_roundtrip():
    q = _rand_quat()
    v = torch.randn(256, 3, dtype=torch.float64)
    assert torch.allclose(quat_apply_inverse(q, quat_apply(q, v)), v, atol=1e-10)


def test_mul_composition():
    u, v = _rand_quat(), _rand_quat()
    x = torch.randn(256, 3, dtype=torch.float64)
    lhs = quat_apply(quat_mul(u, v), x)
    rhs = quat_apply(u, quat_apply(v, x))
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_conjugate_is_inverse():
    q = _rand_quat()
    qq = quat_mul(q, quat_conjugate(q))
    ident = identity_quat(q.shape[0], dtype=torch.float64)
    assert torch.allclose(qq, ident, atol=1e-10)


def test_yaw_quat_euler():
    # 90° rotation about z: q = (cos45, 0, 0, sin45)
    h = math.sqrt(0.5)
    q = torch.tensor([[h, 0.0, 0.0, h]], dtype=torch.float64)
    for fn in (quat_to_euler_xyz, quat_to_rpy):
        rpy = fn(q)[0]
        assert abs(rpy[0].item()) < 1e-9
        assert abs(rpy[1].item()) < 1e-9
        assert abs(rpy[2].item() - math.pi / 2) < 1e-9


def test_euler_matches_genesis():
    genesis_geom = pytest.importorskip("genesis.utils.geom")
    q = _rand_quat().float()
    ref = genesis_geom._tc_quat_to_xyz(q, 1e-8, False)
    assert (ref - quat_to_euler_xyz(q)).abs().max().item() < 1e-4

    v = torch.randn(256, 3)
    assert (genesis_geom.transform_by_quat(v, q) - quat_apply(q, v)).abs().max() < 1e-4
    u = _rand_quat().float()
    assert (genesis_geom.transform_quat_by_quat(q, u) - quat_mul(u, q)).abs().max() < 1e-4
