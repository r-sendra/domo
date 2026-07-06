"""IK/FK round-trip and CPG sanity checks (engine-free)."""

import math

import torch

from domo.control import CPGConfig, CPGLegController, CPGOscillators, LegKinematics
from domo.robot import GO2, GO2_GEOMETRY

DEVICE = torch.device("cpu")


def test_ik_fk_roundtrip_default_pose():
    kin = LegKinematics(GO2_GEOMETRY, DEVICE)
    q = torch.tensor(GO2.default_dof_angles)
    err = kin.self_test(q)
    assert err < 1e-4, f"IK(FK(default)) error {err:.2e} rad"


def test_ik_fk_roundtrip_random_poses():
    kin = LegKinematics(GO2_GEOMETRY, DEVICE)
    torch.manual_seed(0)
    n = 512
    qh = 0.3 * (torch.rand(n, 4) - 0.5)
    qt = 0.6 + 0.6 * torch.rand(n, 4)
    qc = -1.8 + 0.6 * torch.rand(n, 4)
    px, py, pz = kin.fk(qh, qt, qc)
    rh, rt, rc = kin.ik(px, py, pz)
    err = ((rh - qh).abs() + (rt - qt).abs() + (rc - qc).abs()).max().item()
    assert err < 1e-3, f"random-pose round-trip error {err:.2e} rad"


def test_cpg_converges_to_trot():
    cfg = CPGConfig()
    osc = CPGOscillators(cfg, n_envs=4, device=DEVICE)
    osc.reset_idx(torch.arange(4))
    mu = torch.full((4, 4), 1.5)
    omega = torch.full((4, 4), 2.5)
    psi = torch.zeros(4, 4)
    for _ in range(200):  # 4 s at 50 Hz control
        osc.step(mu, omega, psi, control_dt=0.02)

    # Amplitude converges to mu
    assert (osc.r - mu).abs().max().item() < 0.05

    # Kuramoto coupling holds the trot: diagonal pairs in phase, others π
    d = osc.theta[:, 0] - osc.theta[:, 3]
    d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
    assert d.abs().max().item() < 0.2, "FR/RL should be in phase"
    d = osc.theta[:, 0] - osc.theta[:, 1]
    d = torch.remainder(d, 2 * math.pi)
    assert (d - math.pi).abs().max().item() < 0.2, "FR/FL should be antiphase"


def test_cpg_controller_targets_near_default():
    kin = LegKinematics(GO2_GEOMETRY, DEVICE)
    ctrl = CPGLegController(CPGConfig(), kin, n_envs=2, device=DEVICE)
    ctrl.reset_idx(torch.arange(2))
    targets = ctrl.joint_targets(torch.zeros(2, 12), control_dt=0.02)
    assert targets.shape == (2, 12)
    assert torch.isfinite(targets).all()
    # Thigh/calf targets should be in a plausible stance range
    assert (targets[:, 1::3] > 0.0).all() and (targets[:, 1::3] < 1.5).all()
    assert (targets[:, 2::3] < -0.5).all() and (targets[:, 2::3] > -2.5).all()
