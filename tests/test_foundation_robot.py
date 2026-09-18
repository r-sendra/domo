"""
Engine-free checks for domo.robot using a fake Articulation that implements
the domo.sim contract in pure torch: Robot lifecycle, sensor frames, resets,
RobotState helpers and DomainRandomization.
"""

import logging
import math

import pytest
import torch

from domo.robot import GO2, Robot, RobotState
from domo.robot.randomization import DomainRandomization
from domo.sim.base import Articulation, Scene
from domo.utils.rotations import quat_apply

N_ENVS = 3
D = GO2.num_dofs


class FakeArticulation(Articulation):
    """Minimal in-memory articulation; DOF i of joint i, links unknown."""

    def __init__(self, n_envs, n_dofs, has_contacts=False):
        self.n_envs = n_envs
        self.pos = torch.zeros(n_envs, 3)
        self.quat = torch.zeros(n_envs, 4)
        self.quat[:, 0] = 1.0
        self.lin_vel = torch.zeros(n_envs, 3)
        self.ang_vel = torch.zeros(n_envs, 3)
        self.dof_pos = torch.zeros(n_envs, n_dofs)
        self.dof_vel = torch.zeros(n_envs, n_dofs)
        self.kp = self.kd = None
        self.targets = None
        self.has_contacts = has_contacts
        self.velocities_zeroed_for = None

    def dof_indices(self, joint_names):
        return list(range(len(joint_names)))

    def link_indices(self, link_names):
        raise KeyError(f"Link not found: {link_names[0]}")   # merged foot links

    def get_base_position(self):
        return self.pos.clone()

    def get_base_quaternion(self):
        return self.quat.clone()

    def get_base_linear_velocity(self):
        return self.lin_vel.clone()

    def get_base_angular_velocity(self):
        return self.ang_vel.clone()

    def get_joint_positions(self, dof_idx):
        return self.dof_pos[:, list(dof_idx)]

    def get_joint_velocities(self, dof_idx):
        return self.dof_vel[:, list(dof_idx)]

    def get_link_contact_forces(self):
        if not self.has_contacts:
            raise NotImplementedError
        return torch.zeros(self.n_envs, 1, 3)

    def set_pd_gains(self, kp, kd, dof_idx):
        self.kp, self.kd = list(kp), list(kd)

    def set_joint_position_targets(self, targets, dof_idx):
        self.targets = targets.clone()

    def set_base_pose(self, pos, quat, envs_idx):
        self.pos[envs_idx] = pos
        self.quat[envs_idx] = quat

    def set_joint_positions(self, positions, dof_idx, envs_idx, zero_velocity=True):
        self.dof_pos[envs_idx[:, None], torch.tensor(list(dof_idx))] = positions
        if zero_velocity:
            self.dof_vel[envs_idx] = 0.0

    def zero_all_velocities(self, envs_idx):
        self.velocities_zeroed_for = envs_idx.clone()
        self.lin_vel[envs_idx] = 0.0
        self.ang_vel[envs_idx] = 0.0


class FakeScene(Scene):
    """Records what was added; only add_articulation returns a live handle."""

    def __init__(self):
        self.n_envs = 0
        self.articulation = None
        self.urdf = None
        self.built = False

    def add_ground(self, height=0.0): ...
    def add_terrain(self, cfg): ...
    def add_mesh(self, *a, **k): ...
    def add_urdf_prop(self, *a, **k): ...
    def add_box(self, *a, **k): ...
    def add_cylinder(self, *a, **k): ...
    def add_sphere(self, *a, **k): ...
    def add_lidar(self, articulation, cfg): ...

    def add_articulation(self, urdf_path, pos, quat_wxyz):
        self.urdf = urdf_path
        self.articulation = FakeArticulation(N_ENVS, D)
        return self.articulation

    def build(self, n_envs):
        self.built = True
        self.n_envs = n_envs

    def step(self): ...


def _robot(**kw):
    scene = FakeScene()
    robot = Robot(GO2, scene, torch.device("cpu"), **kw)
    scene.build(N_ENVS)
    robot.bind(N_ENVS)
    return robot, scene.articulation


def _yaw_quat(yaw):
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------------------
# Robot lifecycle
# ---------------------------------------------------------------------------

def test_bind_resolves_structure_and_gains():
    robot, art = _robot(kp=55.0, kd=1.5)
    assert robot.n_envs == N_ENVS
    assert robot.dof_idx == list(range(D))
    assert torch.allclose(robot.default_dof_pos, torch.tensor(GO2.default_dof_angles))
    assert art.kp == [55.0] * D and art.kd == [1.5] * D
    assert robot.state.dof_pos.shape == (N_ENVS, D)
    assert robot.state.foot_contacts.shape == (N_ENVS, len(GO2.foot_link_names))


def test_spec_gains_used_when_not_overridden():
    _, art = _robot()
    assert art.kp == [GO2.kp] * D and art.kd == [GO2.kd] * D


def test_missing_foot_links_leave_contact_sensor_none(caplog):
    with caplog.at_level(logging.INFO, logger="domo.robot.robot"):
        robot, _ = _robot()
    assert robot.contact_sensor is None
    assert any("no foot links" in r.message for r in caplog.records)


def test_refresh_fills_body_frame_quantities():
    robot, art = _robot()
    yaw = math.pi / 2
    art.quat[:] = _yaw_quat(yaw)
    art.lin_vel[:] = torch.tensor([1.0, 0.0, 0.0])      # world +x
    art.ang_vel[:] = torch.tensor([0.0, 0.0, 0.5])
    art.pos[:] = torch.tensor([1.0, 2.0, 0.3])
    art.dof_pos[:, 0] = 0.25
    state = robot.refresh()
    # World +x seen from a body yawed 90° is body -y.
    assert torch.allclose(state.base_lin_vel[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)
    assert torch.allclose(state.base_lin_vel_world[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(state.base_ang_vel[0], torch.tensor([0.0, 0.0, 0.5]), atol=1e-6)
    assert torch.allclose(state.projected_gravity[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert abs(state.base_euler[0, 2].item() - yaw) < 1e-6
    assert torch.allclose(state.base_pos[0], torch.tensor([1.0, 2.0, 0.3]))
    assert state.dof_pos[0, 0] == 0.25
    # Consistency with the rotation utils: quat_apply(body → world).
    assert torch.allclose(quat_apply(state.base_quat, state.base_lin_vel),
                          state.base_lin_vel_world, atol=1e-6)


def test_set_joint_targets_forwards_to_actuator():
    robot, art = _robot()
    targets = torch.rand(N_ENVS, D)
    robot.set_joint_targets(targets)
    assert torch.equal(art.targets, targets)


def test_reset_idx_writes_state_and_engine():
    robot, art = _robot(base_init_pos=(0.5, 0.0, 0.4))
    art.dof_pos[:] = 1.0
    art.pos[:] = 9.0
    robot.state.base_lin_vel[:] = 3.0
    envs = torch.tensor([0, 2])
    robot.reset_idx(envs)
    default = torch.tensor(GO2.default_dof_angles)
    assert torch.allclose(art.dof_pos[envs], default.expand(2, D))
    assert torch.allclose(art.dof_pos[1], torch.ones(D))            # untouched
    assert torch.allclose(art.pos[envs], torch.tensor([[0.5, 0.0, 0.4]] * 2))
    assert torch.allclose(robot.state.base_pos[envs], torch.tensor([[0.5, 0.0, 0.4]] * 2))
    assert (robot.state.base_lin_vel[envs] == 0).all()
    assert torch.equal(art.velocities_zeroed_for, envs)


def test_reset_idx_custom_spawn_and_empty():
    robot, art = _robot()
    robot.reset_idx(torch.tensor([], dtype=torch.long))          # no-op
    spawn = torch.tensor([[1.0, 1.0, 0.4]])
    robot.reset_idx(torch.tensor([1]), base_pos=spawn)
    assert torch.allclose(art.pos[1], spawn[0])


# ---------------------------------------------------------------------------
# RobotState / RobotSpec
# ---------------------------------------------------------------------------

def test_robot_state_zeros_and_zero_idx():
    s = RobotState.zeros(4, D, 4, torch.device("cpu"))
    assert torch.equal(s.base_quat[:, 0], torch.ones(4))
    s.base_pos[:] = 1.0
    s.base_quat[:] = 0.5
    s.zero_idx(torch.tensor([1, 3]))
    assert (s.base_pos[[1, 3]] == 0).all() and (s.base_pos[[0, 2]] == 1).all()
    assert torch.equal(s.base_quat[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_go2_spec_invariants():
    assert D == 12
    assert GO2.joint_names[0] == "FR_hip_joint" and GO2.joint_names[-1] == "RL_calf_joint"
    assert GO2.default_dof_angles == tuple(GO2.default_joint_angles[n] for n in GO2.joint_names)
    assert GO2.base_init_quat == (1.0, 0.0, 0.0, 0.0)          # wxyz identity
    with pytest.raises(AttributeError):
        GO2.kp = 1.0                                             # frozen


# ---------------------------------------------------------------------------
# DomainRandomization
# ---------------------------------------------------------------------------

def test_dr_dict_roundtrip_ignores_unknown_and_inactive():
    dr = DomainRandomization.from_dict({
        "friction_range": [0.5, 1.5], "obs_noise_std": 0.01, "bogus": 1})
    assert dr.friction_range == (0.5, 1.5)
    assert dr.to_dict() == {"friction_range": [0.5, 1.5], "obs_noise_std": 0.01}
    assert DomainRandomization().to_dict() == {}


def test_dr_apply_skips_unsupported_capabilities_with_one_warning(caplog):
    robot, _ = _robot()
    dr = DomainRandomization(friction_range=(0.8, 1.2), kp_scale_range=(0.9, 1.1))
    with caplog.at_level(logging.WARNING, logger="domo.robot.randomization"):
        dr.apply(robot, torch.arange(N_ENVS))
        dr.apply(robot, torch.arange(N_ENVS))
    msgs = [r.message for r in caplog.records]
    assert sum("'friction'" in m for m in msgs) == 1
    assert sum("'pd_gains'" in m for m in msgs) == 1
    dr.apply(robot, torch.tensor([], dtype=torch.long))          # no-op


def test_dr_apply_uses_supported_capabilities():
    robot, art = _robot()
    calls = {}
    art.set_friction_ratio = lambda ratio, envs_idx: calls.setdefault("friction", ratio)
    art.set_base_com_shift = lambda shift, envs_idx: calls.setdefault("com", shift)
    torch.manual_seed(0)
    DomainRandomization(friction_range=(0.5, 0.5), com_shift_range=(-0.01, 0.01)).apply(
        robot, torch.tensor([0, 1]))
    assert torch.allclose(calls["friction"], torch.full((2,), 0.5))
    assert calls["com"].shape == (2, 3) and calls["com"].abs().max() <= 0.01
