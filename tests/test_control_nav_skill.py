"""Command skills on fakes — no physics engine.

Pins the TrajectoryTrackingSkill control law (along-track floor, cross-track
correction, heading hold, arrival latch, lazy trajectory latching after
reset_idx), the CommandSkill contract, and LidarAvoidanceSkill's bounded
correction.
"""

import math

import pytest
import torch

from domo.control import (
    CommandSkill,
    LidarAvoidanceSkill,
    NavGains,
    TrajectoryTrackingSkill,
    all_envs,
    planar_pose,
)
from domo.robot import GO2
from domo.robot.state import RobotState

N = 2
DEVICE = torch.device("cpu")


class FakeRobot:
    def __init__(self):
        self.spec = GO2
        self.n_envs = N
        self.device = DEVICE
        self.default_dof_pos = torch.tensor(GO2.default_dof_angles)
        self.state = RobotState.zeros(N, 12, 4, DEVICE)


def test_helpers():
    robot = FakeRobot()
    assert all_envs(robot).tolist() == [0, 1]
    robot.state.base_pos[1] = torch.tensor([1.0, 2.0, 0.3])
    robot.state.base_euler[1, 2] = 0.5
    xy, yaw = planar_pose(robot.state)
    assert xy.shape == (N, 2) and yaw.shape == (N,)
    assert xy[1].tolist() == [1.0, 2.0] and yaw[1].item() == 0.5


def _nav(mode, **params):
    skill = TrajectoryTrackingSkill(mode)
    skill.configure(**params)
    robot = FakeRobot()
    skill.setup(robot)
    skill.reset_idx(torch.arange(N))
    return skill, robot


def _set_pose(state, x, y, yaw=0.0, env=None):
    envs = slice(None) if env is None else env
    state.base_pos[envs, 0] = x
    state.base_pos[envs, 1] = y
    state.base_euler[envs, 2] = yaw


# ---------------------------------------------------------------------------
# CommandSkill contract
# ---------------------------------------------------------------------------

class _NoOp(CommandSkill):
    def update_command(self, state, dt):
        return torch.zeros(state.base_pos.shape[0], 3)


def test_command_skill_contract():
    skill = _NoOp()
    skill.configure()                                   # no params is fine
    with pytest.raises(TypeError, match="takes no parameters"):
        skill.configure(speed=1.0)
    with pytest.raises(TypeError, match="joint targets"):
        skill.update(None, 0.02)                        # must be layered
    assert skill.success_flags(None) is None            # never self-terminates


# ---------------------------------------------------------------------------
# TrajectoryTrackingSkill
# ---------------------------------------------------------------------------

def test_nav_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown nav mode"):
        TrajectoryTrackingSkill("sideways")


def test_forward_cruises_then_corrects_cross_track_then_arrives():
    skill, robot = _nav("forward", distance=1.0, speed=0.5)
    g = skill.cfg
    cmd = skill.update_command(robot.state, 0.02)       # latches at origin
    assert torch.allclose(cmd, torch.tensor([[0.5, 0.0, 0.0]] * N))
    assert not skill.success_flags(robot.state).any()

    # Drifted 0.1 m left of the line at mid-path: pushed back at kp_perp.
    _set_pose(robot.state, 0.5, 0.1)
    cmd = skill.update_command(robot.state, 0.02)
    assert cmd[0, 0].item() == pytest.approx(0.5)
    assert cmd[0, 1].item() == pytest.approx(-g.kp_perp * 0.1)
    assert cmd[0, 2].item() == pytest.approx(0.0)

    # Near the goal the along-track speed is floored, not proportional...
    _set_pose(robot.state, 0.7, 0.0)
    cmd = skill.update_command(robot.state, 0.02)
    assert cmd[0, 0].item() == pytest.approx(g.min_speed)

    # ...and inside tol_pos the skill latches "arrived" and holds still.
    _set_pose(robot.state, 0.9, 0.0)
    cmd = skill.update_command(robot.state, 0.02)
    assert (cmd == 0).all()
    assert skill.success_flags(robot.state).all()
    _set_pose(robot.state, 0.5, 0.0)                    # pushed back: stays done
    assert (skill.update_command(robot.state, 0.02) == 0).all()


def test_backward_reverses_without_turning():
    skill, robot = _nav("backward", distance=1.0, speed=0.5)
    cmd = skill.update_command(robot.state, 0.02)
    assert torch.allclose(cmd, torch.tensor([[-0.5, 0.0, 0.0]] * N))


def test_goto_faces_the_target_and_uses_body_frame():
    skill, robot = _nav("goto", x=0.0, y=2.0, speed=0.5)
    g = skill.cfg
    cmd = skill.update_command(robot.state, 0.02)       # target straight left
    assert cmd[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert cmd[0, 1].item() == pytest.approx(0.5)
    assert cmd[0, 2].item() == pytest.approx(g.max_vyaw)   # heading error π/2
    # Once facing the target the same world velocity is pure body-forward.
    _set_pose(robot.state, 0.0, 0.0, yaw=math.pi / 2)
    cmd = skill.update_command(robot.state, 0.02)
    assert cmd[0, 0].item() == pytest.approx(0.5)
    assert cmd[0, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert cmd[0, 2].item() == pytest.approx(0.0, abs=1e-6)


def test_speed_floor_is_capped_by_cruise_speed():
    skill, robot = _nav("forward", distance=1.0, speed=0.1)
    _set_pose(robot.state, 0.7, 0.0)
    skill.update_command(robot.state, 0.02)             # latch here
    skill.reset_idx(torch.arange(N))
    _set_pose(robot.state, 0.0, 0.0)
    skill.update_command(robot.state, 0.02)
    _set_pose(robot.state, 0.7, 0.0)
    cmd = skill.update_command(robot.state, 0.02)
    assert cmd[0, 0].item() == pytest.approx(0.1)       # min(speed, min_speed)


def test_reset_relatches_trajectory_from_current_pose_per_env():
    skill, robot = _nav("forward", distance=1.0)
    _set_pose(robot.state, 0.9, 0.0, env=1)             # env 1 starts elsewhere
    skill.update_command(robot.state, 0.02)
    assert torch.allclose(skill._start[1], torch.tensor([0.9, 0.0]))
    _set_pose(robot.state, 1.8, 0.0, env=1)             # env 1 arrives
    assert skill.success_flags(robot.state).tolist() == [False, False]
    skill.update_command(robot.state, 0.02)
    assert skill.success_flags(robot.state).tolist() == [False, True]

    skill.reset_idx(torch.tensor([1]))                  # re-activate env 1 only
    assert skill.success_flags(robot.state).tolist() == [False, False]
    skill.update_command(robot.state, 0.02)
    assert torch.allclose(skill._start[1], torch.tensor([1.8, 0.0]))
    assert torch.allclose(skill._start[0], torch.tensor([0.0, 0.0]))


def test_configure_and_gains():
    skill = TrajectoryTrackingSkill("goto", target=(1, 2), cfg=NavGains(tol_pos=0.5))
    assert skill.target == (1.0, 2.0) and skill.cfg.tol_pos == 0.5
    skill.configure(x=3, speed=0.8, distance=4)
    assert skill.target == (3.0, 2.0)
    assert skill.speed == 0.8 and skill.distance == 4.0


# ---------------------------------------------------------------------------
# LidarAvoidanceSkill
# ---------------------------------------------------------------------------

class FakeLidar:
    def __init__(self):
        self.dist = torch.full((N, 36), 4.0)

    def read(self):
        return self.dist


def test_avoidance_correction_is_bounded_by_deltas():
    lidar = FakeLidar()
    lidar.dist[0, 3] = 0.5
    skill = LidarAvoidanceSkill(lambda obs: torch.full((obs.shape[0], 3), 10.0),
                                lidar, deltas=(0.25, 0.5, 1.2))
    skill.setup(FakeRobot())
    assert skill.min_distance().tolist() == [0.5, 4.0]
    cmd = skill.update_command(None, 0.02)
    assert torch.allclose(cmd, torch.tensor([[0.25, 0.5, 1.2]] * N), atol=1e-6)
