"""PositionController converges on a kinematic point-robot fake (no physics)."""

import math

import torch

from domo.control import NavConfig, PositionController


class FakeRobot:
    """Perfect velocity tracking: integrate commands directly."""

    def __init__(self, dt=0.02):
        self.dt = dt
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0

    def step(self, cmd: torch.Tensor):
        vx, vy, vyaw = float(cmd[0, 0]), float(cmd[0, 1]), float(cmd[0, 2])
        self.x += (vx * math.cos(self.yaw) - vy * math.sin(self.yaw)) * self.dt
        self.y += (vx * math.sin(self.yaw) + vy * math.cos(self.yaw)) * self.dt
        self.yaw += vyaw * self.dt

    def pose(self):
        return (self.x, self.y, self.yaw)


def _controller(robot):
    return PositionController(robot.step, robot.pose,
                              NavConfig(dt=robot.dt, verbose=False))


def test_go_to_reaches_goal():
    robot = FakeRobot()
    ctrl = _controller(robot)
    assert ctrl.go_to(2.0, 1.0) == "done"
    assert math.hypot(robot.x - 2.0, robot.y - 1.0) < 0.2


def test_turn_left_and_right():
    robot = FakeRobot()
    ctrl = _controller(robot)
    assert ctrl.turn(90.0) == "done"
    assert abs(robot.yaw - math.pi / 2) < 0.1
    assert ctrl.turn(-90.0) == "done"
    assert abs(robot.yaw) < 0.1


def test_forward_then_backward_returns_home():
    robot = FakeRobot()
    ctrl = _controller(robot)
    assert ctrl.go_forward(3.0) == "done"
    assert abs(robot.x - 3.0) < 0.25
    assert ctrl.go_backward(3.0) == "done"
    assert abs(robot.x) < 0.4
