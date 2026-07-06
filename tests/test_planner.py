"""PlanningController: runtime program authoring, outcomes, error recovery."""

import torch

from domo.robot import GO2
from domo.robot.state import RobotState
from domo.skills import PlanningController, make_go2_library

N = 2
DEVICE = torch.device("cpu")


class FakeRobot:
    def __init__(self):
        self.spec = GO2
        self.n_envs = N
        self.device = DEVICE
        self.default_dof_pos = torch.tensor(GO2.default_dof_angles)
        self.state = RobotState.zeros(N, 12, 4, DEVICE)
        self.state.base_pos[:, 2] = 0.32


class FakeLidar:
    def __init__(self):
        self.dist = torch.full((N, 36), 4.0)

    def read(self):
        return self.dist


def _library():
    return make_go2_library(lambda o: torch.zeros(o.shape[0], 12),
                            lambda o: torch.zeros(o.shape[0], 3),
                            FakeLidar())


class ScriptedPlanner(PlanningController):
    """Issues a fixed queue of programs; records what plan() was told."""

    def __init__(self, library, queue):
        super().__init__(library, decision_interval=1)
        self.queue = list(queue)
        self.seen = []

    def plan(self, state, last):
        self.seen.append(last)
        return self.queue.pop(0) if self.queue else None


def _drive(ctrl, robot, steps):
    for _ in range(steps):
        targets = ctrl.update(robot.state, 0.02)
        assert torch.isfinite(targets).all()


def test_planner_authors_and_replans():
    ctrl = ScriptedPlanner(_library(), [
        "walk(vx=0.5).for(0.04)",
        "stand.for(0.04)",
    ])
    robot = FakeRobot()
    ctrl.setup(robot)
    ctrl.reset_idx(torch.arange(N))

    _drive(ctrl, robot, 1)
    assert ctrl.program is not None and "walk" in ctrl.program.source

    _drive(ctrl, robot, 12)                      # both programs run out
    assert ctrl.idle
    assert [o.program for o in ctrl.history] == [
        "walk(vx=0.5).for(0.04)", "stand.for(0.04)"]
    assert all(o.succeeded for o in ctrl.history)
    # plan() got the previous outcome each time: None, then walk, then stand
    assert ctrl.seen[0] is None
    assert "walk" in ctrl.seen[1].program
    assert "stand" in ctrl.seen[2].program


def test_planner_survives_bad_program():
    ctrl = ScriptedPlanner(_library(), [
        "fly.for(1)",                            # unknown skill → CompileError
        "stand.for(0.04)",
    ])
    robot = FakeRobot()
    ctrl.setup(robot)
    ctrl.reset_idx(torch.arange(N))

    _drive(ctrl, robot, 8)                       # never crashes the loop
    assert ctrl.idle
    assert ctrl.history[0].compile_error is not None
    assert not ctrl.history[0].succeeded
    assert ctrl.history[1].succeeded              # recovered with valid one


def test_planner_reports_failure_outcome():
    ctrl = ScriptedPlanner(_library(), ["walk(vx=0.5).for(5)"])
    robot = FakeRobot()
    ctrl.setup(robot)
    ctrl.reset_idx(torch.arange(N))

    _drive(ctrl, robot, 2)
    robot.state.base_euler[:, 1] = 1.2           # tip → walk card aborts
    _drive(ctrl, robot, 3)
    robot.state.base_euler[:, 1] = 0.0
    _drive(ctrl, robot, 2)

    assert ctrl.idle
    assert len(ctrl.history) == 1
    outcome = ctrl.history[0]
    assert not outcome.succeeded
    assert any("FAILURE" in line for line in outcome.trace)
