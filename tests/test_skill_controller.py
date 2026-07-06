"""Skill / Controller / ControlLoop layer — tested on fakes, no physics."""

import torch

from domo.control import (Controller, CPGLocomotionSkill, SingleSkillController,
                          SimControlLoop, StandSkill)
from domo.control.cpg import CPG_OBS_DIM
from domo.robot import GO2
from domo.robot.state import RobotState

N = 3
DEVICE = torch.device("cpu")


class FakeRobot:
    """Just enough of the Robot facade for skills/controllers/loops."""

    def __init__(self, n_envs=N):
        self.spec = GO2
        self.n_envs = n_envs
        self.device = DEVICE
        self.default_dof_pos = torch.tensor(GO2.default_dof_angles)
        self.state = RobotState.zeros(n_envs, 12, 4, DEVICE)
        self.applied = []
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1
        return self.state

    def set_joint_targets(self, targets):
        self.applied.append(targets.clone())

    def reset_idx(self, envs_idx, base_pos=None):
        pass


class FakeScene:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _zero_policy(obs):
    assert obs.shape[-1] == CPG_OBS_DIM
    return torch.zeros(obs.shape[0], 12)


def test_stand_skill_outputs_default_pose():
    robot = FakeRobot()
    skill = StandSkill()
    skill.setup(robot)
    targets = skill.update(robot.state, 0.02)
    assert targets.shape == (N, 12)
    assert torch.allclose(targets, robot.default_dof_pos.expand(N, 12))


def test_cpg_locomotion_skill_runs_and_resets():
    robot = FakeRobot()
    skill = CPGLocomotionSkill(_zero_policy)
    skill.setup(robot)
    skill.reset_idx(torch.arange(N))
    skill.command[:, 0] = 0.5

    theta0 = skill.oscillators.theta.clone()
    for _ in range(10):
        targets = skill.update(robot.state, 0.02)
    assert targets.shape == (N, 12) and torch.isfinite(targets).all()
    assert not torch.allclose(skill.oscillators.theta, theta0)  # phase advanced
    assert skill.stance_mask().shape == (N, 4)

    skill.reset_idx(torch.arange(N))
    assert torch.allclose(skill.oscillators.theta, theta0)
    assert (skill.command == 0).all()


class CountingController(Controller):
    def __init__(self, skills, initial, interval):
        super().__init__(skills, initial, decision_interval=interval)
        self.decisions = 0

    def decide(self, state):
        self.decisions += 1


def test_controller_decision_cadence_and_delegation():
    robot = FakeRobot()
    ctrl = CountingController({"stand": StandSkill()}, "stand", interval=5)
    ctrl.setup(robot)
    for _ in range(10):
        targets = ctrl.update(robot.state, 0.02)
    assert ctrl.decisions == 2                       # ticks 0 and 5
    assert torch.allclose(targets, robot.default_dof_pos.expand(N, 12))


def test_controller_activate_resets_incoming_skill():
    robot = FakeRobot()
    walk = CPGLocomotionSkill(_zero_policy)
    ctrl = CountingController({"walk": walk, "stand": StandSkill()},
                              "stand", interval=1)
    ctrl.setup(robot)
    walk.reset_idx(torch.arange(N))
    theta0 = walk.oscillators.theta.clone()

    walk.update(robot.state, 0.02)                   # dirty the oscillator state
    assert not torch.allclose(walk.oscillators.theta, theta0)
    ctrl.activate("walk")
    assert ctrl.active == "walk"
    assert torch.allclose(walk.oscillators.theta, theta0)


def test_sim_control_loop_cycle():
    robot = FakeRobot()
    scene = FakeScene()
    ctrl = SingleSkillController(StandSkill())
    ctrl.setup(robot)

    filtered = []

    def clamp_filter(cmd, state):
        filtered.append(True)
        return cmd.clamp(-2.0, 2.0)

    loop = SimControlLoop(scene, robot, ctrl, dt=0.02,
                          command_filter=clamp_filter)
    loop.reset()
    seen = []
    loop.run(7, callback=lambda i, s: seen.append(i))

    assert scene.steps == 7
    assert len(robot.applied) == 7                   # one actuator write/cycle
    assert len(filtered) == 7                        # safety filter unbypassable
    assert robot.refresh_count == 8                  # reset + 7 cycles
    assert seen == list(range(7))
