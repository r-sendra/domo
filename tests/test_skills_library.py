"""Skill library internals — cards, conditions, compiler, layer semantics.

Complements tests/test_skill_grammar.py with the composition-core rules
added for navigation: a FRESH command-skill instance per '@' with grammar
params applied, LayerNode SUCCESS via success_flags (registered by the
enclosing SequenceNode on the NEXT tick), override vs additive layering,
and the passive SLAM layer.
"""

import pytest
import torch

from domo.robot import GO2
from domo.robot.state import RobotState
from domo.skills import (
    CMD_VELOCITY,
    MOTOR,
    RUNNING,
    SUCCESS,
    ParamSpec,
    SkillCard,
    make_go2_library,
    standard_conditions,
)
from domo.skills.grammar import GrammarError, parse, parse_condition

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
    n_sectors = 36
    max_range = 4.0

    def __init__(self):
        self.dist = torch.full((N, self.n_sectors), self.max_range)

    def read(self):
        return self.dist


def _library(avoid_out=0.0):
    def walk(obs):
        return torch.zeros(obs.shape[0], 12)

    def avoid(obs):
        return torch.full((obs.shape[0], 3), avoid_out)

    return make_go2_library(walk, avoid, FakeLidar())


def _compiled(text, **kw):
    prog = _library(**kw).compile(text)
    robot = FakeRobot()
    prog.setup(robot)
    prog.reset_idx(torch.arange(N))
    return prog, robot


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def test_card_describe_states_the_channel_correctly():
    cmd = SkillCard("nudge", "push", interface=CMD_VELOCITY,
                    params=[ParamSpec("k", "gain", 1.0, (0.0, 2.0), "x")],
                    max_duration_s=3.0)
    text = cmd.describe()
    assert "drives the 'velocity' channel" in text
    assert "k=1.0 [x] range=[0.0, 2.0]: gain" in text
    assert "never by itself" in text and "aborts after 3.0s" in text

    motor = SkillCard("walk", "go", interface=MOTOR, accepts="velocity")
    assert "[motor, accepts 'velocity']" in motor.describe()
    plain = SkillCard("stand", "hold")
    assert plain.describe().startswith("SKILL stand  [motor]")
    with pytest.raises(KeyError, match="no parameter"):
        plain.param("vx")


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def test_standard_conditions():
    reg = standard_conditions()
    assert reg.names() == ["fallen", "moved", "still", "timeout", "tipped"]
    state = RobotState.zeros(N, 12, 4, DEVICE)
    state.base_pos[:, 2] = 0.3
    assert reg.make("timeout", 1.0)(state, 0.5).tolist() == [False, False]
    assert reg.make("timeout", 1.0)(state, 1.0).tolist() == [True, True]
    state.base_euler[0, 1] = 0.8
    assert reg.make("tipped", 0.7)(state, 0).tolist() == [True, False]
    state.base_pos[1, 2] = 0.1
    assert reg.make("fallen", 0.18)(state, 0).tolist() == [False, True]
    state.base_lin_vel[0, 0] = 1.0
    assert reg.make("still", 0.05)(state, 0).tolist() == [False, True]
    with pytest.raises(KeyError, match="unknown condition"):
        reg.make("flying")


def test_moved_latches_origin_and_relatches_when_clock_restarts():
    moved = standard_conditions().make("moved", 1.0)
    state = RobotState.zeros(N, 12, 4, DEVICE)
    state.base_pos[:, 0] = 5.0
    assert not moved(state, 0.0).any()                 # origin latched at x=5
    state.base_pos[:, 0] = 6.5
    assert moved(state, 1.0).all()
    assert not moved(state, 0.0).any()                 # t_s went back → re-latch


def test_parse_condition():
    ref = parse_condition("tipped(0.9)")
    assert ref.name == "tipped" and ref.args == (0.9,)
    assert parse_condition("still").to_text() == "still"
    with pytest.raises(GrammarError, match="trailing"):
        parse_condition("still >> x")
    with pytest.raises(GrammarError, match="unexpected character"):
        parse("walk $ stand")


def test_to_text_emits_canonical_minimal_parentheses():
    text = "((avoid @ walk) | stand).for(2) >> (walk(vx=1).for(1)).repeat(2)"
    # '@' binds tighter than '|' and modifiers chain, so those parens vanish;
    # the ones needed to keep '|' under '.for' and '>>' survive.
    canonical = "(avoid @ walk | stand).for(2) >> walk(vx=1).for(1).repeat(2)"
    assert parse(text).to_text() == canonical
    assert parse(canonical).to_text() == canonical


# ---------------------------------------------------------------------------
# Compiler: instances and parameters
# ---------------------------------------------------------------------------

def test_each_layer_occurrence_gets_a_fresh_configured_command_skill():
    prog, _ = _compiled("goto(x=2, y=1) @ walk >> goto(x=0, y=0) @ walk")
    keys = sorted(prog.instances)
    assert keys == ["goto#1", "goto#2", "walk"]           # walk shared, goto not
    assert prog.instances["goto#1"].target == (2.0, 1.0)
    assert prog.instances["goto#2"].target == (0.0, 0.0)
    assert prog.instances["goto#1"] is not prog.instances["goto#2"]


def test_layer_success_flags_complete_the_layer_next_tick():
    prog, robot = _compiled("goto(x=1, y=0) @ walk(vx=0.9) >> stand.for(0.02)")
    robot.state.base_pos[:, 0] = 0.95                     # already within tol
    prog.update(robot.state, 0.02)
    assert prog.status == RUNNING
    assert any("goto → SUCCESS (goal reached)" in line for line in prog.trace)
    # Override mode: walk sees goto's (zero, arrived) command, not vx=0.9.
    assert (prog.instances["walk"].command == 0).all()
    prog.update(robot.state, 0.02)                        # sequence advances
    assert any("sequence → step 2/2" in line for line in prog.trace)
    prog.update(robot.state, 0.02)
    prog.update(robot.state, 0.02)
    assert prog.status == SUCCESS


def test_additive_layer_stacks_on_override_layer_and_is_clamped():
    prog, robot = _compiled("(avoid @ goto(x=1, y=0) @ walk).for(1)",
                            avoid_out=10.0)               # tanh → deltas
    robot.state.base_pos[:, 0] = 0.95                     # goto arrived → 0
    prog.update(robot.state, 0.02)
    walk = prog.instances["walk"]
    assert torch.allclose(walk.command, torch.tensor([[0.8, 0.5, 1.5]] * N))


def test_slam_layer_is_transparent_to_control():
    prog, robot = _compiled("(slam @ avoid @ walk(vx=0.6)).for(1)")
    prog.update(robot.state, 0.02)
    walk = prog.instances["walk"]
    assert torch.allclose(walk.command, torch.tensor([[0.6, 0.0, 0.0]] * N))
    slam = next(s for k, s in prog.instances.items() if k.startswith("slam"))
    assert slam.occupancy_prob().shape[0] == N


def test_library_without_lidar_has_nav_but_no_avoid_or_slam():
    lib = make_go2_library(lambda o: torch.zeros(o.shape[0], 12))
    assert lib.names() == ["backward", "forward", "goto", "stand", "walk"]
    assert "blocked" not in lib.conditions.names()
