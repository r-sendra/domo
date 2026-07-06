"""Composition grammar: parsing, type checking, execution semantics."""

import pytest
import torch

from domo.skills import (CompileError, GrammarError, FAILURE, RUNNING,
                         SUCCESS, make_go2_library, parse)
from domo.skills.grammar import Fallback, Layer, Modified, Sequence, SkillRef
from domo.robot import GO2
from domo.robot.state import RobotState

N = 2
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeRobot:
    def __init__(self):
        self.spec = GO2
        self.n_envs = N
        self.device = DEVICE
        self.default_dof_pos = torch.tensor(GO2.default_dof_angles)
        self.state = RobotState.zeros(N, 12, 4, DEVICE)
        self.state.base_pos[:, 2] = 0.32       # healthy standing height


class FakeLidar:
    def __init__(self):
        self.dist = torch.full((N, 36), 4.0)

    def read(self):
        return self.dist


def _library(lidar=None):
    lidar = lidar or FakeLidar()
    walk = lambda obs: torch.zeros(obs.shape[0], 12)
    avoid = lambda obs: torch.zeros(obs.shape[0], 3)
    return make_go2_library(walk, avoid, lidar), lidar


def _compiled(text, lidar=None):
    lib, lidar = _library(lidar)
    prog = lib.compile(text)
    robot = FakeRobot()
    prog.setup(robot)
    prog.reset_idx(torch.arange(N))
    return prog, robot, lidar


def _run(prog, robot, steps, dt=0.02):
    for _ in range(steps):
        targets = prog.update(robot.state, dt)
        assert torch.isfinite(targets).all()
    return prog


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_precedence_and_roundtrip():
    ast = parse("avoid @ walk(vx=0.6) | stand >> walk.for(3)")
    # '>>' loosest: Sequence( Fallback(Layer, stand), Modified(walk) )
    assert isinstance(ast, Sequence)
    assert isinstance(ast.children[0], Fallback)
    assert isinstance(ast.children[0].children[0], Layer)
    assert isinstance(ast.children[1], Modified)
    # round-trips through to_text → parse
    assert parse(ast.to_text()).to_text() == ast.to_text()


def test_parse_layer_right_assoc_and_params():
    ast = parse("a @ b @ walk(vx=0.5, vyaw=-0.3)")
    assert isinstance(ast, Layer) and isinstance(ast.base, Layer)
    base = ast.base.base
    assert isinstance(base, SkillRef)
    assert base.params == {"vx": 0.5, "vyaw": -0.3}


def test_parse_modifiers_stack():
    ast = parse("(walk(vx=0.5).for(2)).repeat(3)")
    assert isinstance(ast, Modified) and ast.repeat == 3
    inner = ast.child
    assert isinstance(inner, Modified) and inner.for_s == 2.0


def test_parse_errors():
    with pytest.raises(GrammarError):
        parse("walk >>")
    with pytest.raises(GrammarError):
        parse("walk.sometimes(3)")
    with pytest.raises(GrammarError):
        parse("walk(vx=)")


# ---------------------------------------------------------------------------
# Type checking / compilation
# ---------------------------------------------------------------------------

def test_command_skill_cannot_run_alone():
    lib, _ = _library()
    with pytest.raises(CompileError, match="cannot run alone"):
        lib.compile("avoid.for(3)")


def test_motor_skill_cannot_be_layer_top():
    lib, _ = _library()
    with pytest.raises(CompileError, match="motor skill"):
        lib.compile("stand @ walk")


def test_layer_channel_mismatch():
    lib, _ = _library()
    with pytest.raises(CompileError, match="accepts"):
        lib.compile("avoid @ stand")      # stand consumes no command channel


def test_unknown_skill_and_param_and_range():
    lib, _ = _library()
    with pytest.raises(CompileError, match="unknown skill"):
        lib.compile("fly.for(1)")
    with pytest.raises(CompileError, match="no parameter"):
        lib.compile("walk(speed=1).for(1)")
    with pytest.raises(CompileError, match="outside allowed range"):
        lib.compile("walk(vx=5).for(1)")


def test_modifier_inside_layer_base_rejected():
    lib, _ = _library()
    with pytest.raises(CompileError, match="right side of '@'"):
        lib.compile("avoid @ (walk(vx=0.5).for(3))")


# ---------------------------------------------------------------------------
# Execution semantics
# ---------------------------------------------------------------------------

def test_for_modifier_and_sequence_advance():
    prog, robot, _ = _compiled("walk(vx=0.5).for(0.1) >> stand.for(0.1)")
    _run(prog, robot, 4)                          # 0.08 s — still in walk
    assert prog.status == RUNNING
    _run(prog, robot, 4)                          # crosses both boundaries?
    _run(prog, robot, 8)
    assert prog.status == SUCCESS
    text = "\n".join(prog.trace)
    assert "sequence → step 2/2" in text and "program → SUCCESS" in text


def test_layer_adds_correction_to_base_command():
    prog, robot, lidar = _compiled("(avoid @ walk(vx=0.6)).for(1)")
    _run(prog, robot, 2)
    walk = prog.instances["walk"]
    # zero-policy avoidance → tanh(0)=0 correction → command == base
    assert torch.allclose(walk.command, torch.tensor([[0.6, 0.0, 0.0]] * N))


def test_layer_clamps_to_base_constraints():
    lib, lidar = _library()
    # avoidance policy pushing hard forward: Δvx→+0.8 ⇒ 0.6+0.8=1.4 < 2 ok;
    # use big base vx to force clamping at the card limit (vx ≤ 2).
    avoid = lambda obs: torch.full((obs.shape[0], 3), 10.0)   # tanh→1
    lib2 = make_go2_library(lambda o: torch.zeros(o.shape[0], 12), avoid, lidar)
    prog = lib2.compile("(avoid @ walk(vx=1.9)).for(1)")
    robot = FakeRobot()
    prog.setup(robot)
    prog.reset_idx(torch.arange(N))
    prog.update(robot.state, 0.02)
    walk = prog.instances["walk"]
    assert walk.command[0, 0].item() == pytest.approx(2.0)    # clamped
    assert walk.command[0, 2].item() == pytest.approx(1.5)    # vyaw clamp


def test_fail_condition_triggers_fallback():
    prog, robot, _ = _compiled("walk(vx=0.5).for(5) | stand.for(0.1)")
    robot.state.base_euler[:, 1] = 1.2            # pitch beyond tipped(0.9)
    prog.update(robot.state, 0.02)                # walk fails
    robot.state.base_euler[:, 1] = 0.0
    _run(prog, robot, 8)                          # stand completes
    assert prog.status == SUCCESS
    assert any("fallback → alternative" in line for line in prog.trace)


def test_failure_without_fallback_fails_program():
    prog, robot, _ = _compiled("walk(vx=0.5).for(5) >> stand.for(1)")
    robot.state.base_pos[:, 2] = 0.05             # fallen
    prog.update(robot.state, 0.02)
    prog.update(robot.state, 0.02)
    assert prog.status == FAILURE
    # terminal → holds default pose
    hold = prog.update(robot.state, 0.02)
    assert torch.allclose(hold, robot.default_dof_pos.expand(N, 12))


def test_until_condition_with_sensor():
    prog, robot, lidar = _compiled(
        "(avoid @ walk(vx=0.6)).until(blocked(0.5)) >> stand.for(0.1)")
    _run(prog, robot, 3)
    assert prog.status == RUNNING
    lidar.dist[:, 5] = 0.3                        # obstacle appears close
    _run(prog, robot, 10)
    assert prog.status == SUCCESS                 # until fired, then stand


def test_repeat_modifier():
    import re
    prog, robot, _ = _compiled("(walk(vx=0.5).for(0.04)).repeat(3)")
    _run(prog, robot, 12)
    assert prog.status == SUCCESS
    re_entries = [l for l in prog.trace if re.search(r"repeat \d+/3 \(", l)]
    assert len(re_entries) == 2                    # runs 2/3 and 3/3


def test_library_describe_mentions_everything():
    lib, _ = _library()
    text = lib.describe()
    for token in ("SKILL walk", "SKILL avoid", "SKILL stand",
                  "COMPOSITION GRAMMAR", "vx", "blocked", "fails when"):
        assert token in text, token
