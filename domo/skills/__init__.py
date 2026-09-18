"""
The skill library layer (M5): SkillCards (LLM-legible metadata), a symbolic
composition grammar (@ layering, >> sequence, | fallback, .for/.until/.repeat
modifiers), a type-checking compiler, and executable CompositeSkills.

    from domo.skills import make_go2_library
    lib = make_go2_library(walk_policy, avoid_policy, lidar)
    prog = lib.compile("(avoid @ walk(vx=0.6)).until(moved(3)) >> stand.for(2)")
    prog.setup(robot)      # a CompositeSkill is an ordinary Skill
    print(lib.describe())  # the catalog block an LLM plans against
"""

from .builtin import (
                     AVOID_CARD,
                     BACKWARD_CARD,
                     FORWARD_CARD,
                     GOTO_CARD,
                     SLAM_CARD,
                     STAND_CARD,
                     WALK_CARD,
                     make_go2_library,
)
from .card import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard
from .conditions import ConditionRegistry, standard_conditions
from .grammar import GrammarError, parse
from .library import CompileError, SkillLibrary
from .nodes import FAILURE, RUNNING, SUCCESS, CompositeSkill
from .planner import PlanningController, PlanOutcome

__all__ = [
                     "AVOID_CARD",
                     "BACKWARD_CARD",
                     "CMD_VELOCITY",
                     "FAILURE",
                     "FORWARD_CARD",
                     "GOTO_CARD",
                     "MOTOR",
                     "RUNNING",
                     "SLAM_CARD",
                     "STAND_CARD",
                     "SUCCESS",
                     "WALK_CARD",
                     "CompileError",
                     "CompositeSkill",
                     "ConditionRegistry",
                     "GrammarError",
                     "ParamSpec",
                     "PlanOutcome",
                     "PlanningController",
                     "SkillCard",
                     "SkillLibrary",
                     "make_go2_library",
                     "parse",
                     "standard_conditions",
]
