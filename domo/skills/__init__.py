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

from .card import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard
from .builtin import AVOID_CARD, STAND_CARD, WALK_CARD, make_go2_library
from .conditions import ConditionRegistry, standard_conditions
from .grammar import GrammarError, parse
from .library import CompileError, SkillLibrary
from .nodes import FAILURE, RUNNING, SUCCESS, CompositeSkill
from .planner import PlanningController, PlanOutcome

__all__ = [
    "SkillCard", "ParamSpec", "MOTOR", "CMD_VELOCITY",
    "SkillLibrary", "CompileError", "GrammarError", "parse",
    "CompositeSkill", "RUNNING", "SUCCESS", "FAILURE",
    "ConditionRegistry", "standard_conditions",
    "make_go2_library", "STAND_CARD", "WALK_CARD", "AVOID_CARD",
    "PlanningController", "PlanOutcome",
]
