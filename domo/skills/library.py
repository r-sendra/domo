"""
SkillLibrary: the M5 registry — skills + cards + conditions — and the
compiler from grammar programs to executable CompositeSkills.

    library = SkillLibrary()
    library.register(walk_card, lambda: CPGLocomotionSkill(policy))
    library.register(avoid_card, lambda: LidarAvoidanceSkill(policy, lidar))

    program = library.compile("(avoid @ walk(vx=0.6)).for(8) >> stand.for(2)")
    program.setup(robot)          # CompositeSkill — host it in any Controller

    print(library.describe())     # the full catalog as an LLM prompt block

compile() type-checks the program against the cards:
  * every execution position must be motor-typed
  * '@' requires top.interface == "command:X" and base.accepts == "X"
  * parameters must exist on the card and lie within their declared range
  * conditions must exist in the registry
so an ill-formed (e.g. LLM-generated) program fails at compile time with a
readable error, never on the robot.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from domo.control.skill import CommandSkill

from . import grammar
from .card import MOTOR, SkillCard
from .conditions import ConditionRegistry, standard_conditions
from .nodes import (
    CompositeSkill,
    ExecContext,
    FallbackNode,
    LayerNode,
    ModifiedNode,
    MotorLeaf,
    SequenceNode,
)

__all__ = ["CompileError", "SkillLibrary"]

# Canonical parameter order of the velocity command channel.
VELOCITY_PARAMS = ("vx", "vy", "vyaw")

# Clamp bounds used for a velocity component the card leaves unconstrained.
_UNBOUNDED = (-1e9, 1e9)


class CompileError(ValueError):
    """A well-formed program that violates the library's type rules."""


class SkillLibrary:
    """Registry of SkillCards + skill factories + conditions, and the compiler.

    Args:
        conditions: condition registry to compile `.until(...)` and card
            conditions against; defaults to `standard_conditions()`.
    """

    def __init__(self, conditions: ConditionRegistry | None = None):
        self._cards: dict[str, SkillCard] = {}
        self._factories: dict[str, Callable[[], object]] = {}
        self.conditions = conditions or standard_conditions()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, card: SkillCard, factory: Callable[[], object]) -> None:
        """factory() → a fresh Skill/CommandSkill instance for this library."""
        self._cards[card.name] = card
        self._factories[card.name] = factory

    def register_condition(self, name: str, factory) -> None:
        """Add a condition factory (see conditions.py) usable in programs/cards."""
        self.conditions.register(name, factory)

    def card(self, name: str) -> SkillCard:
        """Look up a card by skill name.

        Raises:
            CompileError: unknown skill (the LLM-facing error).
        """
        if name not in self._cards:
            raise CompileError(
                f"unknown skill '{name}' (library has: {sorted(self._cards)})")
        return self._cards[name]

    def names(self) -> list[str]:
        """Registered skill names, sorted."""
        return sorted(self._cards)

    # ------------------------------------------------------------------
    # LLM-facing catalog
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Every card's prompt block plus a grammar cheat-sheet (planner prompt)."""
        blocks = [self._cards[n].describe() for n in self.names()]
        blocks.append(
            "COMPOSITION GRAMMAR\n"
            "  A @ B    execute command-skill A on top of motor-skill B\n"
            "  A >> B   A then B (B starts when A succeeds)\n"
            "  A | B    try A; if A fails, B\n"
            "  X.for(T)         X succeeds after T seconds\n"
            "  X.until(cond)    X succeeds when cond holds\n"
            "  X.repeat(n)      loop X n times\n"
            f"  conditions: {', '.join(self.conditions.names())}\n"
            "  precedence (tightest first): modifiers, '@', '|', '>>'; "
            "parentheses group.")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, program_text: str, device="cpu") -> CompositeSkill:
        """Parse + type-check + instantiate a program into a CompositeSkill.

        Args:
            program_text: grammar source (see grammar.py).
            device: where constant command tensors live (the robot's device).

        Raises:
            GrammarError: malformed text.
            CompileError: unknown skill/condition, bad parameter, or an
                ill-typed layering/position.
        """
        ast = grammar.parse(program_text)
        build = _Compilation(self, torch.device(device))
        root = build.node(ast)
        return CompositeSkill(root, build.ctx, build.instances, ast.to_text())

    # ------------------------------------------------------------------

    @staticmethod
    def _check_params(card: SkillCard, params: dict):
        """Every grammar parameter must exist on the card and lie in its range."""
        for key, value in params.items():
            try:
                spec = card.param(key)
            except KeyError as e:
                raise CompileError(str(e)) from None
            if spec.range is not None and not (spec.range[0] <= value <= spec.range[1]):
                raise CompileError(
                    f"{card.name}.{key}={value} outside allowed range "
                    f"{list(spec.range)}")

    @staticmethod
    def _command_setup(card: SkillCard, params: dict, device):
        """Constant command vector [3] + clamp bounds [3] for velocity consumers.

        Returns (None, None, None) for motor skills without a command channel.
        """
        if card.accepts != "velocity":
            if params:
                raise CompileError(
                    f"'{card.name}' takes no parameters (got {params})")
            return None, None, None
        defaults = {p.name: p.default for p in card.params}
        defaults.update(params)
        const = torch.tensor([defaults.get(k, 0.0) for k in VELOCITY_PARAMS],
                             device=device, dtype=torch.float32)
        lo = torch.tensor([card.constraints.get(k, _UNBOUNDED)[0]
                           for k in VELOCITY_PARAMS], device=device)
        hi = torch.tensor([card.constraints.get(k, _UNBOUNDED)[1]
                           for k in VELOCITY_PARAMS], device=device)
        return const, lo, hi


class _Compilation:
    """One compile() run: lowers the AST into ExecNodes, sharing one context.

    Instance policy:
      * motor skills are instantiated ONCE per program and shared by every
        reference (keyed by name) — their per-leaf parameters are constant
        commands owned by the MotorLeaf, not by the skill;
      * command skills get a FRESH instance per '@' occurrence (keyed
        `name#k`) because they carry per-occurrence goal state (e.g. a nav
        target) that would collide if shared, and the grammar's parameters
        are applied to that instance via `configure()`.
    """

    def __init__(self, library: SkillLibrary, device: torch.device):
        self.lib = library
        self.device = device
        self.ctx = ExecContext()
        self.instances: dict[str, object] = {}

    # -- helpers -----------------------------------------------------------

    def _motor_instance(self, name: str):
        if name not in self.instances:
            self.instances[name] = self.lib._factories[name]()
        return self.instances[name]

    def _conditions(self, exprs):
        """Card condition expressions → callables (raises on unknown names)."""
        out = []
        for expr in exprs:
            ref = grammar.parse_condition(expr)
            out.append(self.lib.conditions.make(ref.name, *ref.args))
        return out

    # -- lowering, one method per AST node type ----------------------------

    def node(self, node):
        if isinstance(node, grammar.SkillRef):
            return self._skill_ref(node)
        if isinstance(node, grammar.Layer):
            return self._layer(node)
        if isinstance(node, grammar.Sequence):
            return SequenceNode(self.ctx, [self.node(c) for c in node.children])
        if isinstance(node, grammar.Fallback):
            return FallbackNode(self.ctx, [self.node(c) for c in node.children])
        if isinstance(node, grammar.Modified):
            return self._modified(node)
        raise CompileError(f"unsupported AST node {type(node).__name__}")

    def _skill_ref(self, node: grammar.SkillRef) -> MotorLeaf:
        # A bare skill reference sits at an execution position, so it must be
        # a motor skill (command skills only appear as the top of a Layer).
        card = self.lib.card(node.name)
        if card.interface != MOTOR:
            raise CompileError(
                f"'{node.name}' is a {card.interface} skill — it "
                f"cannot run alone; layer it: '{node.name} @ "
                f"<motor skill>'")
        self.lib._check_params(card, node.params)
        const_cmd, lo, hi = self.lib._command_setup(card, node.params, self.device)
        return MotorLeaf(self.ctx, node.name, self._motor_instance(node.name),
                         card, const_cmd, lo, hi,
                         self._conditions(card.success_when),
                         self._conditions(card.fail_when))

    def _layer(self, node: grammar.Layer) -> LayerNode:
        top = node.top
        if not isinstance(top, grammar.SkillRef):
            raise CompileError("the left side of '@' must be a single "
                               "command skill")
        top_card = self.lib.card(top.name)
        if top_card.interface == MOTOR:
            raise CompileError(
                f"'{top.name}' is a motor skill — only command skills "
                f"can be layered on top ('@'). Did you mean "
                f"'{node.base.to_text()} >> {top.name}'?")
        self.lib._check_params(top_card, top.params)
        base = self.node(node.base)
        if not isinstance(base, (MotorLeaf, LayerNode)):
            raise CompileError(
                f"the right side of '@' must be a (possibly layered) "
                f"motor skill — apply modifiers to the whole layer "
                f"instead: '({node.to_text()}).for(...)'")
        # Channel check against the motor leaf at the bottom of the stack.
        channel = top_card.channel
        base_card = self.lib.card(base.motor_leaf.name
                                  if isinstance(base, LayerNode)
                                  else base.name)
        if base_card.accepts != channel:
            raise CompileError(
                f"cannot layer '{top.name}' (drives '{channel}') on "
                f"'{base_card.name}' (accepts "
                f"'{base_card.accepts or 'nothing'}')")
        top_skill = self.lib._factories[top.name]()
        if not isinstance(top_skill, CommandSkill):
            raise CompileError(
                f"'{top.name}' card says {top_card.interface} but the "
                f"instance is not a CommandSkill")
        top_skill.configure(**top.params)
        self.instances[f"{top.name}#{len(self.instances)}"] = top_skill
        return LayerNode(self.ctx, top.name, top_skill, top_card,
                         self._conditions(top_card.fail_when), base)

    def _modified(self, node: grammar.Modified) -> ModifiedNode:
        until = None
        until_text = ""
        if node.until is not None:
            until = self.lib.conditions.make(node.until.name, *node.until.args)
            until_text = node.until.to_text()
        return ModifiedNode(self.ctx, self.node(node.child),
                            for_s=node.for_s, until=until,
                            until_text=until_text, repeat=node.repeat)
