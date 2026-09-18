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

from typing import Callable, Dict, List, Optional

import torch

from domo.control.skill import CommandSkill

from . import grammar
from .card import MOTOR, SkillCard
from .conditions import ConditionRegistry, standard_conditions
from .nodes import (CompositeSkill, ExecContext, FallbackNode, LayerNode,
                    ModifiedNode, MotorLeaf, SequenceNode)

__all__ = ["SkillLibrary", "CompileError"]

# Canonical parameter order of the velocity command channel.
VELOCITY_PARAMS = ("vx", "vy", "vyaw")


class CompileError(ValueError):
    pass


class SkillLibrary:

    def __init__(self, conditions: Optional[ConditionRegistry] = None):
        self._cards: Dict[str, SkillCard] = {}
        self._factories: Dict[str, Callable[[], object]] = {}
        self.conditions = conditions or standard_conditions()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, card: SkillCard, factory: Callable[[], object]) -> None:
        """factory() → a fresh Skill/CommandSkill instance for this library."""
        self._cards[card.name] = card
        self._factories[card.name] = factory

    def register_condition(self, name: str, factory) -> None:
        self.conditions.register(name, factory)

    def card(self, name: str) -> SkillCard:
        if name not in self._cards:
            raise CompileError(
                f"unknown skill '{name}' (library has: {sorted(self._cards)})")
        return self._cards[name]

    def names(self) -> List[str]:
        return sorted(self._cards)

    # ------------------------------------------------------------------
    # LLM-facing catalog
    # ------------------------------------------------------------------

    def describe(self) -> str:
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
        ast = grammar.parse(program_text)
        ctx = ExecContext()
        instances: Dict[str, object] = {}
        device = torch.device(device)

        def instance(name):
            if name not in instances:
                instances[name] = self._factories[name]()
            return instances[name]

        def conds(exprs):
            out = []
            for expr in exprs:
                ref = grammar.parse_condition(expr)
                out.append(self.conditions.make(ref.name, *ref.args))
            return out

        def build(node, position="execution"):
            if isinstance(node, grammar.SkillRef):
                card = self.card(node.name)
                if position == "execution" and card.interface != MOTOR:
                    raise CompileError(
                        f"'{node.name}' is a {card.interface} skill — it "
                        f"cannot run alone; layer it: '{node.name} @ "
                        f"<motor skill>'")
                self._check_params(card, node.params)
                if card.interface != MOTOR:
                    return node                       # resolved by Layer parent
                const_cmd, lo, hi = self._command_setup(card, node.params, device)
                return MotorLeaf(ctx, node.name, instance(node.name), card,
                                 const_cmd, lo, hi,
                                 conds(card.success_when),
                                 conds(card.fail_when))

            if isinstance(node, grammar.Layer):
                top = node.top
                if not isinstance(top, grammar.SkillRef):
                    raise CompileError("the left side of '@' must be a single "
                                       "command skill")
                top_card = self.card(top.name)
                if top_card.interface == MOTOR:
                    raise CompileError(
                        f"'{top.name}' is a motor skill — only command skills "
                        f"can be layered on top ('@'). Did you mean "
                        f"'{node.base.to_text()} >> {top.name}'?")
                self._check_params(top_card, top.params)
                base = build(node.base)
                if not isinstance(base, (MotorLeaf, LayerNode)):
                    raise CompileError(
                        f"the right side of '@' must be a (possibly layered) "
                        f"motor skill — apply modifiers to the whole layer "
                        f"instead: '({node.to_text()}).for(...)'")
                channel = top_card.interface.split(":", 1)[1]
                base_card = self.card(base.motor_leaf.name
                                      if isinstance(base, LayerNode)
                                      else base.name)
                if base_card.accepts != channel:
                    raise CompileError(
                        f"cannot layer '{top.name}' (drives '{channel}') on "
                        f"'{base_card.name}' (accepts "
                        f"'{base_card.accepts or 'nothing'}')")
                # Command skills carry per-occurrence goal state (e.g. a nav
                # target), so each '@' gets its own instance — never shared —
                # and the grammar's parameters are applied to it.
                top_skill = self._factories[top.name]()
                if not isinstance(top_skill, CommandSkill):
                    raise CompileError(
                        f"'{top.name}' card says {top_card.interface} but the "
                        f"instance is not a CommandSkill")
                top_skill.configure(**top.params)
                instances[f"{top.name}#{len(instances)}"] = top_skill
                return LayerNode(ctx, top.name, top_skill, top_card,
                                 conds(top_card.fail_when), base)

            if isinstance(node, grammar.Sequence):
                return SequenceNode(ctx, [build(c) for c in node.children])

            if isinstance(node, grammar.Fallback):
                return FallbackNode(ctx, [build(c) for c in node.children])

            if isinstance(node, grammar.Modified):
                until = None
                until_text = ""
                if node.until is not None:
                    until = self.conditions.make(node.until.name,
                                                 *node.until.args)
                    until_text = node.until.to_text()
                return ModifiedNode(ctx, build(node.child),
                                    for_s=node.for_s, until=until,
                                    until_text=until_text, repeat=node.repeat)

            raise CompileError(f"unsupported AST node {type(node).__name__}")

        root = build(ast)
        if isinstance(root, grammar.SkillRef):
            raise CompileError("internal: unresolved skill reference")
        return CompositeSkill(root, ctx, instances, ast.to_text())

    # ------------------------------------------------------------------

    @staticmethod
    def _check_params(card: SkillCard, params: dict):
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
        """Constant command vector + clamp bounds for velocity consumers."""
        if card.accepts != "velocity":
            if params:
                raise CompileError(
                    f"'{card.name}' takes no parameters (got {params})")
            return None, None, None
        defaults = {p.name: p.default for p in card.params}
        defaults.update(params)
        const = torch.tensor([defaults.get(k, 0.0) for k in VELOCITY_PARAMS],
                             device=device, dtype=torch.float32)
        lo = torch.tensor([card.constraints.get(k, (-1e9, 1e9))[0]
                           for k in VELOCITY_PARAMS], device=device)
        hi = torch.tensor([card.constraints.get(k, (-1e9, 1e9))[1]
                           for k in VELOCITY_PARAMS], device=device)
        return const, lo, hi
