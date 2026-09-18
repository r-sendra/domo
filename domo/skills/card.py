"""
SkillCard: the structured, LLM-legible description of one skill.

Every entry in the skill library (M5) carries a card holding everything a
planner — human or LLM — needs to decide when and how to use the skill:
natural-language description, typed parameters, interface type, termination
conditions, command constraints, and safety limits. `describe()` renders
the card as the prompt block the M1/M5 reasoning consumes.

Interface typing is the grammar's type system:
  * "motor"              — emits joint targets; can terminate a composition
  * "command:velocity"   — emits (Δvx, Δvy, Δvyaw); must be layered `@` on
                           a motor skill that `accepts` the same channel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["ParamSpec", "SkillCard", "MOTOR", "CMD_VELOCITY"]

MOTOR = "motor"
CMD_VELOCITY = "command:velocity"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    description: str
    default: float = 0.0
    range: Optional[Tuple[float, float]] = None
    unit: str = ""


@dataclass
class SkillCard:
    name: str
    description: str                      # natural language, one paragraph
    interface: str = MOTOR                # MOTOR | CMD_VELOCITY
    accepts: Optional[str] = None         # command channel consumed (motor skills)
    params: List[ParamSpec] = field(default_factory=list)

    # Semantics for the planner (natural language; not machine-checked).
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)

    # Termination — condition expressions in the grammar's condition syntax,
    # evaluated by the executor every decision tick.
    success_when: List[str] = field(default_factory=list)   # [] → runs until a modifier ends it
    fail_when: List[str] = field(default_factory=list)      # safety/abort conditions

    # Hard limits on the command channel this skill CONSUMES (clamped by the
    # executor after layering, before the skill sees the command).
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Safety documentation + failsafe duration (FAILURE after this long).
    safety_notes: List[str] = field(default_factory=list)
    max_duration_s: Optional[float] = None

    # ------------------------------------------------------------------

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"skill '{self.name}' has no parameter '{name}'")

    def describe(self) -> str:
        """Render as an LLM prompt block."""
        if self.interface != MOTOR:
            channel = self.interface.split(":", 1)[1]
            head = f", drives the '{channel}' channel of a base skill]"
        else:
            head = f", accepts '{self.accepts}']" if self.accepts else "]"
        lines = [f"SKILL {self.name}  [{self.interface}" + head,
                 f"  {self.description}"]
        if self.params:
            lines.append("  parameters:")
            for p in self.params:
                rng = f" range={list(p.range)}" if p.range else ""
                unit = f" [{p.unit}]" if p.unit else ""
                lines.append(f"    - {p.name}={p.default}{unit}{rng}: {p.description}")
        if self.preconditions:
            lines.append("  preconditions: " + "; ".join(self.preconditions))
        if self.effects:
            lines.append("  effects: " + "; ".join(self.effects))
        lines.append("  succeeds when: "
                     + ("; ".join(self.success_when) if self.success_when
                        else "(never by itself — bound its duration with "
                             ".for(seconds) or .until(condition))"))
        if self.fail_when:
            lines.append("  fails when: " + "; ".join(self.fail_when))
        if self.constraints:
            lines.append("  command constraints: " + ", ".join(
                f"{k}∈[{lo},{hi}]" for k, (lo, hi) in self.constraints.items()))
        if self.safety_notes:
            lines.append("  safety: " + "; ".join(self.safety_notes))
        if self.max_duration_s:
            lines.append(f"  failsafe: aborts after {self.max_duration_s}s")
        return "\n".join(lines)
