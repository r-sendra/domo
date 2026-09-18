"""
Controller: the programmable orchestration layer of the control hierarchy.

This is the slot where a researcher — and, in the DOMO roadmap, the LLM
(M1/M5) — writes behaviour as ordinary code: which skill runs, with what
parameters, switching on what conditions. Subclass and implement `decide()`;
the base class handles the decision cadence and delegates the 50 Hz motor
work to the active skill.

    class Patrol(Controller):
        def decide(self, state):
            if state.base_euler[:, 0].abs().max() > 0.5:
                self.activate("stand")            # reflex: about to tip
            elif self._phase_elapsed() > 4.0:
                self.skills["walk"].command[:, 2] = 0.8   # turn
            ...

Decisions run every `decision_interval` control steps (1–10 Hz typical);
`update()` runs every step. One skill is active at a time — for all envs —
which fits the intended use (n_envs=1 evaluation / deployment). Blending or
per-env skill selection can subclass `update()`.

LLM-generated behaviour (Paper 3 composition) targets exactly this API:
the generated program is a Controller subclass calling the skill library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .skill import Skill, all_envs

__all__ = ["Controller", "SingleSkillController"]


class Controller(ABC):
    """Programmable skill orchestrator: owns a named set of skills, one active.

    Args:
        skills: name → Skill (or CompositeSkill) instances this controller
            may activate; all are set up and reset together.
        initial: name of the skill active after construction.
        decision_interval: control steps between `decide()` calls (≥ 1).

    Raises:
        ValueError: `initial` is not one of `skills`.
    """

    def __init__(self, skills: dict[str, Skill], initial: str,
                 decision_interval: int = 5):
        if initial not in skills:
            raise ValueError(f"initial skill '{initial}' not in {list(skills)}")
        self.skills = skills
        self.active = initial
        self.decision_interval = max(1, decision_interval)
        self._tick = 0
        self.robot = None

    # ------------------------------------------------------------------
    # Lifecycle (mirrors Skill)
    # ------------------------------------------------------------------

    def setup(self, robot) -> None:
        """Bind every skill to `robot` (after robot.bind())."""
        self.robot = robot
        for skill in self.skills.values():
            skill.setup(robot)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset every skill for `envs_idx` and restart the decision clock."""
        for skill in self.skills.values():
            skill.reset_idx(envs_idx)
        self._tick = 0

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def activate(self, name: str) -> None:
        """Switch the active skill; the newcomer starts from a clean state."""
        if name == self.active:
            return
        if name not in self.skills:
            raise KeyError(f"unknown skill '{name}' (have {list(self.skills)})")
        self.skills[name].reset_idx(all_envs(self.robot))
        self.active = name

    @abstractmethod
    def decide(self, state) -> None:
        """
        The programmable brain: inspect state (and any exteroceptive sensors
        the subclass holds), then activate skills / set their parameters.
        Called every `decision_interval` control steps.
        """

    def update(self, state, dt: float) -> torch.Tensor:
        """One control cycle: (maybe) decide, then delegate to the active skill.

        Returns:
            Joint position targets [N, D] from the active skill.
        """
        if self._tick % self.decision_interval == 0:
            self.decide(state)
        self._tick += 1
        return self.skills[self.active].update(state, dt)

    @property
    def ticks(self) -> int:
        """Control steps since the last reset (time = ticks × dt)."""
        return self._tick


class SingleSkillController(Controller):
    """Run exactly one skill forever (e.g. a compiled CompositeSkill)."""

    def __init__(self, skill: Skill, name: str | None = None):
        name = name or skill.name
        super().__init__({name: skill}, initial=name, decision_interval=1)

    def decide(self, state) -> None:
        """Nothing to decide: the single skill stays active."""
