"""
PlanningController: the Controller whose behaviour is AUTHORED at runtime.

This closes the loop the architecture was built for: skill programs are not
configuration handed in from outside — they are produced *inside* the
controller by whatever planning intelligence occupies the `plan()` slot:

    class Patrol(PlanningController):
        def plan(self, state, last):
            if last is None:                       # first mission
                return "(avoid @ walk(vx=0.6)).until(moved(4)) >> stand.for(1)"
            if not last.succeeded:                 # react to what happened
                return "stand.for(3)"              # cool off, retry later
            return None                            # nothing to do → idle

Today `plan()` is researcher code. In the DOMO roadmap it is the LLM/HRL
seat (M1/M5): the supervisor reads `library.describe()` and the previous
`PlanOutcome` (program text + success + execution trace) and returns the
next program — or, one level up, generates the whole PlanningController
subclass. The loop below stays a dumb metronome either way; planning lives
here, in the Controller layer, by design.

Lifecycle: `plan()` is consulted (at the controller's decision cadence)
whenever there is no active program or the active one just finished.
Returning a program string compiles + installs it; returning None idles in
the fallback skill (stand) until a later `plan()` call returns something.
Compile errors don't crash the robot: they are recorded on the outcome and
`plan()` is asked again — the same feedback path an LLM needs to repair
its own programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from domo.control import Controller, StandSkill

from .library import CompileError, SkillLibrary
from .grammar import GrammarError

__all__ = ["PlanOutcome", "PlanningController"]


@dataclass
class PlanOutcome:
    """What the planner learns about its previous decision."""
    program: str
    succeeded: bool
    trace: List[str] = field(default_factory=list)
    compile_error: Optional[str] = None    # program never ran


class PlanningController(Controller):

    IDLE = "__idle__"

    def __init__(self, library: SkillLibrary, decision_interval: int = 10):
        super().__init__({self.IDLE: StandSkill()}, initial=self.IDLE,
                         decision_interval=decision_interval)
        self.library = library
        self.program = None                    # active CompositeSkill
        self.history: List[PlanOutcome] = []   # everything that happened

    # ------------------------------------------------------------------
    # The intelligence slot
    # ------------------------------------------------------------------

    def plan(self, state, last: Optional[PlanOutcome]) -> Optional[str]:
        """
        Produce the next skill program (grammar text), or None to idle.
        `last` is the outcome of the previous program (None on the very
        first call, or after an idle period with no program).
        Override this — it is where LLM / HRL / researcher logic lives.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Machinery
    # ------------------------------------------------------------------

    def decide(self, state) -> None:
        last = None
        if self.program is not None and self.program.finished:
            last = PlanOutcome(self.program.source, self.program.succeeded,
                               self.program.trace)
            self.history.append(last)
            self.program = None
            self.activate(self.IDLE)

        if self.program is None:
            text = self.plan(state, last)
            if text:
                self._install(text)

    def _install(self, text: str) -> None:
        try:
            program = self.library.compile(text, device=self.robot.device)
        except (CompileError, GrammarError) as e:
            # Bad program (e.g. from an LLM): record, stay idle, let the
            # next plan() call try again with the error in hand.
            self.history.append(PlanOutcome(text, False, compile_error=str(e)))
            print(f"  [planner] compile error: {e}")
            return
        program.setup(self.robot)
        self.skills["program"] = program
        program.reset_idx(torch.arange(self.robot.n_envs,
                                       device=self.robot.device))
        self.active = "program"
        self.program = program

    @property
    def idle(self) -> bool:
        return self.program is None

    @property
    def last_outcome(self) -> Optional[PlanOutcome]:
        return self.history[-1] if self.history else None
