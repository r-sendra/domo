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

from domo.control import Controller, StandSkill, all_envs

from .grammar import GrammarError
from .library import CompileError, SkillLibrary

__all__ = ["PlanOutcome", "PlanningController"]


@dataclass
class PlanOutcome:
    """What the planner learns about its previous decision."""
    program: str
    succeeded: bool
    trace: list[str] = field(default_factory=list)
    compile_error: str | None = None    # program never ran


class PlanningController(Controller):
    """Controller that compiles and runs programs returned by `plan()`.

    Skills: the idle StandSkill (`IDLE`) plus, while one runs, the active
    CompositeSkill under the name "program".

    Args:
        library: the SkillLibrary programs are compiled against.
        decision_interval: control steps between `plan()` consultations.
    """

    IDLE = "__idle__"
    PROGRAM = "program"

    def __init__(self, library: SkillLibrary, decision_interval: int = 10):
        super().__init__({self.IDLE: StandSkill()}, initial=self.IDLE,
                         decision_interval=decision_interval)
        self.library = library
        self.program = None                    # active CompositeSkill
        self.history: list[PlanOutcome] = []   # everything that happened

    # ------------------------------------------------------------------
    # The intelligence slot
    # ------------------------------------------------------------------

    def plan(self, state, last: PlanOutcome | None) -> str | None:
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
        """Close out a finished program, then ask `plan()` if nothing is running."""
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
        """Compile `text` and make it the active skill; record compile errors."""
        try:
            program = self.library.compile(text, device=self.robot.device)
        except (CompileError, GrammarError) as e:
            # Bad program (e.g. from an LLM): record, stay idle, let the
            # next plan() call try again with the error in hand.
            self.history.append(PlanOutcome(text, False, compile_error=str(e)))
            print(f"  [planner] compile error: {e}")
            return
        program.setup(self.robot)
        self.skills[self.PROGRAM] = program
        # Installed directly rather than via activate(): the program must be
        # reset even when it replaces a previous "program" entry of the same name.
        program.reset_idx(all_envs(self.robot))
        self.active = self.PROGRAM
        self.program = program

    @property
    def idle(self) -> bool:
        """True while no program is running (holding the stand pose)."""
        return self.program is None

    @property
    def last_outcome(self) -> PlanOutcome | None:
        """Most recent PlanOutcome (finished or failed-to-compile), if any."""
        return self.history[-1] if self.history else None
