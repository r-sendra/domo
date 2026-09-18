"""
Executable composition nodes: the compiled form of a grammar program.

A compiled program is a tree of ExecNodes wrapped in a `CompositeSkill` —
which implements the ordinary Skill interface, so a composition can be
hosted by any Controller/ControlLoop and can itself appear inside a larger
composition later (closure under composition).

Status protocol: every node exposes `.status` ∈ {RUNNING, SUCCESS, FAILURE}
after each update. Containers react to child status at the START of their
next update (sequence advances on SUCCESS, fallback advances on FAILURE).
Program-counter transitions are global across envs (success requires all
envs, failure triggers on any) — compositions execute at deployment
(n_envs = 1); training remains the job of tasks.

Every transition is appended to `ctx.trace` with a timestamp — the
execution log the LLM supervisor reads to diagnose what happened (M4/M5).
"""

from __future__ import annotations

import torch

__all__ = [
    "FAILURE",
    "RUNNING",
    "STATUS_NAMES",
    "SUCCESS",
    "CompositeSkill",
    "ExecContext",
    "ExecNode",
    "FallbackNode",
    "LayerNode",
    "ModifiedNode",
    "MotorLeaf",
    "SequenceNode",
]

RUNNING, SUCCESS, FAILURE = 0, 1, 2
STATUS_NAMES = {RUNNING: "RUNNING", SUCCESS: "SUCCESS", FAILURE: "FAILURE"}


class ExecContext:
    """Shared execution state: clock, trace log, hold-pose targets."""

    def __init__(self):
        self.t = 0.0
        self.trace: list[str] = []
        self._hold = None

    def bind(self, robot):
        self._hold = robot.default_dof_pos.unsqueeze(0).repeat(
            robot.n_envs, 1).clone()

    def hold(self) -> torch.Tensor:
        return self._hold

    def log(self, msg: str):
        self.trace.append(f"[t={self.t:7.2f}s] {msg}")


class ExecNode:
    status: int = RUNNING

    def __init__(self, ctx: ExecContext):
        self.ctx = ctx
        self.t0 = 0.0

    @property
    def t_local(self) -> float:
        return self.ctx.t - self.t0

    def enter(self):
        self.status = RUNNING
        self.t0 = self.ctx.t

    def update(self, state, dt: float) -> torch.Tensor:
        raise NotImplementedError

    def label(self) -> str:
        return type(self).__name__


def _eval_conds(conds, state, t):
    """OR over a list of conditions → bool tensor [N] (or None if empty)."""
    result = None
    for cond in conds:
        flags = cond(state, t)
        result = flags if result is None else (result | flags)
    return result


# ---------------------------------------------------------------------------
# Leaves
# ---------------------------------------------------------------------------

class MotorLeaf(ExecNode):
    """
    A motor skill at an execution position. Applies its constant command
    parameters (plus any layered corrections handed down by a LayerNode,
    clamped to the card's constraints), runs the skill, then evaluates the
    card's success/fail conditions.
    """

    def __init__(self, ctx, name, skill, card, const_cmd: torch.Tensor | None,
                 cmd_lo: torch.Tensor | None, cmd_hi: torch.Tensor | None,
                 success_conds, fail_conds):
        super().__init__(ctx)
        self.name = name
        self.skill = skill
        self.card = card
        self.const_cmd = const_cmd
        self.cmd_lo, self.cmd_hi = cmd_lo, cmd_hi
        self.success_conds = success_conds
        self.fail_conds = fail_conds

    def label(self):
        return self.name

    def enter(self):
        super().enter()
        n = self.skill.robot.n_envs
        self.skill.reset_idx(torch.arange(n, device=self.skill.robot.device))
        self.ctx.log(f"enter {self.name}")

    def update(self, state, dt, extra_cmd: torch.Tensor | None = None):
        if self.const_cmd is not None:
            cmd = self.const_cmd.unsqueeze(0)
            if extra_cmd is not None:
                cmd = cmd + extra_cmd
            if self.cmd_lo is not None:
                cmd = torch.clamp(cmd, self.cmd_lo, self.cmd_hi)
            self.skill.command[:] = cmd
        targets = self.skill.update(state, dt)

        fail = _eval_conds(self.fail_conds, state, self.t_local)
        if fail is not None and bool(fail.any()):
            self.status = FAILURE
            self.ctx.log(f"{self.name} → FAILURE (safety/abort condition)")
        elif self.card.max_duration_s and self.t_local >= self.card.max_duration_s:
            self.status = FAILURE
            self.ctx.log(f"{self.name} → FAILURE (failsafe "
                         f"max_duration {self.card.max_duration_s}s)")
        else:
            done = _eval_conds(self.success_conds, state, self.t_local)
            if done is not None and bool(done.all()):
                self.status = SUCCESS
                self.ctx.log(f"{self.name} → SUCCESS")
        return targets


class LayerNode(ExecNode):
    """
    'top @ base' — top is a CommandSkill whose output is accumulated into
    the base's command channel each control cycle. The base clamps the
    combined command to its card constraints, so no layer can exceed them.
    """

    def __init__(self, ctx, top_name, top_skill, top_card, top_fail_conds, base):
        super().__init__(ctx)
        self.top_name = top_name
        self.top_skill = top_skill
        self.top_card = top_card
        self.top_fail_conds = top_fail_conds
        self.base = base

    def label(self):
        return f"{self.top_name} @ {self.base.label()}"

    @property
    def motor_leaf(self) -> MotorLeaf:
        node = self.base
        while isinstance(node, LayerNode):
            node = node.base
        return node

    def enter(self):
        super().enter()
        n = self.top_skill.robot.n_envs
        self.top_skill.reset_idx(
            torch.arange(n, device=self.top_skill.robot.device))
        self.base.enter()
        self.ctx.log(f"layer {self.top_name} active")

    def update(self, state, dt, extra_cmd: torch.Tensor | None = None):
        delta = self.top_skill.update_command(state, dt)
        if not self.top_skill.additive:
            # Override mode: cancel the motor leaf's constant command so the
            # effective command equals this skill's output (plus any layers
            # stacked above this one).
            delta = delta - self.motor_leaf.const_cmd.unsqueeze(0)
        total = delta if extra_cmd is None else delta + extra_cmd
        targets = self.base.update(state, dt, extra_cmd=total)

        self.status = self.base.status
        # A goal-directed command skill (e.g. navigation) can complete the
        # layer on its own once every env reaches the goal.
        done = self.top_skill.success_flags(state)
        if self.status == RUNNING and done is not None and bool(done.all()):
            self.status = SUCCESS
            self.ctx.log(f"{self.top_name} → SUCCESS (goal reached)")
        fail = _eval_conds(self.top_fail_conds, state, self.t_local)
        if fail is not None and bool(fail.any()):
            self.status = FAILURE
            self.ctx.log(f"{self.top_name} → FAILURE (layer abort condition)")
        return targets


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

class SequenceNode(ExecNode):
    """children run in order; advance on SUCCESS; any FAILURE fails the seq."""

    def __init__(self, ctx, children):
        super().__init__(ctx)
        self.children = children
        self.i = 0

    def label(self):
        return " >> ".join(c.label() for c in self.children)

    def enter(self):
        super().enter()
        self.i = 0
        self.children[0].enter()

    def update(self, state, dt):
        child = self.children[self.i]
        if child.status == FAILURE:
            self.status = FAILURE
            return self.ctx.hold()
        if child.status == SUCCESS:
            if self.i + 1 < len(self.children):
                self.i += 1
                child = self.children[self.i]
                self.ctx.log(f"sequence → step {self.i + 1}/"
                             f"{len(self.children)} ({child.label()})")
                child.enter()
            else:
                self.status = SUCCESS
                return self.ctx.hold()
        targets = child.update(state, dt)
        if child.status == FAILURE:
            # Fast-path: a failure below propagates the same control tick
            # (a Fallback ancestor still intercepts it on its next update).
            self.status = FAILURE
        return targets


class FallbackNode(ExecNode):
    """children tried in order; advance on FAILURE; first SUCCESS succeeds."""

    def __init__(self, ctx, children):
        super().__init__(ctx)
        self.children = children
        self.i = 0

    def label(self):
        return " | ".join(c.label() for c in self.children)

    def enter(self):
        super().enter()
        self.i = 0
        self.children[0].enter()

    def update(self, state, dt):
        child = self.children[self.i]
        if child.status == SUCCESS:
            self.status = SUCCESS
            return self.ctx.hold()
        if child.status == FAILURE:
            if self.i + 1 < len(self.children):
                self.i += 1
                child = self.children[self.i]
                self.ctx.log(f"fallback → alternative {self.i + 1}/"
                             f"{len(self.children)} ({child.label()})")
                child.enter()
            else:
                self.status = FAILURE
                return self.ctx.hold()
        return child.update(state, dt)


class ModifiedNode(ExecNode):
    """.for(T) / .until(cond) / .repeat(n) wrapper."""

    def __init__(self, ctx, child, for_s=None, until=None, until_text="",
                 repeat=None):
        super().__init__(ctx)
        self.child = child
        self.for_s = for_s
        self.until = until
        self.until_text = until_text
        self.repeat = repeat
        self._runs = 0

    def label(self):
        t = self.child.label()
        if self.for_s is not None:
            t += f".for({self.for_s:g})"
        if self.until is not None:
            t += f".until({self.until_text})"
        if self.repeat is not None:
            t += f".repeat({self.repeat})"
        return t

    def enter(self):
        super().enter()
        self._runs = 0
        self.child.enter()

    def update(self, state, dt):
        # Own success criteria first (checked on the fresh state).
        if self.for_s is not None and self.t_local >= self.for_s:
            self.status = SUCCESS
            self.ctx.log(f"{self.child.label()} → SUCCESS (.for {self.for_s:g}s)")
            return self.ctx.hold()
        if self.until is not None:
            flags = self.until(state, self.t_local)
            if bool(flags.all()):
                self.status = SUCCESS
                self.ctx.log(f"{self.child.label()} → SUCCESS "
                             f"(.until {self.until_text})")
                return self.ctx.hold()

        if self.child.status == FAILURE:
            self.status = FAILURE
            return self.ctx.hold()
        if self.child.status == SUCCESS:
            self._runs += 1
            if self.repeat is not None and self._runs < self.repeat:
                self.ctx.log(f"repeat {self._runs + 1}/{self.repeat} "
                             f"({self.child.label()})")
                self.child.enter()
            else:
                self.status = SUCCESS
                return self.ctx.hold()
        targets = self.child.update(state, dt)
        if self.child.status == FAILURE:
            self.status = FAILURE          # same-tick failure propagation
        return targets


# ---------------------------------------------------------------------------
# CompositeSkill — a compiled program IS a Skill
# ---------------------------------------------------------------------------

class CompositeSkill:
    """
    Skill-interface wrapper around a compiled program tree. Terminal states
    hold the default stance; the hosting Controller reads `.status` /
    `.finished` and the `.trace` log to decide what happens next.
    """

    def __init__(self, root: ExecNode, ctx: ExecContext, instances: dict,
                 source: str):
        self.name = f"program[{source}]"
        self.root = root
        self.ctx = ctx
        self.instances = instances
        self.source = source
        self.robot = None

    # -- Skill interface ---------------------------------------------------

    def setup(self, robot):
        self.robot = robot
        for skill in self.instances.values():
            skill.setup(robot)
        self.ctx.bind(robot)

    def reset_idx(self, envs_idx):
        for skill in self.instances.values():
            skill.reset_idx(envs_idx)
        self.ctx.t = 0.0
        self.ctx.trace.clear()
        self.ctx.log(f"start: {self.source}")
        self.root.enter()

    def update(self, state, dt: float):
        if self.root.status != RUNNING:
            return self.ctx.hold()
        targets = self.root.update(state, dt)
        self.ctx.t += dt
        if self.root.status != RUNNING:
            self.ctx.log(f"program → {STATUS_NAMES[self.root.status]}")
        return targets

    # -- Introspection ------------------------------------------------------

    @property
    def status(self) -> int:
        return self.root.status

    @property
    def finished(self) -> bool:
        return self.root.status != RUNNING

    @property
    def succeeded(self) -> bool:
        return self.root.status == SUCCESS

    @property
    def trace(self):
        return list(self.ctx.trace)
