"""
Conditions: named boolean predicates over robot state, used by the grammar
(`.until(cond)`), by SkillCards (`success_when` / `fail_when`), and by
controllers.

A condition factory takes literal arguments from the program text and
returns a callable `cond(state, t_s) -> BoolTensor [n_envs]` (t_s = seconds
since the enclosing node was entered). Factories that need sensors (e.g.
lidar clearance) are registered as closures over the sensor object.

Standard registry (see also SkillLibrary.register_condition):
  timeout(T)        — true after T seconds in the current node
  tipped(rad=0.7)   — |roll| or |pitch| beyond limit
  fallen(h=0.18)    — base height below limit
  still(v=0.05)     — base speed below limit
  moved(d)          — displaced more than d metres from node entry (needs
                      privileged/estimated position)
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ["Condition", "ConditionRegistry", "standard_conditions"]

Condition = Callable[[object, float], torch.Tensor]   # (state, t_s) -> bool [N]


class ConditionRegistry:
    """Name → factory map; `make(name, *args)` instantiates a Condition."""

    def __init__(self):
        self._factories: dict[str, Callable[..., Condition]] = {}

    def register(self, name: str, factory: Callable[..., Condition]) -> None:
        """Register `factory(*args) -> Condition` under `name` (replaces)."""
        self._factories[name] = factory

    def names(self):
        """Registered condition names, sorted (listed in library.describe())."""
        return sorted(self._factories)

    def make(self, name: str, *args) -> Condition:
        """Instantiate a condition from grammar literals.

        Raises:
            KeyError: unknown condition name.
        """
        if name not in self._factories:
            raise KeyError(f"unknown condition '{name}' "
                           f"(known: {self.names()})")
        return self._factories[name](*args)


# ---------------------------------------------------------------------------
# Standard conditions (state-only; sensor-bound ones are added by the caller)
# ---------------------------------------------------------------------------

def _timeout(t_limit: float) -> Condition:
    def cond(state, t_s):
        n = state.base_pos.shape[0]
        return torch.full((n,), t_s >= t_limit, dtype=torch.bool,
                          device=state.base_pos.device)
    return cond


def _tipped(limit: float = 0.7) -> Condition:
    def cond(state, t_s):
        return state.base_euler[:, :2].abs().amax(dim=1) > limit
    return cond


def _fallen(height: float = 0.18) -> Condition:
    def cond(state, t_s):
        return state.base_pos[:, 2] < height
    return cond


def _still(v_limit: float = 0.05) -> Condition:
    def cond(state, t_s):
        return state.base_lin_vel.norm(dim=1) < v_limit
    return cond


class _Moved:
    """Stateful: latches the entry position on first evaluation after reset.

    Conditions get no reset() call, so re-entry of the enclosing node is
    detected from the node-local clock going backwards (t_s < last t_s).
    """

    def __init__(self, distance: float):
        self.distance = distance
        self._origin = None
        self._last_t = None

    def __call__(self, state, t_s):
        if self._origin is None or (self._last_t is not None and t_s < self._last_t):
            self._origin = state.base_pos[:, :2].clone()
        self._last_t = t_s
        d = (state.base_pos[:, :2] - self._origin).norm(dim=1)
        return d > self.distance


def standard_conditions() -> ConditionRegistry:
    """The state-only conditions listed in the module docstring."""
    reg = ConditionRegistry()
    reg.register("timeout", _timeout)
    reg.register("tipped", _tipped)
    reg.register("fallen", _fallen)
    reg.register("still", _still)
    reg.register("moved", _Moved)
    return reg
