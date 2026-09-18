"""
Domain randomization: per-env physics perturbations, resampled on reset.

This is the sim-side half of sim-to-real robustness and the surface DrEureka
optimises over: the LLM proposes ranges for these parameters, training
resamples inside them, and the resulting policy tolerates the reality gap.

All parameters are optional (None → untouched). Physics parameters go
through the Articulation DR capabilities (Genesis-backed today); observation
noise and action latency are task-level and consumed by tasks that support
them. Unsupported capabilities are skipped with a one-time warning, so the
same config runs on any backend.

`from_dict` / `to_dict` are the JSON seam used by the LLM loop and by
checkpoints; `to_dict` omits inactive parameters so a config prints as the
minimal set of knobs that are actually on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, fields

import torch

__all__ = ["DomainRandomization"]

_log = logging.getLogger(__name__)


def _u(lo: float, hi: float, n, device) -> torch.Tensor:
    """Uniform draw in [lo, hi) of shape `n` (int or tuple)."""
    return lo + (hi - lo) * torch.rand(n, device=device)


@dataclass
class DomainRandomization:
    # Physics (per env, resampled at reset)
    friction_range: tuple[float, float] | None = None       # ratio, ~1.0
    base_mass_range: tuple[float, float] | None = None      # added kg
    com_shift_range: tuple[float, float] | None = None      # ±m, each xyz axis
    kp_scale_range: tuple[float, float] | None = None       # × nominal kp
    kd_scale_range: tuple[float, float] | None = None       # × nominal kd

    # Task-level (consumed by tasks that support them)
    obs_noise_std: float = 0.0
    action_latency_steps: int = 0

    # Capabilities already reported as unsupported (warn once per instance).
    _warned: set[str] = field(default_factory=set, repr=False, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> DomainRandomization:
        """Build from a JSON-like dict; unknown keys are ignored, lists become tuples."""
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        clean = {}
        for k, v in d.items():
            if k not in known:
                continue
            clean[k] = tuple(v) if isinstance(v, (list, tuple)) else v
        return cls(**clean)

    def to_dict(self) -> dict:
        """JSON-ready dict of the ACTIVE parameters only (None / 0 omitted)."""
        out = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            v = getattr(self, f.name)
            if v is not None and v != 0.0 and v != 0:
                out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    # ------------------------------------------------------------------

    def apply(self, robot, envs_idx: torch.Tensor) -> None:
        """
        Resample physics parameters for the given envs.

        Args:
            robot: a bound `domo.robot.Robot` (uses its articulation, dof_idx
                and actuator gains).
            envs_idx: env indices being reset (empty → no-op).
        """
        if len(envs_idx) == 0:
            return
        n, device = len(envs_idx), robot.device
        art = robot.articulation

        if self.friction_range is not None:
            self._try("friction", art.set_friction_ratio,
                      _u(*self.friction_range, n, device), envs_idx)

        if self.base_mass_range is not None:
            self._try("base_mass", art.set_base_mass_shift,
                      _u(*self.base_mass_range, n, device), envs_idx)

        if self.com_shift_range is not None:
            # Independent offset on each of x, y, z within the range.
            com = _u(*self.com_shift_range, (n, 3), device)
            self._try("com_shift", art.set_base_com_shift, com, envs_idx)

        if self.kp_scale_range is not None or self.kd_scale_range is not None:
            # One draw per reset group (engine gains broadcast over envs);
            # env diversity emerges because envs reset at different times.
            n_dofs = len(robot.dof_idx)
            kp_s = (_u(*self.kp_scale_range, 1, device).item()
                    if self.kp_scale_range else 1.0)
            kd_s = (_u(*self.kd_scale_range, 1, device).item()
                    if self.kd_scale_range else 1.0)
            kp = torch.full((n_dofs,), robot.actuator.kp * kp_s, device=device)
            kd = torch.full((n_dofs,), robot.actuator.kd * kd_s, device=device)
            self._try("pd_gains", art.set_pd_gains_scaled,
                      kp, kd, robot.dof_idx, envs_idx)

    def _try(self, name: str, fn: Callable[..., None], *args) -> None:
        """Call a DR capability; skip (warning once) if the backend lacks it."""
        try:
            fn(*args)
        except NotImplementedError:
            if name not in self._warned:
                self._warned.add(name)
                _log.warning("[dr] backend does not support '%s' — skipped", name)
