"""
Walled obstacle arena with randomised household-like obstacles — the
training/eval ground of the avoidance experiments, extracted from the task
so any consumer (VecTask training, composed-skill evaluation, future
LLM-generated scenes) builds the identical world.

Two-phase like everything scene-related: construct before `scene.build()`,
call `randomise()` after it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from domo.sim.base import Scene

__all__ = ["ObstacleArena", "ObstacleArenaConfig"]


def _rand(lo, hi, shape, device):
    return (hi - lo) * torch.rand(size=shape, device=device) + lo


@dataclass
class ObstacleArenaConfig:
    """Walled square arena with randomised household-like obstacles."""
    half_size: float = 4.0
    wall_height: float = 0.6
    wall_thickness: float = 0.15
    ring_min: float = 2.0       # obstacle spawn ring around robot spawn
    ring_max: float = 3.5

    n_chairs: int = 6
    n_sofas: int = 3
    n_pillars: int = 4
    n_steps: int = 2
    n_balls: int = 2

    chair_leg_radius: float = 0.03
    chair_leg_height: float = 0.45
    chair_leg_spread: float = 0.25
    sofa_width_range: tuple[float, float] = (0.8, 1.6)
    sofa_depth_range: tuple[float, float] = (0.3, 0.5)
    sofa_height_range: tuple[float, float] = (0.35, 0.50)
    pillar_radius_range: tuple[float, float] = (0.04, 0.10)
    pillar_height_range: tuple[float, float] = (0.60, 1.20)
    step_size_range: tuple[float, float] = (0.20, 0.50)
    step_height_range: tuple[float, float] = (0.04, 0.15)
    ball_radius_range: tuple[float, float] = (0.06, 0.16)


class ObstacleArena:
    """Builds walls + obstacle entities; randomises obstacle poses per env."""

    def __init__(self, scene: Scene, cfg: ObstacleArenaConfig,
                 spawn_xy: tuple[float, float] = (0.0, 0.0)):
        self.cfg = cfg
        self.spawn_xy = spawn_xy
        self.termination_distance = cfg.half_size + 0.5
        self._obstacles = []

        s, t, h = cfg.half_size, cfg.wall_thickness, cfg.wall_height
        for wx, wy, sx, sy in [
            (0.0, s, 2 * s + 2 * t, t),
            (0.0, -s, 2 * s + 2 * t, t),
            (s, 0.0, t, 2 * s),
            (-s, 0.0, t, 2 * s),
        ]:
            scene.add_box(size=(sx, sy, h), pos=(wx, wy, h / 2), fixed=True)

        def lerp(rng, i, n):
            f = i / max(n - 1, 1)
            return rng[0] + (rng[1] - rng[0]) * f

        FAR = (99.0, 99.0)          # parked until randomise()
        leg_offsets = [
            (cfg.chair_leg_spread, cfg.chair_leg_spread),
            (cfg.chair_leg_spread, -cfg.chair_leg_spread),
            (-cfg.chair_leg_spread, cfg.chair_leg_spread),
            (-cfg.chair_leg_spread, -cfg.chair_leg_spread),
        ]
        for _ in range(cfg.n_chairs):
            legs = [scene.add_cylinder(
                radius=cfg.chair_leg_radius, height=cfg.chair_leg_height,
                pos=(*FAR, cfg.chair_leg_height / 2), fixed=True)
                for _ in leg_offsets]
            self._obstacles.append(
                ("chair", legs, cfg.chair_leg_height, leg_offsets))

        for i in range(cfg.n_sofas):
            w = lerp(cfg.sofa_width_range, i, cfg.n_sofas)
            d = lerp(cfg.sofa_depth_range, i, cfg.n_sofas)
            hh = lerp(cfg.sofa_height_range, i, cfg.n_sofas)
            e = scene.add_box(size=(w, d, hh), pos=(*FAR, hh / 2), fixed=True)
            self._obstacles.append(("sofa", e, hh, None))

        for i in range(cfg.n_pillars):
            r = lerp(cfg.pillar_radius_range, i, cfg.n_pillars)
            hh = lerp(cfg.pillar_height_range, i, cfg.n_pillars)
            e = scene.add_cylinder(radius=r, height=hh,
                                   pos=(*FAR, hh / 2), fixed=True)
            self._obstacles.append(("pillar", e, hh, None))

        for i in range(cfg.n_steps):
            ss = lerp(cfg.step_size_range, i, cfg.n_steps)
            hh = lerp(cfg.step_height_range, i, cfg.n_steps)
            e = scene.add_box(size=(ss, ss, hh), pos=(*FAR, hh / 2), fixed=True)
            self._obstacles.append(("step", e, hh, None))

        for i in range(cfg.n_balls):
            r = lerp(cfg.ball_radius_range, i, cfg.n_balls)
            e = scene.add_sphere(radius=r, pos=(*FAR, r), fixed=True)
            self._obstacles.append(("ball", e, r * 2, None))

    def randomise(self, envs_idx: torch.Tensor, device) -> None:
        """Re-scatter all obstacles in the spawn ring for the given envs."""
        if len(envs_idx) == 0:
            return
        cfg = self.cfg
        n = len(envs_idx)
        sx, sy = self.spawn_xy
        for obs_type, entity, height, meta in self._obstacles:
            angles = _rand(0.0, 2 * math.pi, (n,), device)
            radii = _rand(cfg.ring_min, cfg.ring_max, (n,), device)
            cx = sx + radii * torch.cos(angles)
            cy = sy + radii * torch.sin(angles)
            if obs_type == "chair":
                yaw = _rand(0.0, 2 * math.pi, (n,), device)
                for leg, (dx, dy) in zip(entity, meta):
                    lx = cx + dx * torch.cos(yaw) - dy * torch.sin(yaw)
                    ly = cy + dx * torch.sin(yaw) + dy * torch.cos(yaw)
                    lz = torch.full((n,), height / 2, device=device)
                    leg.set_position(torch.stack([lx, ly, lz], dim=-1),
                                     envs_idx=envs_idx)
            else:
                z = torch.full((n,), height / 2, device=device)
                entity.set_position(torch.stack([cx, cy, z], dim=-1),
                                    envs_idx=envs_idx)
