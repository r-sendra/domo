"""
Walled obstacle arena with randomised household-like obstacles — the
training/eval ground of the avoidance experiments, extracted from the task
so any consumer (VecTask training, composed-skill evaluation, future
LLM-generated scenes) builds the identical world.

Two-phase like everything scene-related: construct before `scene.build()`,
call `randomise()` after it. All obstacles are FIXED bodies (the robot
cannot push them) and exist in every env; per-env variety comes purely from
`randomise()` teleporting them, since entity counts are fixed at build time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from domo.sim.base import RigidObject, Scene

__all__ = ["ObstacleArena", "ObstacleArenaConfig"]

# Obstacles are created here (far outside the arena) and only enter it on
# the first `randomise()`; a fresh build therefore has an empty arena.
_PARKED_XY = (99.0, 99.0)

# The robot is considered "out of the arena" this far beyond the walls.
_TERMINATION_MARGIN = 0.5


def _rand(lo: float, hi: float, shape, device) -> torch.Tensor:
    """Uniform draw in [lo, hi) of the given shape."""
    return (hi - lo) * torch.rand(size=shape, device=device) + lo


def _lerp(rng: tuple[float, float], i: int, n: int) -> float:
    """Evenly space item i of n across `rng` (single item → rng[0])."""
    f = i / max(n - 1, 1)
    return rng[0] + (rng[1] - rng[0]) * f


class _Obstacle(NamedTuple):
    kind: str                                   # "chair" | "sofa" | "pillar" | "step" | "ball"
    entities: list[RigidObject]                 # one body, or 4 legs for a chair
    height: float                               # full height (m); centre sits at height/2
    leg_offsets: list[tuple[float, float]]      # chair legs in the chair frame (m)


@dataclass
class ObstacleArenaConfig:
    """Walled square arena with randomised household-like obstacles (metres)."""
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

    # Chairs are 4 thin legs (lidar sees them as sparse returns, like real chairs).
    chair_leg_radius: float = 0.03
    chair_leg_height: float = 0.45
    chair_leg_spread: float = 0.25
    # Size ranges are spread evenly over the n instances (not sampled).
    sofa_width_range: tuple[float, float] = (0.8, 1.6)
    sofa_depth_range: tuple[float, float] = (0.3, 0.5)
    sofa_height_range: tuple[float, float] = (0.35, 0.50)
    pillar_radius_range: tuple[float, float] = (0.04, 0.10)
    pillar_height_range: tuple[float, float] = (0.60, 1.20)
    step_size_range: tuple[float, float] = (0.20, 0.50)
    step_height_range: tuple[float, float] = (0.04, 0.15)
    ball_radius_range: tuple[float, float] = (0.06, 0.16)


class ObstacleArena:
    """
    Builds walls + obstacle entities; randomises obstacle poses per env.

    Args:
        scene: un-built scene to add the bodies to.
        cfg: arena layout.
        spawn_xy: robot spawn (m); the obstacle ring is centred on it.

    Attributes:
        termination_distance: distance from the origin beyond which a task
            should consider the robot to have left the arena.
    """

    def __init__(self, scene: Scene, cfg: ObstacleArenaConfig,
                 spawn_xy: tuple[float, float] = (0.0, 0.0)):
        self.cfg = cfg
        self.spawn_xy = spawn_xy
        self.termination_distance = cfg.half_size + _TERMINATION_MARGIN
        self._obstacles: list[_Obstacle] = []
        self._build_walls(scene)
        self._build_obstacles(scene)

    # -- construction (before build) -----------------------------------------

    def _build_walls(self, scene: Scene) -> None:
        s, t, h = self.cfg.half_size, self.cfg.wall_thickness, self.cfg.wall_height
        # (centre x, centre y, size x, size y); the ±y walls overlap the corners.
        for wx, wy, sx, sy in [
            (0.0, s, 2 * s + 2 * t, t),
            (0.0, -s, 2 * s + 2 * t, t),
            (s, 0.0, t, 2 * s),
            (-s, 0.0, t, 2 * s),
        ]:
            scene.add_box(size=(sx, sy, h), pos=(wx, wy, h / 2), fixed=True)

    def _build_obstacles(self, scene: Scene) -> None:
        cfg = self.cfg
        leg_offsets = [
            (cfg.chair_leg_spread, cfg.chair_leg_spread),
            (cfg.chair_leg_spread, -cfg.chair_leg_spread),
            (-cfg.chair_leg_spread, cfg.chair_leg_spread),
            (-cfg.chair_leg_spread, -cfg.chair_leg_spread),
        ]
        for _ in range(cfg.n_chairs):
            legs = [scene.add_cylinder(
                radius=cfg.chair_leg_radius, height=cfg.chair_leg_height,
                pos=(*_PARKED_XY, cfg.chair_leg_height / 2), fixed=True)
                for _ in leg_offsets]
            self._obstacles.append(
                _Obstacle("chair", legs, cfg.chair_leg_height, leg_offsets))

        for i in range(cfg.n_sofas):
            w = _lerp(cfg.sofa_width_range, i, cfg.n_sofas)
            d = _lerp(cfg.sofa_depth_range, i, cfg.n_sofas)
            hh = _lerp(cfg.sofa_height_range, i, cfg.n_sofas)
            e = scene.add_box(size=(w, d, hh), pos=(*_PARKED_XY, hh / 2), fixed=True)
            self._obstacles.append(_Obstacle("sofa", [e], hh, []))

        for i in range(cfg.n_pillars):
            r = _lerp(cfg.pillar_radius_range, i, cfg.n_pillars)
            hh = _lerp(cfg.pillar_height_range, i, cfg.n_pillars)
            e = scene.add_cylinder(radius=r, height=hh,
                                   pos=(*_PARKED_XY, hh / 2), fixed=True)
            self._obstacles.append(_Obstacle("pillar", [e], hh, []))

        for i in range(cfg.n_steps):
            ss = _lerp(cfg.step_size_range, i, cfg.n_steps)
            hh = _lerp(cfg.step_height_range, i, cfg.n_steps)
            e = scene.add_box(size=(ss, ss, hh), pos=(*_PARKED_XY, hh / 2), fixed=True)
            self._obstacles.append(_Obstacle("step", [e], hh, []))

        for i in range(cfg.n_balls):
            r = _lerp(cfg.ball_radius_range, i, cfg.n_balls)
            e = scene.add_sphere(radius=r, pos=(*_PARKED_XY, r), fixed=True)
            self._obstacles.append(_Obstacle("ball", [e], r * 2, []))

    # -- randomisation (after build) -----------------------------------------

    def randomise(self, envs_idx: torch.Tensor, device) -> None:
        """
        Re-scatter all obstacles in the spawn ring for the given envs.

        Each obstacle draws an independent polar position (uniform angle,
        uniform radius in [ring_min, ring_max]); chairs also draw a yaw.
        Uses torch's global RNG — seed with `torch.manual_seed` to reproduce.
        """
        if len(envs_idx) == 0:
            return
        cfg = self.cfg
        n = len(envs_idx)
        sx, sy = self.spawn_xy
        for obs in self._obstacles:
            angles = _rand(0.0, 2 * math.pi, (n,), device)
            radii = _rand(cfg.ring_min, cfg.ring_max, (n,), device)
            cx = sx + radii * torch.cos(angles)
            cy = sy + radii * torch.sin(angles)
            z = torch.full((n,), obs.height / 2, device=device)
            if obs.kind == "chair":
                self._place_chair(obs, cx, cy, z, envs_idx, device)
            else:
                obs.entities[0].set_position(torch.stack([cx, cy, z], dim=-1),
                                             envs_idx=envs_idx)

    @staticmethod
    def _place_chair(obs: _Obstacle, cx: torch.Tensor, cy: torch.Tensor,
                     z: torch.Tensor, envs_idx: torch.Tensor, device) -> None:
        """Rotate the 4 leg offsets by a random yaw about the chair centre."""
        n = cx.shape[0]
        yaw = _rand(0.0, 2 * math.pi, (n,), device)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        for leg, (dx, dy) in zip(obs.entities, obs.leg_offsets):
            lx = cx + dx * cos_y - dy * sin_y
            ly = cy + dx * sin_y + dy * cos_y
            leg.set_position(torch.stack([lx, ly, z], dim=-1), envs_idx=envs_idx)
