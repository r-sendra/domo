"""
Engine-agnostic simulation interfaces.

This module defines the *only* contract the rest of DOMO is allowed to code
against. Concrete physics engines (Genesis; MuJoCo, Isaac, or the real
robot's state-publishing) implement these ABCs in their own
backend module and register themselves in `domo.sim` — nothing outside
`domo/sim/` may import a physics package directly.

Conventions (identical everywhere in DOMO):
  * quaternions are wxyz (scalar-first), see `domo.utils.rotations`
  * all batched quantities are torch tensors of shape [n_envs, ...]
    living on the simulation device
  * world-frame quantities unless a method name says otherwise
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "SimConfig",
    "ViewerConfig",
    "TerrainConfig",
    "LidarConfig",
    "RigidObject",
    "Articulation",
    "LidarSensorHandle",
    "Scene",
    "PhysicsEngine",
]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ViewerConfig:
    camera_pos: Tuple[float, float, float] = (2.0, -2.0, 1.5)
    camera_lookat: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    camera_fov: float = 50.0
    max_fps: Optional[int] = None


@dataclass
class SimConfig:
    dt: float = 0.02
    substeps: int = 2
    device: str = "cuda"          # "cuda" | "cpu" | "mps"
    headless: bool = True
    solver_iterations: Optional[int] = None   # None → engine default
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


@dataclass
class TerrainConfig:
    """Procedural rough-terrain grid (engine maps this to its own morph)."""
    n_subterrains: Tuple[int, int] = (4, 4)
    subterrain_size: Tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.25
    vertical_scale: float = 0.005
    randomize: bool = True
    position: Tuple[float, float, float] = (-16.0, -16.0, 0.0)
    subterrain_types: str = "random_uniform_terrain"


@dataclass
class LidarConfig:
    n_horizontal: int = 36
    n_vertical: int = 5
    fov_deg: Tuple[float, float] = (360.0, 50.0)   # (horizontal, vertical)
    max_range: float = 4.0
    pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.35)
    draw_debug: bool = False


# ---------------------------------------------------------------------------
# Handles returned by a Scene
# ---------------------------------------------------------------------------

class RigidObject(ABC):
    """A (possibly fixed) rigid body: obstacle, prop, furniture piece."""

    @abstractmethod
    def set_position(self, pos: torch.Tensor, envs_idx: Optional[torch.Tensor] = None) -> None:
        """pos: [len(envs_idx), 3] world positions."""


class Articulation(ABC):
    """
    Handle to a robot articulation inside a built scene.

    Query methods return [n_envs, ...] torch tensors on the sim device.
    `dof_idx` arguments are the engine-local DOF indices previously resolved
    via `dof_indices()`.
    """

    # -- structure --------------------------------------------------------
    @abstractmethod
    def dof_indices(self, joint_names: Sequence[str]) -> Sequence[int]: ...

    @abstractmethod
    def link_indices(self, link_names: Sequence[str]) -> Sequence[int]: ...

    # -- state queries -----------------------------------------------------
    @abstractmethod
    def get_base_position(self) -> torch.Tensor: ...

    @abstractmethod
    def get_base_quaternion(self) -> torch.Tensor:
        """wxyz."""

    @abstractmethod
    def get_base_linear_velocity(self) -> torch.Tensor:
        """World frame."""

    @abstractmethod
    def get_base_angular_velocity(self) -> torch.Tensor:
        """World frame."""

    @abstractmethod
    def get_joint_positions(self, dof_idx: Sequence[int]) -> torch.Tensor: ...

    @abstractmethod
    def get_joint_velocities(self, dof_idx: Sequence[int]) -> torch.Tensor: ...

    def get_link_contact_forces(self) -> torch.Tensor:
        """[n_envs, n_links, 3] net contact forces. Optional capability."""
        raise NotImplementedError(f"{type(self).__name__} does not expose contact forces")

    # -- domain randomization (optional capabilities; DrEureka / sim-to-real) --
    def set_friction_ratio(self, ratio: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """ratio: [len(envs_idx)] multiplier applied to all links' friction."""
        raise NotImplementedError(f"{type(self).__name__} does not support friction DR")

    def set_base_mass_shift(self, shift_kg: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """shift_kg: [len(envs_idx)] added payload mass on the base link."""
        raise NotImplementedError(f"{type(self).__name__} does not support mass DR")

    def set_base_com_shift(self, shift_m: torch.Tensor, envs_idx: torch.Tensor) -> None:
        """shift_m: [len(envs_idx), 3] center-of-mass offset on the base link (m)."""
        raise NotImplementedError(f"{type(self).__name__} does not support COM DR")

    def set_pd_gains_scaled(self, kp: torch.Tensor, kd: torch.Tensor,
                            dof_idx: Sequence[int],
                            envs_idx: torch.Tensor) -> None:
        """
        PD gains [n_dofs] applied to the given envs (one draw per reset
        group — Genesis gains are at most per-DOF, broadcast over envs).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support scoped gains")

    # -- actuation ---------------------------------------------------------
    @abstractmethod
    def set_pd_gains(self, kp: Sequence[float], kd: Sequence[float],
                     dof_idx: Sequence[int]) -> None: ...

    @abstractmethod
    def set_joint_position_targets(self, targets: torch.Tensor,
                                   dof_idx: Sequence[int]) -> None: ...

    # -- resets ------------------------------------------------------------
    @abstractmethod
    def set_base_pose(self, pos: torch.Tensor, quat: torch.Tensor,
                      envs_idx: torch.Tensor) -> None: ...

    @abstractmethod
    def set_joint_positions(self, positions: torch.Tensor, dof_idx: Sequence[int],
                            envs_idx: torch.Tensor, zero_velocity: bool = True) -> None: ...

    @abstractmethod
    def zero_all_velocities(self, envs_idx: torch.Tensor) -> None: ...


class LidarSensorHandle(ABC):
    """Raycasting range sensor attached to an articulation base."""

    config: LidarConfig

    @abstractmethod
    def read_sector_distances(self) -> torch.Tensor:
        """
        [n_envs, n_horizontal] distances — minimum over the vertical rays of
        each horizontal sector, clamped to `config.max_range`.
        """

    @abstractmethod
    def read_ranges(self) -> torch.Tensor:
        """
        Raw beam ranges [n_envs, n_vertical, n_horizontal], clamped to
        `config.max_range`. Channel 0 is the lowest-elevation beam.
        """

    def read_points(self):
        """
        World-frame hit points and their ranges for every beam:
        points [n_envs, n_beams, 3], ranges [n_envs, n_beams]
        (n_beams = n_horizontal × n_vertical, flattened). Beams that did not
        hit a surface within max_range are still present — mask them with the
        returned ranges (>= max_range ⇒ no return). Optional: backends that do
        not expose per-beam points raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose per-beam points")


class CameraHandle(ABC):
    """Offscreen RGB camera for visualisation. `render()` costs frame time."""

    @abstractmethod
    def render(self):
        """Return the latest RGB frame as a uint8 array [H, W, 3]."""


# ---------------------------------------------------------------------------
# Scene & engine
# ---------------------------------------------------------------------------

class Scene(ABC):
    """
    A simulation scene. Entities are added first, then `build(n_envs)` is
    called exactly once; afterwards only `step()`, queries, and per-env
    state writes are allowed (mirrors Genesis/Isaac build semantics).
    """

    n_envs: int = 0

    @abstractmethod
    def add_ground(self, height: float = 0.0) -> None: ...

    @abstractmethod
    def add_terrain(self, cfg: TerrainConfig) -> None: ...

    @abstractmethod
    def add_mesh(self, file_path: str, pos: Tuple[float, float, float],
                 quat_wxyz: Tuple[float, float, float, float],
                 fixed: bool = True, scale: float = 1.0) -> RigidObject:
        """Static/prop mesh asset (.glb / .obj)."""

    @abstractmethod
    def add_urdf_prop(self, file_path: str, pos: Tuple[float, float, float],
                      quat_wxyz: Tuple[float, float, float, float],
                      fixed: bool = True) -> RigidObject:
        """Non-robot URDF asset (furniture, appliances, ...)."""

    @abstractmethod
    def add_box(self, size: Tuple[float, float, float],
                pos: Tuple[float, float, float], fixed: bool = True) -> RigidObject: ...

    @abstractmethod
    def add_cylinder(self, radius: float, height: float,
                     pos: Tuple[float, float, float], fixed: bool = True) -> RigidObject: ...

    @abstractmethod
    def add_sphere(self, radius: float,
                   pos: Tuple[float, float, float], fixed: bool = True) -> RigidObject: ...

    @abstractmethod
    def add_articulation(self, urdf_path: str,
                         pos: Tuple[float, float, float],
                         quat_wxyz: Tuple[float, float, float, float]) -> Articulation: ...

    @abstractmethod
    def add_lidar(self, articulation: Articulation, cfg: LidarConfig) -> LidarSensorHandle: ...

    def add_camera(self, res=(320, 240), pos=(3.0, -3.0, 2.0),
                   lookat=(0.0, 0.0, 0.3), fov: float = 50.0) -> "CameraHandle":
        """
        Optional offscreen RGB camera for visualisation (e.g. the dashboard's
        'Genesis window'). Must be called BEFORE build(). Backends that do not
        support it raise NotImplementedError. NOT part of the sim contract —
        rendering it costs frame time, so it is only for twin/eval viz.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no offscreen camera")

    @abstractmethod
    def build(self, n_envs: int) -> None: ...

    @abstractmethod
    def step(self) -> None: ...


class PhysicsEngine(ABC):
    """Entry point of a backend: initialises the engine, creates scenes."""

    name: str = "abstract"

    @abstractmethod
    def create_scene(self, cfg: SimConfig) -> Scene: ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Torch device simulation state tensors live on."""
