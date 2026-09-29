"""
Engine-agnostic simulation interfaces.

This module defines the *only* contract the rest of DOMO is allowed to code
against. Concrete physics engines (Genesis; MuJoCo, Isaac, or the real
robot's state-publishing) implement these ABCs in their own backend module
and register themselves in `domo.sim` — nothing outside `domo/sim/` may
import a physics package directly.

Conventions (identical everywhere in DOMO):
  * quaternions are wxyz (scalar-first), see `domo.utils.rotations`
  * all batched quantities are torch tensors of shape [n_envs, ...]
    living on the simulation device
  * world-frame quantities unless a method name says otherwise
  * SI units: metres, seconds, radians (degrees only where a field name
    says `_deg`), kilograms

Lifecycle: entities and sensors are added to a `Scene`, then `build(n_envs)`
is called exactly once; afterwards only `step()`, queries and per-env state
writes are allowed. `Robot.bind()` and the sensors are the layer above this.

Optional capabilities (contact forces, domain randomization, per-beam
lidar points, offscreen camera, viewer ground picking) have non-abstract
defaults that raise `NotImplementedError`; callers probe them and degrade
gracefully so the same code runs on every backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

__all__ = [
    "Articulation",
    "CameraHandle",
    "LidarConfig",
    "LidarSensorHandle",
    "PhysicsEngine",
    "RigidObject",
    "Scene",
    "SimConfig",
    "TerrainConfig",
    "ViewerConfig",
    "lidar_ranges_to_grid",
]


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ViewerConfig:
    """Interactive viewer camera (only used when `SimConfig.headless` is False)."""
    camera_pos: tuple[float, float, float] = (2.0, -2.0, 1.5)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.3)
    camera_fov: float = 50.0          # degrees
    max_fps: int | None = None        # None → backend default (derived from dt)


@dataclass
class SimConfig:
    """Physics-step configuration handed to `PhysicsEngine.create_scene`."""
    dt: float = 0.02                  # control/physics step (s); substeps subdivide it
    substeps: int = 2
    device: str = "cuda"              # "cuda" | "cpu" | "mps"
    headless: bool = True
    solver_iterations: int | None = None   # None → engine default
    viewer: ViewerConfig = field(default_factory=ViewerConfig)


@dataclass
class TerrainConfig:
    """Procedural rough-terrain grid (engine maps this to its own morph)."""
    n_subterrains: tuple[int, int] = (4, 4)
    subterrain_size: tuple[float, float] = (8.0, 8.0)      # metres per sub-terrain
    horizontal_scale: float = 0.25                          # heightfield cell size (m)
    vertical_scale: float = 0.005                           # heightfield unit (m)
    randomize: bool = True
    position: tuple[float, float, float] = (-16.0, -16.0, 0.0)   # grid corner
    subterrain_types: str = "random_uniform_terrain"


@dataclass
class LidarConfig:
    """
    Raycast lidar geometry as seen by the backend (the device model that
    adds noise/dropout lives in `domo.robot.lidar_models`).
    """
    n_horizontal: int = 36            # azimuth samples over fov_deg[0]
    n_vertical: int = 5               # elevation channels over fov_deg[1]
    fov_deg: tuple[float, float] = (360.0, 50.0)   # (horizontal, vertical)
    max_range: float = 4.0            # metres; no-returns are clamped to this
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)   # from base link (m)
    draw_debug: bool = False


# ---------------------------------------------------------------------------
# Handles returned by a Scene
# ---------------------------------------------------------------------------

class RigidObject(ABC):
    """A (possibly fixed) rigid body: obstacle, prop, furniture piece."""

    @abstractmethod
    def set_position(self, pos: torch.Tensor, envs_idx: torch.Tensor | None = None) -> None:
        """
        Teleport the body.

        Args:
            pos: [len(envs_idx), 3] world positions.
            envs_idx: env indices to write; None → all envs.
        """


class Articulation(ABC):
    """
    Handle to a robot articulation inside a built scene.

    Query methods return [n_envs, ...] torch tensors on the sim device.
    `dof_idx` arguments are the engine-local DOF indices previously resolved
    via `dof_indices()`; DOMO passes them in the spec's canonical joint order
    so every returned joint tensor follows that order too.
    """

    # -- structure --------------------------------------------------------
    @abstractmethod
    def dof_indices(self, joint_names: Sequence[str]) -> Sequence[int]:
        """Engine-local DOF indices for the named joints, in the given order."""

    @abstractmethod
    def link_indices(self, link_names: Sequence[str]) -> Sequence[int]:
        """
        Engine-local link indices. May raise an engine-specific error when a
        link does not exist (URDF importers often merge fixed links).
        """

    # -- state queries -----------------------------------------------------
    @abstractmethod
    def get_base_position(self) -> torch.Tensor:
        """[N, 3] world position of the base link."""

    @abstractmethod
    def get_base_quaternion(self) -> torch.Tensor:
        """[N, 4] wxyz orientation of the base link."""

    @abstractmethod
    def get_base_linear_velocity(self) -> torch.Tensor:
        """[N, 3] world-frame linear velocity."""

    @abstractmethod
    def get_base_angular_velocity(self) -> torch.Tensor:
        """[N, 3] world-frame angular velocity."""

    @abstractmethod
    def get_joint_positions(self, dof_idx: Sequence[int]) -> torch.Tensor:
        """[N, len(dof_idx)] joint angles (rad)."""

    @abstractmethod
    def get_joint_velocities(self, dof_idx: Sequence[int]) -> torch.Tensor:
        """[N, len(dof_idx)] joint velocities (rad/s)."""

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
                     dof_idx: Sequence[int]) -> None:
        """Set per-DOF PD gains (shared by all envs)."""

    @abstractmethod
    def set_joint_position_targets(self, targets: torch.Tensor,
                                   dof_idx: Sequence[int]) -> None:
        """targets: [N, len(dof_idx)] desired joint angles for the engine PD loop."""

    # -- resets ------------------------------------------------------------
    @abstractmethod
    def set_base_pose(self, pos: torch.Tensor, quat: torch.Tensor,
                      envs_idx: torch.Tensor) -> None:
        """pos [len(envs_idx), 3], quat [len(envs_idx), 4] wxyz. Does not zero velocities."""

    @abstractmethod
    def set_joint_positions(self, positions: torch.Tensor, dof_idx: Sequence[int],
                            envs_idx: torch.Tensor, zero_velocity: bool = True) -> None:
        """positions: [len(envs_idx), len(dof_idx)] joint angles to write."""

    @abstractmethod
    def zero_all_velocities(self, envs_idx: torch.Tensor) -> None:
        """Zero base and joint velocities of the given envs."""


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
        `config.max_range`. Channel 0 is the lowest-elevation beam; azimuth
        index 0 is the start of the horizontal FOV. Backends whose engine
        returns another layout must normalise it (see `lidar_ranges_to_grid`).
        """

    def read_points(self):
        """
        World-frame hit points and their ranges for every beam:
        points [n_envs, n_beams, 3], ranges [n_envs, n_beams]
        (n_beams = n_horizontal × n_vertical, flattened in the backend's
        native beam order — pair each point with its own range, never with
        `read_ranges()`). Beams that did not hit a surface within max_range
        are still present — mask them with the returned ranges
        (>= max_range ⇒ no return). Optional: backends that do not expose
        per-beam points raise NotImplementedError.
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

    Positions are world-frame metres, orientations wxyz quaternions.
    """

    n_envs: int = 0

    @abstractmethod
    def add_ground(self, height: float = 0.0) -> None:
        """Infinite ground plane at z = height."""

    @abstractmethod
    def add_terrain(self, cfg: TerrainConfig) -> None:
        """Procedural rough-terrain heightfield."""

    @abstractmethod
    def add_mesh(self, file_path: str, pos: tuple[float, float, float],
                 quat_wxyz: tuple[float, float, float, float],
                 fixed: bool = True, scale: float = 1.0) -> RigidObject:
        """Static/prop mesh asset (.glb / .obj)."""

    @abstractmethod
    def add_urdf_prop(self, file_path: str, pos: tuple[float, float, float],
                      quat_wxyz: tuple[float, float, float, float],
                      fixed: bool = True) -> RigidObject:
        """Non-robot URDF asset (furniture, appliances, ...)."""

    @abstractmethod
    def add_box(self, size: tuple[float, float, float],
                pos: tuple[float, float, float], fixed: bool = True) -> RigidObject:
        """Axis-aligned box; `size` is the full extent (m), `pos` its centre."""

    @abstractmethod
    def add_cylinder(self, radius: float, height: float,
                     pos: tuple[float, float, float], fixed: bool = True) -> RigidObject:
        """Vertical cylinder; `pos` is its centre."""

    @abstractmethod
    def add_sphere(self, radius: float,
                   pos: tuple[float, float, float], fixed: bool = True) -> RigidObject: ...

    @abstractmethod
    def add_articulation(self, urdf_path: str,
                         pos: tuple[float, float, float],
                         quat_wxyz: tuple[float, float, float, float]) -> Articulation:
        """Robot URDF; the returned handle is bound by `domo.robot.Robot`."""

    @abstractmethod
    def add_lidar(self, articulation: Articulation, cfg: LidarConfig) -> LidarSensorHandle:
        """Raycast lidar rigidly attached to the articulation's base link."""

    def add_camera(self, res=(320, 240), pos=(3.0, -3.0, 2.0),
                   lookat=(0.0, 0.0, 0.3), fov: float = 50.0) -> CameraHandle:
        """
        Optional offscreen RGB camera for visualisation (e.g. the dashboard's
        'Genesis window'). Must be called BEFORE build(). Backends that do not
        support it raise NotImplementedError. NOT part of the sim contract —
        rendering it costs frame time, so it is only for twin/eval viz.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no offscreen camera")

    def on_ground_click(self, callback, ground_z: float = 0.0) -> None:
        """
        Optional: call `callback(x, y)` when the user left-clicks the ground
        plane in the engine's INTERACTIVE viewer. Backends without a viewer
        picking API raise NotImplementedError.

        Viewer-only, so it has no meaning headless — a headless backend should
        refuse rather than silently never fire (nothing would ever click).

        `x, y` are world-frame metres of the point where the click ray meets
        the horizontal plane `z = ground_z`; clicks that miss that plane (sky,
        a camera looking up) are dropped. The callback runs on the viewer/UI
        thread, NOT the sim thread: keep it to a cheap, thread-safe store —
        stash the goal and let the control loop pick it up — because anything
        slow stalls rendering and anything unsynchronised races the sim.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no viewer ground picking")

    @abstractmethod
    def build(self, n_envs: int) -> None:
        """Compile the scene for `n_envs` parallel copies. Exactly once."""

    @abstractmethod
    def step(self) -> None:
        """Advance physics by `SimConfig.dt`."""


class PhysicsEngine(ABC):
    """Entry point of a backend: initialises the engine, creates scenes."""

    name: str = "abstract"

    @abstractmethod
    def create_scene(self, cfg: SimConfig) -> Scene: ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Torch device simulation state tensors live on."""


# ---------------------------------------------------------------------------
# Helpers for backend implementers (pure torch, engine-free)
# ---------------------------------------------------------------------------

def lidar_ranges_to_grid(raw: torch.Tensor, n_vertical: int, n_horizontal: int,
                         azimuth_major: bool = True) -> torch.Tensor:
    """
    Normalise an engine's raw lidar distance buffer to the contract layout
    [N, n_vertical, n_horizontal].

    Raycasters differ in how they lay beams out: Genesis' SphericalPattern
    returns [N, n_horizontal, n_vertical] (azimuth-major, one row per
    azimuth), other engines may return elevation-major grids or a flat
    [N, n_beams] buffer. This helper handles all three:

      * 3-D [N, n_horizontal, n_vertical]  → transposed
      * 3-D [N, n_vertical, n_horizontal]  → returned as is
      * 2-D [N, n_beams]                   → viewed with the layout given by
        `azimuth_major` (True: beams of one azimuth are contiguous)

    A square grid (n_vertical == n_horizontal) is ambiguous by shape alone,
    so it is interpreted according to `azimuth_major` as well.

    Args:
        raw: engine distances, [N, a, b] or [N, a*b].
        n_vertical / n_horizontal: expected beam counts.
        azimuth_major: the engine's native order for flat / square buffers.

    Raises:
        RuntimeError: if the beam count does not match n_vertical × n_horizontal.
    """
    n_env = raw.shape[0]
    n_beams = n_vertical * n_horizontal
    if raw.dim() == 3 and n_vertical != n_horizontal:
        if tuple(raw.shape[1:]) == (n_horizontal, n_vertical):
            return raw.transpose(1, 2)
        if tuple(raw.shape[1:]) == (n_vertical, n_horizontal):
            return raw
    flat = raw.reshape(n_env, -1)
    if flat.shape[1] != n_beams:
        raise RuntimeError(
            f"Lidar returned {flat.shape[1]} rays/env, expected "
            f"{n_vertical}x{n_horizontal}={n_beams}")
    if azimuth_major:
        return flat.view(n_env, n_horizontal, n_vertical).transpose(1, 2)
    return flat.view(n_env, n_vertical, n_horizontal)
