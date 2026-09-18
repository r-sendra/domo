"""
Genesis implementation of the DOMO simulation interfaces.

This is the ONLY module in the library that imports `genesis`. Everything it
returns to callers is either a torch tensor or one of the `domo.sim.base`
handle types. Genesis conveniently shares DOMO's wxyz quaternion convention,
so quaternions pass through unconverted.

Genesis 1.0.0 quirks encoded here (keep in sync with the comments below):
  * `gs.init` may only run once per process — guarded by `_ensure_gs_init`.
  * `joint.dof_idx_local` is deprecated; `dofs_idx_local` returns a list
    even for 1-DOF joints, so DOF lists are flattened.
  * `VisOptions(n_rendered_envs=1)` is deprecated → `rendered_envs_idx=[0]`.
  * The bundled go2 URDF merges fixed foot links (`FR_foot` etc. do not
    exist as links) and triggers a benign "qpos0 exceeds joint limits"
    warning. `Robot.bind` tolerates the missing links.
  * `SphericalPattern(n_points=(n_h, n_v))` returns distances as
    [N, n_h, n_v] (azimuth-major); the DOMO contract is [N, n_v, n_h].
  * Per-DOF gains broadcast over envs; `set_pd_gains_scaled` therefore
    applies one draw to a whole reset group.
"""

from __future__ import annotations

from collections.abc import Sequence

import genesis as gs
import torch

from .base import (
    Articulation,
    CameraHandle,
    LidarConfig,
    LidarSensorHandle,
    PhysicsEngine,
    RigidObject,
    Scene,
    SimConfig,
    TerrainConfig,
    lidar_ranges_to_grid,
)

__all__ = [
    "GenesisArticulation",
    "GenesisCamera",
    "GenesisEngine",
    "GenesisLidar",
    "GenesisRigidObject",
    "GenesisScene",
]

# Genesis link index of the articulation root (base) link — DR targets it.
_BASE_LINK_IDX = 0

_GS_INITIALIZED = False


def _ensure_gs_init(device: str) -> torch.device:
    """
    Initialise Genesis once per process; return the torch device to use.

    Genesis raises if `gs.init` is called twice, and "cuda" silently falls
    back to CPU when no GPU is available so the same config runs anywhere.
    """
    global _GS_INITIALIZED
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if not _GS_INITIALIZED:
        gs.init(
            backend=gs.cuda if use_cuda else gs.cpu,
            precision="32",
            logging_level="warning",
            performance_mode=True,
        )
        _GS_INITIALIZED = True
    return torch.device("cuda" if use_cuda else "cpu")


def _default_viewer_fps(dt: float) -> int:
    """Viewer cap when the config leaves it unset: half the control rate."""
    return int(0.5 / dt)


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

class GenesisRigidObject(RigidObject):
    """Wraps a Genesis rigid entity (box/cylinder/sphere/mesh/URDF prop)."""

    def __init__(self, entity):
        self._entity = entity

    def set_position(self, pos: torch.Tensor, envs_idx: torch.Tensor | None = None) -> None:
        self._entity.set_pos(pos, envs_idx=envs_idx)


class GenesisArticulation(Articulation):
    """Wraps a Genesis URDF entity; see `Articulation` for the contract."""

    def __init__(self, entity):
        self._entity = entity

    @property
    def entity(self):
        """Escape hatch for Genesis-specific experimentation. Avoid in library code."""
        return self._entity

    # -- structure --------------------------------------------------------
    def dof_indices(self, joint_names: Sequence[str]) -> Sequence[int]:
        # `dofs_idx_local` returns a list per joint (1 element for hinges);
        # the deprecated scalar `dof_idx_local` must not be used.
        idx: list[int] = []
        for name in joint_names:
            idx.extend(self._entity.get_joint(name).dofs_idx_local)
        return idx

    def link_indices(self, link_names: Sequence[str]) -> Sequence[int]:
        # Raises gs.GenesisException for unknown links (e.g. merged foot links).
        return [self._entity.get_link(n).idx_local for n in link_names]

    # -- state queries -----------------------------------------------------
    def get_base_position(self) -> torch.Tensor:
        return self._entity.get_pos()

    def get_base_quaternion(self) -> torch.Tensor:
        return self._entity.get_quat()          # already wxyz

    def get_base_linear_velocity(self) -> torch.Tensor:
        return self._entity.get_vel()

    def get_base_angular_velocity(self) -> torch.Tensor:
        return self._entity.get_ang()

    def get_joint_positions(self, dof_idx: Sequence[int]) -> torch.Tensor:
        return self._entity.get_dofs_position(dof_idx)

    def get_joint_velocities(self, dof_idx: Sequence[int]) -> torch.Tensor:
        return self._entity.get_dofs_velocity(dof_idx)

    def get_link_contact_forces(self) -> torch.Tensor:
        return self._entity.get_links_net_contact_force()

    # -- domain randomization ----------------------------------------------
    def set_friction_ratio(self, ratio: torch.Tensor, envs_idx: torch.Tensor) -> None:
        # Genesis wants a per-link ratio [n_envs, n_links]; broadcast the
        # per-env scalar over every link.
        n_links = self._entity.n_links
        self._entity.set_friction_ratio(
            ratio.unsqueeze(1).expand(-1, n_links),
            links_idx_local=list(range(n_links)), envs_idx=envs_idx)

    def set_base_mass_shift(self, shift_kg: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self._entity.set_mass_shift(
            shift_kg.unsqueeze(1), links_idx_local=[_BASE_LINK_IDX], envs_idx=envs_idx)

    def set_base_com_shift(self, shift_m: torch.Tensor, envs_idx: torch.Tensor) -> None:
        # Genesis expects [n_envs, n_links, 3]; apply to the base link only.
        self._entity.set_COM_shift(
            shift_m.unsqueeze(1), links_idx_local=[_BASE_LINK_IDX], envs_idx=envs_idx)

    def set_pd_gains_scaled(self, kp: torch.Tensor, kd: torch.Tensor,
                            dof_idx: Sequence[int],
                            envs_idx: torch.Tensor) -> None:
        self._entity.set_dofs_kp(kp, dof_idx, envs_idx=envs_idx)
        self._entity.set_dofs_kv(kd, dof_idx, envs_idx=envs_idx)

    # -- actuation ---------------------------------------------------------
    def set_pd_gains(self, kp: Sequence[float], kd: Sequence[float],
                     dof_idx: Sequence[int]) -> None:
        self._entity.set_dofs_kp(list(kp), dof_idx)
        self._entity.set_dofs_kv(list(kd), dof_idx)

    def set_joint_position_targets(self, targets: torch.Tensor,
                                   dof_idx: Sequence[int]) -> None:
        self._entity.control_dofs_position(targets, dof_idx)

    # -- resets ------------------------------------------------------------
    def set_base_pose(self, pos: torch.Tensor, quat: torch.Tensor,
                      envs_idx: torch.Tensor) -> None:
        # Velocities are zeroed separately by `zero_all_velocities` so that a
        # pose write alone never hides a velocity reset (or the lack of one).
        self._entity.set_pos(pos, zero_velocity=False, envs_idx=envs_idx)
        self._entity.set_quat(quat, zero_velocity=False, envs_idx=envs_idx)

    def set_joint_positions(self, positions: torch.Tensor, dof_idx: Sequence[int],
                            envs_idx: torch.Tensor, zero_velocity: bool = True) -> None:
        self._entity.set_dofs_position(
            position=positions,
            dofs_idx_local=dof_idx,
            zero_velocity=zero_velocity,
            envs_idx=envs_idx,
        )

    def zero_all_velocities(self, envs_idx: torch.Tensor) -> None:
        self._entity.zero_all_dofs_velocity(envs_idx)


class GenesisLidar(LidarSensorHandle):
    """
    Genesis raycast lidar. Genesis lays beams out azimuth-major
    ([N, n_horizontal, n_vertical]); every reader below normalises to the
    contract layout so consumers never see the engine's order.
    """

    def __init__(self, sensor, cfg: LidarConfig, device: torch.device):
        self._sensor = sensor
        self.config = cfg
        self._device = device

    def read_ranges(self) -> torch.Tensor:
        """[N, n_vertical, n_horizontal] ranges clamped to max_range."""
        cfg = self.config
        raw = self._sensor.read().distances
        grid = lidar_ranges_to_grid(raw, cfg.n_vertical, cfg.n_horizontal,
                                    azimuth_major=True)
        return torch.clamp(grid, 0.0, cfg.max_range)

    def read_sector_distances(self) -> torch.Tensor:
        """[N, n_horizontal]: min over the vertical channels of each azimuth."""
        return self.read_ranges().min(dim=1).values

    def read_points(self):
        """
        World-frame hit points [N, n_beams, 3] + ranges [N, n_beams].
        Genesis returns the points in world frame (return_world_frame=True);
        both tensors are flattened in Genesis' native azimuth-major order and
        stay paired with each other.
        """
        data = self._sensor.read()
        pts = data.points                              # [N, n_h, n_v, 3]
        rng = data.distances                           # [N, n_h, n_v]
        n_env = pts.shape[0]
        return (pts.reshape(n_env, -1, 3).to(self._device),
                torch.clamp(rng.reshape(n_env, -1), 0.0,
                            self.config.max_range).to(self._device))


class GenesisCamera(CameraHandle):
    """Offscreen RGB camera; render() returns a uint8 [H, W, 3] frame."""

    def __init__(self, cam):
        self._cam = cam

    def render(self):
        # Genesis returns (rgb, depth, seg, normal) when several outputs are
        # enabled, or the bare rgb array otherwise.
        out = self._cam.render()
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class GenesisScene(Scene):
    """`gs.Scene` wrapper exposing only the `Scene` contract."""

    def __init__(self, cfg: SimConfig, device: torch.device):
        self._cfg = cfg
        self._device = device
        self.n_envs = 0

        rigid_kwargs = dict(
            dt=cfg.dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        )
        if cfg.solver_iterations is not None:
            rigid_kwargs["iterations"] = cfg.solver_iterations

        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=cfg.substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=cfg.viewer.max_fps or _default_viewer_fps(cfg.dt),
                camera_pos=cfg.viewer.camera_pos,
                camera_lookat=cfg.viewer.camera_lookat,
                camera_fov=cfg.viewer.camera_fov,
            ),
            # `n_rendered_envs=1` is deprecated in Genesis 1.0; render env 0 only.
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(**rigid_kwargs),
            show_viewer=not cfg.headless,
        )

    def add_ground(self, height: float = 0.0) -> None:
        self._scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, height)))

    def add_mesh(self, file_path, pos, quat_wxyz, fixed=True, scale=1.0) -> RigidObject:
        return GenesisRigidObject(self._scene.add_entity(gs.morphs.Mesh(
            file=file_path, pos=list(pos), quat=list(quat_wxyz),
            fixed=fixed, scale=(scale, scale, scale))))

    def add_urdf_prop(self, file_path, pos, quat_wxyz, fixed=True) -> RigidObject:
        return GenesisRigidObject(self._scene.add_entity(gs.morphs.URDF(
            file=file_path, pos=list(pos), quat=list(quat_wxyz), fixed=fixed)))

    def add_terrain(self, cfg: TerrainConfig) -> None:
        self._scene.add_entity(gs.morphs.Terrain(
            n_subterrains=cfg.n_subterrains,
            subterrain_size=cfg.subterrain_size,
            horizontal_scale=cfg.horizontal_scale,
            vertical_scale=cfg.vertical_scale,
            randomize=cfg.randomize,
            pos=cfg.position,
            subterrain_types=cfg.subterrain_types,
        ))

    def add_box(self, size, pos, fixed: bool = True) -> RigidObject:
        return GenesisRigidObject(self._scene.add_entity(
            gs.morphs.Box(size=size, pos=pos, fixed=fixed)))

    def add_cylinder(self, radius, height, pos, fixed: bool = True) -> RigidObject:
        return GenesisRigidObject(self._scene.add_entity(
            gs.morphs.Cylinder(radius=radius, height=height, pos=pos, fixed=fixed)))

    def add_sphere(self, radius, pos, fixed: bool = True) -> RigidObject:
        return GenesisRigidObject(self._scene.add_entity(
            gs.morphs.Sphere(radius=radius, pos=pos, fixed=fixed)))

    def add_articulation(self, urdf_path, pos, quat_wxyz) -> Articulation:
        entity = self._scene.add_entity(gs.morphs.URDF(
            file=urdf_path, pos=list(pos), quat=list(quat_wxyz)))
        return GenesisArticulation(entity)

    def add_lidar(self, articulation: Articulation, cfg: LidarConfig) -> LidarSensorHandle:
        if not isinstance(articulation, GenesisArticulation):
            raise TypeError(
                f"GenesisScene.add_lidar needs a GenesisArticulation, "
                f"got {type(articulation).__name__}")
        sensor = self._scene.add_sensor(gs.sensors.Lidar(
            pattern=gs.sensors.SphericalPattern(
                fov=cfg.fov_deg,
                n_points=(cfg.n_horizontal, cfg.n_vertical),   # (azimuth, elevation)
            ),
            entity_idx=articulation.entity.idx,
            pos_offset=cfg.pos_offset,
            return_world_frame=True,
            draw_debug=cfg.draw_debug,
        ))
        return GenesisLidar(sensor, cfg, self._device)

    def add_camera(self, res=(320, 240), pos=(3.0, -3.0, 2.0),
                   lookat=(0.0, 0.0, 0.3), fov: float = 50.0) -> CameraHandle:
        cam = self._scene.add_camera(res=tuple(res), pos=tuple(pos),
                                     lookat=tuple(lookat), fov=fov, GUI=False)
        return GenesisCamera(cam)

    def build(self, n_envs: int) -> None:
        self._scene.build(n_envs=n_envs)
        self.n_envs = n_envs

    def step(self) -> None:
        self._scene.step()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GenesisEngine(PhysicsEngine):
    """Genesis backend; created through `domo.sim.create_engine("genesis")`."""

    name = "genesis"

    def __init__(self, device: str = "cuda"):
        self._device = _ensure_gs_init(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def create_scene(self, cfg: SimConfig) -> Scene:
        return GenesisScene(cfg, self._device)
