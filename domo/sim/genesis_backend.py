"""
Genesis implementation of the DOMO simulation interfaces.

This is the ONLY module in the library that imports `genesis`. Everything it
returns to callers is either a torch tensor or one of the `domo.sim.base`
handle types. Genesis conveniently shares DOMO's wxyz quaternion convention,
so quaternions pass through unconverted.
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
)

_GS_INITIALIZED = False


def _ensure_gs_init(device: str) -> torch.device:
    """Initialise Genesis once per process; return the torch device to use."""
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


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

class GenesisRigidObject(RigidObject):
    def __init__(self, entity):
        self._entity = entity

    def set_position(self, pos: torch.Tensor, envs_idx: torch.Tensor | None = None) -> None:
        self._entity.set_pos(pos, envs_idx=envs_idx)


class GenesisArticulation(Articulation):
    def __init__(self, entity):
        self._entity = entity

    @property
    def entity(self):
        """Escape hatch for Genesis-specific experimentation. Avoid in library code."""
        return self._entity

    # -- structure --------------------------------------------------------
    def dof_indices(self, joint_names: Sequence[str]) -> Sequence[int]:
        # dofs_idx_local returns a list per joint (1 element for hinges).
        idx = []
        for name in joint_names:
            idx.extend(self._entity.get_joint(name).dofs_idx_local)
        return idx

    def link_indices(self, link_names: Sequence[str]) -> Sequence[int]:
        return [self._entity.get_link(n).idx_local for n in link_names]

    # -- state queries -----------------------------------------------------
    def get_base_position(self) -> torch.Tensor:
        return self._entity.get_pos()

    def get_base_quaternion(self) -> torch.Tensor:
        return self._entity.get_quat()

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
        n_links = self._entity.n_links
        self._entity.set_friction_ratio(
            ratio.unsqueeze(1).expand(-1, n_links),
            links_idx_local=list(range(n_links)), envs_idx=envs_idx)

    def set_base_mass_shift(self, shift_kg: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self._entity.set_mass_shift(
            shift_kg.unsqueeze(1), links_idx_local=[0], envs_idx=envs_idx)

    def set_base_com_shift(self, shift_m: torch.Tensor, envs_idx: torch.Tensor) -> None:
        # Genesis expects [n_envs, n_links, 3]; apply to the base link only.
        self._entity.set_COM_shift(
            shift_m.unsqueeze(1), links_idx_local=[0], envs_idx=envs_idx)

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
    def __init__(self, sensor, cfg: LidarConfig, device: torch.device):
        self._sensor = sensor
        self.config = cfg
        self._device = device

    def read_sector_distances(self) -> torch.Tensor:
        """
        Genesis returns distances shaped [N, n_vert, n_horiz] (or flattened,
        depending on version/pattern); normalise to [N, n_horizontal] taking
        the min over vertical rays per sector.
        """
        cfg = self.config
        raw = self._sensor.read().distances
        if raw.dim() == 3:
            n_env, _, n_horiz = raw.shape
            if n_horiz == cfg.n_horizontal:
                sectors = raw.min(dim=1).values
            else:
                flat = raw.reshape(n_env, -1)
                rays_per_sector = flat.shape[1] // cfg.n_horizontal
                n_used = rays_per_sector * cfg.n_horizontal
                sectors = flat[:, :n_used].view(
                    n_env, cfg.n_horizontal, rays_per_sector).min(dim=2).values
        else:
            flat = raw.reshape(raw.shape[0], -1)
            rays_per_sector = max(flat.shape[1] // cfg.n_horizontal, 1)
            n_used = rays_per_sector * cfg.n_horizontal
            sectors = flat[:, :n_used].view(
                flat.shape[0], cfg.n_horizontal, rays_per_sector).min(dim=2).values
        return torch.clamp(sectors, 0.0, cfg.max_range)

    def read_ranges(self) -> torch.Tensor:
        cfg = self.config
        raw = self._sensor.read().distances
        n_env = raw.shape[0]
        if raw.dim() == 3 and raw.shape[1] == cfg.n_vertical \
                and raw.shape[2] == cfg.n_horizontal:
            ranges = raw
        else:
            flat = raw.reshape(n_env, -1)
            expected = cfg.n_vertical * cfg.n_horizontal
            if flat.shape[1] != expected:
                raise RuntimeError(
                    f"Lidar returned {flat.shape[1]} rays/env, expected "
                    f"{cfg.n_vertical}x{cfg.n_horizontal}={expected}")
            ranges = flat.view(n_env, cfg.n_vertical, cfg.n_horizontal)
        return torch.clamp(ranges, 0.0, cfg.max_range)

    def read_points(self):
        """World-frame hit points [N, n_beams, 3] + ranges [N, n_beams].
        Genesis returns the sensor in world frame (return_world_frame=True);
        beam (horizontal/vertical) layout is irrelevant once flattened."""
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
        out = self._cam.render()             # (rgb, depth, seg, normal)
        rgb = out[0] if isinstance(out, tuple) else out
        return rgb


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class GenesisScene(Scene):
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
                max_FPS=cfg.viewer.max_fps or int(0.5 / cfg.dt),
                camera_pos=cfg.viewer.camera_pos,
                camera_lookat=cfg.viewer.camera_lookat,
                camera_fov=cfg.viewer.camera_fov,
            ),
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
        assert isinstance(articulation, GenesisArticulation)
        sensor = self._scene.add_sensor(gs.sensors.Lidar(
            pattern=gs.sensors.SphericalPattern(
                fov=cfg.fov_deg,
                n_points=(cfg.n_horizontal, cfg.n_vertical),
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
    name = "genesis"

    def __init__(self, device: str = "cuda"):
        self._device = _ensure_gs_init(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def create_scene(self, cfg: SimConfig) -> Scene:
        return GenesisScene(cfg, self._device)
