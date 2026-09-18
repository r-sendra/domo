"""
World: the digital-twin runtime — the top of the DOMO stack.

A World is a robot spawned in an environment with its sensors. Nothing
more: no goal, no rewards, no episodes. This is the resident entry point
of the system — the twin (and later the real robot) simply *exists* here,
running whatever Controller currently governs it:

    world = World(WorldConfig(scene_kind="arena"))
    controller = MyPlanner(library)          # LLM / HRL / researcher code
    controller.setup(world.robot)
    loop = world.make_loop(controller)
    loop.reset()
    loop.run(...)

RL enters the picture the other way around: when the supervisor decides a
new skill must be LEARNED (M1 → M3), it instantiates a vectorised training
task (domo.tasks) — a temporary, reward-bearing lens over the same scene
builders — trains, registers the new skill in the library, and control
returns to the World. Tasks are subordinate procedures, not the entry
point.

Construction order matters and is fixed here: engine → scene → environment
entities → robot articulation → lidar/camera (must precede build) →
`scene.build(n_envs)` → `robot.bind(n_envs)` → device-model sensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from domo.robot import GO2, Robot, RobotSpec, SimulatedLidar
from domo.robot.lidar_models import LidarModelConfig, generic_sector_lidar
from domo.scenes import ObstacleArena, ObstacleArenaConfig
from domo.sim import (
    CameraHandle,
    LidarSensorHandle,
    Scene,
    SimConfig,
    TerrainConfig,
    ViewerConfig,
    create_engine,
)

__all__ = ["World", "WorldConfig"]

SCENE_KINDS = ("flat", "rough", "arena", "replica")


@dataclass
class WorldConfig:
    """
    Everything needed to stand up a twin. Defaults are the twin/eval
    settings (viewer on, stiff 100/2 gains); training tasks build their own
    WorldConfig from their task config.
    """
    # Simulation
    engine: str = "genesis"
    device: str = "cuda"
    dt: float = 0.02
    substeps: int = 2
    solver_iterations: int | None = 100
    headless: bool = False
    viewer: ViewerConfig = field(default_factory=lambda: ViewerConfig(
        camera_pos=(3.0, -3.0, 2.5), camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=50.0))

    # Environment: one of SCENE_KINDS ("flat" | "rough" | "arena" | "replica")
    scene_kind: str = "flat"
    ground_height: float = 0.0
    arena: ObstacleArenaConfig = field(default_factory=ObstacleArenaConfig)
    rough_terrain: TerrainConfig = field(default_factory=TerrainConfig)
    replica_scene_json: str = ""
    replica_asset_root: str = ""

    # Robot
    kp: float = 100.0
    kd: float = 2.0
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)
    base_init_yaw_deg: float = 0.0

    # Sensors (None → no lidar)
    lidar_model: LidarModelConfig | None = field(
        default_factory=generic_sector_lidar)
    lidar_sectors: int = 36

    # Optional offscreen dashboard camera (None → none). (w, h) resolution.
    # Rendering it costs frame time — twin/eval viz only, never training.
    camera_res: tuple[int, int] | None = None
    camera_pos: tuple[float, float, float] = (4.0, -4.0, 3.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.3)


def _yaw_quat_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    """wxyz quaternion for a pure rotation about z."""
    return (math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))


class World:
    """
    Engine + scene + robot + sensors, built and bound. Goal-free.

    Args:
        cfg: world configuration.
        n_envs: parallel copies (1 for the twin; tasks use thousands).
        spec: robot description (Go2 by default).

    Attributes:
        engine, scene, device, dt: the sim layer objects.
        robot: bound `Robot`; `robot.state` is refreshed by the control loop.
        arena: the `ObstacleArena` when scene_kind == "arena", else None.
        lidar: `SimulatedLidar` when a lidar model is configured, else None.
        camera: offscreen `CameraHandle` when `camera_res` is set, else None.
    """

    def __init__(self, cfg: WorldConfig, n_envs: int = 1,
                 spec: RobotSpec = GO2):
        self.cfg = cfg
        self.n_envs = n_envs
        self.spec = spec

        self.engine = create_engine(cfg.engine, device=cfg.device)
        self.device = self.engine.device
        self.dt = cfg.dt

        self.scene: Scene = self.engine.create_scene(SimConfig(
            dt=cfg.dt, substeps=cfg.substeps, device=cfg.device,
            headless=cfg.headless, solver_iterations=cfg.solver_iterations,
            viewer=cfg.viewer))

        self.arena: ObstacleArena | None = self._build_environment(cfg)

        # ---- robot + sensors (all added before build) -----------------------
        self.robot = Robot(
            spec, self.scene, self.device, kp=cfg.kp, kd=cfg.kd,
            base_init_pos=cfg.base_init_pos,
            base_init_quat=_yaw_quat_wxyz(math.radians(cfg.base_init_yaw_deg)))

        lidar_handle: LidarSensorHandle | None = None
        if cfg.lidar_model is not None:
            lidar_handle = self.scene.add_lidar(
                self.robot.articulation, cfg.lidar_model.to_lidar_config())

        self.camera: CameraHandle | None = None
        if cfg.camera_res is not None:
            self.camera = self.scene.add_camera(
                res=cfg.camera_res, pos=cfg.camera_pos,
                lookat=cfg.camera_lookat, fov=cfg.viewer.camera_fov)

        self.scene.build(n_envs)
        self.robot.bind(n_envs)

        self.lidar: SimulatedLidar | None = None
        if lidar_handle is not None:
            self.lidar = SimulatedLidar(
                lidar_handle, cfg.lidar_model, n_envs,
                n_sectors=cfg.lidar_sectors, device=self.device,
                control_dt=cfg.dt)

    def _build_environment(self, cfg: WorldConfig) -> ObstacleArena | None:
        """Add ground/terrain/props for `cfg.scene_kind`; returns the arena if any."""
        if cfg.scene_kind == "flat":
            self.scene.add_ground(cfg.ground_height)
        elif cfg.scene_kind == "rough":
            self.scene.add_terrain(cfg.rough_terrain)
        elif cfg.scene_kind == "arena":
            self.scene.add_ground(cfg.ground_height)
            return ObstacleArena(self.scene, cfg.arena,
                                 spawn_xy=cfg.base_init_pos[:2])
        elif cfg.scene_kind == "replica":
            from domo.scenes import load_replica_scene
            load_replica_scene(self.scene, cfg.replica_scene_json,
                               cfg.replica_asset_root)
            self.scene.add_ground(cfg.ground_height)
        else:
            raise ValueError(
                f"unknown scene_kind '{cfg.scene_kind}' (expected one of {SCENE_KINDS})")
        return None

    # ------------------------------------------------------------------

    @property
    def sensors(self) -> list[SimulatedLidar]:
        """Exteroceptive sensors the control loop must tick each step."""
        return [self.lidar] if self.lidar is not None else []

    def make_loop(self, controller, command_filter=None):
        """Wire a Controller to this world's standard control loop."""
        from domo.control import SimControlLoop
        return SimControlLoop(self.scene, self.robot, controller,
                              dt=self.dt, sensors=self.sensors,
                              command_filter=command_filter)

    def _all_envs(self) -> torch.Tensor:
        return torch.arange(self.n_envs, device=self.device)

    def randomise_obstacles(self, envs_idx: torch.Tensor | None = None) -> None:
        """Re-scatter the arena obstacles (no-op for other scene kinds); None → all envs."""
        if self.arena is None:
            return
        if envs_idx is None:
            envs_idx = self._all_envs()
        self.arena.randomise(envs_idx, self.device)

    def reset_robot(self) -> None:
        """
        Teleport the robot back to its spawn pose (sim-only convenience for
        independent evaluation trials; a deployed robot recovers, it does
        not teleport).
        """
        envs = self._all_envs()
        self.robot.reset_idx(envs)
        if self.lidar is not None:
            self.lidar.reset_idx(envs)
        self.robot.refresh()
