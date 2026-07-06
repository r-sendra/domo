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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from domo.robot import GO2, Robot, RobotSpec, SimulatedLidar
from domo.robot.lidar_models import LidarModelConfig, generic_sector_lidar
from domo.scenes import ObstacleArena, ObstacleArenaConfig
from domo.sim import SimConfig, TerrainConfig, ViewerConfig, create_engine

__all__ = ["WorldConfig", "World"]


@dataclass
class WorldConfig:
    # Simulation
    engine: str = "genesis"
    device: str = "cuda"
    dt: float = 0.02
    substeps: int = 2
    solver_iterations: Optional[int] = 100
    headless: bool = False
    viewer: ViewerConfig = field(default_factory=lambda: ViewerConfig(
        camera_pos=(3.0, -3.0, 2.5), camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=50.0))

    # Environment: "flat" | "rough" | "arena" | "replica"
    scene_kind: str = "flat"
    ground_height: float = 0.0
    arena: ObstacleArenaConfig = field(default_factory=ObstacleArenaConfig)
    rough_terrain: TerrainConfig = field(default_factory=TerrainConfig)
    replica_scene_json: str = ""
    replica_asset_root: str = ""

    # Robot
    kp: float = 100.0
    kd: float = 2.0
    base_init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.35)
    base_init_yaw_deg: float = 0.0

    # Sensors (None → no lidar)
    lidar_model: Optional[LidarModelConfig] = field(
        default_factory=generic_sector_lidar)
    lidar_sectors: int = 36


class World:
    """Engine + scene + robot + sensors, built and bound. Goal-free."""

    def __init__(self, cfg: WorldConfig, n_envs: int = 1,
                 spec: RobotSpec = GO2):
        self.cfg = cfg
        self.n_envs = n_envs
        self.spec = spec

        self.engine = create_engine(cfg.engine, device=cfg.device)
        self.device = self.engine.device
        self.dt = cfg.dt

        self.scene = self.engine.create_scene(SimConfig(
            dt=cfg.dt, substeps=cfg.substeps, device=cfg.device,
            headless=cfg.headless, solver_iterations=cfg.solver_iterations,
            viewer=cfg.viewer))

        # ---- environment ---------------------------------------------------
        self.arena: Optional[ObstacleArena] = None
        if cfg.scene_kind == "flat":
            self.scene.add_ground(cfg.ground_height)
        elif cfg.scene_kind == "rough":
            self.scene.add_terrain(cfg.rough_terrain)
        elif cfg.scene_kind == "arena":
            self.scene.add_ground(cfg.ground_height)
            self.arena = ObstacleArena(self.scene, cfg.arena,
                                       spawn_xy=cfg.base_init_pos[:2])
        elif cfg.scene_kind == "replica":
            from domo.scenes import load_replica_scene
            load_replica_scene(self.scene, cfg.replica_scene_json,
                               cfg.replica_asset_root)
            self.scene.add_ground(cfg.ground_height)
        else:
            raise ValueError(f"unknown scene_kind '{cfg.scene_kind}'")

        # ---- robot + sensors -----------------------------------------------
        yaw = math.radians(cfg.base_init_yaw_deg)
        self.robot = Robot(
            spec, self.scene, self.device, kp=cfg.kp, kd=cfg.kd,
            base_init_pos=cfg.base_init_pos,
            base_init_quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))

        lidar_handle = None
        if cfg.lidar_model is not None:
            lidar_handle = self.scene.add_lidar(
                self.robot.articulation, cfg.lidar_model.to_lidar_config())

        self.scene.build(n_envs)
        self.robot.bind(n_envs)

        self.lidar: Optional[SimulatedLidar] = None
        if lidar_handle is not None:
            self.lidar = SimulatedLidar(
                lidar_handle, cfg.lidar_model, n_envs,
                n_sectors=cfg.lidar_sectors, device=self.device,
                control_dt=cfg.dt)

    # ------------------------------------------------------------------

    @property
    def sensors(self):
        return [self.lidar] if self.lidar is not None else []

    def make_loop(self, controller, command_filter=None):
        """Wire a Controller to this world's standard control loop."""
        from domo.control import SimControlLoop
        return SimControlLoop(self.scene, self.robot, controller,
                              dt=self.dt, sensors=self.sensors,
                              command_filter=command_filter)

    def randomise_obstacles(self, envs_idx=None):
        if self.arena is None:
            return
        import torch
        if envs_idx is None:
            envs_idx = torch.arange(self.n_envs, device=self.device)
        self.arena.randomise(envs_idx, self.device)

    def reset_robot(self):
        """
        Teleport the robot back to its spawn pose (sim-only convenience for
        independent evaluation trials; a deployed robot recovers, it does
        not teleport).
        """
        import torch
        envs = torch.arange(self.n_envs, device=self.device)
        self.robot.reset_idx(envs)
        if self.lidar is not None:
            self.lidar.reset_idx(envs)
        self.robot.refresh()
