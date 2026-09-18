"""
Go2 velocity-tracking locomotion task (flat or rough terrain).

Functional port of the original `domo/robot/go2.py::Go2WalkEnv` (which
followed the official Genesis locomotion example) onto the layered
architecture: physics access goes through `domo.sim` handles wrapped in the
robot's sensors/actuators; all math is engine-agnostic.

Observation (45):
  [0:3]   base angular velocity (body frame, ×0.25)
  [3:6]   projected gravity (body frame)
  [6:9]   velocity command (scaled)
  [9:21]  joint pos − default (×1.0)
  [21:33] joint velocities (×0.05)
  [33:45] previous action

Action (12): joint-position offsets from default stance × action_scale,
executed with 1 step of latency (matches real Go2 command pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from domo.robot import GO2, Robot, RobotSpec
from domo.sim import SimConfig, TerrainConfig, ViewerConfig, create_engine

from .base import VecTask, rand_uniform

__all__ = ["Go2WalkConfig", "Go2WalkTask"]


@dataclass
class Go2WalkConfig:
    # Vectorisation / timing
    n_envs: int = 4096
    dt: float = 0.02
    max_episode_steps: int = 1000
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"
    terrain: str = "flat"                    # "flat" | "rough"

    # Control
    action_scale: float = 0.25
    clip_actions: float = 100.0
    kp: float = 20.0
    kd: float = 0.5
    simulate_action_latency: bool = True     # real Go2 has ~1 step delay

    # Commands
    resampling_time_s: float = 4.0
    lin_vel_x_range: tuple[float, float] = (0.5, 0.5)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_range: tuple[float, float] = (0.0, 0.0)

    # Termination
    termination_pitch: float = 1.0           # [rad]
    termination_roll: float = 1.0            # [rad]
    termination_height: float = 0.20         # [m], flat terrain only

    # Rewards
    tracking_sigma: float = 0.25
    base_height_target: float = 0.34
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -1.0,
        "base_height": -50.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    })

    # Observation scales
    obs_scales: dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05,
    })

    # Rough terrain
    rough_terrain: TerrainConfig = field(default_factory=TerrainConfig)
    rough_spawn_height: float = 0.5


class Go2WalkTask(VecTask):

    OBS_DIM = 45
    ACT_DIM = 12

    def __init__(self, cfg: Go2WalkConfig, spec: RobotSpec = GO2):
        # ---- engine & scene -------------------------------------------------
        engine = create_engine(cfg.engine, device=cfg.device)
        device = engine.device
        super().__init__(cfg.n_envs, self.OBS_DIM, self.ACT_DIM, device,
                         cfg.dt, max_episode_length=cfg.max_episode_steps)
        self.cfg = cfg
        self.episode_length_s = cfg.max_episode_steps * cfg.dt

        self.scene = engine.create_scene(SimConfig(
            dt=cfg.dt,
            substeps=2,
            device=cfg.device,
            headless=cfg.headless,
            viewer=ViewerConfig(camera_pos=(2.0, 0.0, 2.5),
                                camera_lookat=(0.0, 0.0, 0.5),
                                camera_fov=40.0),
        ))

        if cfg.terrain == "flat":
            self.scene.add_ground()
            self.spawn_positions = None
        else:
            self.scene.add_terrain(cfg.rough_terrain)
            self.spawn_positions = self._terrain_spawn_grid(cfg)

        self.robot = Robot(spec, self.scene, device, kp=cfg.kp, kd=cfg.kd)
        self.scene.build(cfg.n_envs)
        self.robot.bind(cfg.n_envs)

        # ---- rewards ---------------------------------------------------------
        self.register_rewards(cfg.reward_scales)

        # ---- buffers ---------------------------------------------------------
        N, f = cfg.n_envs, torch.float32
        self.commands = torch.zeros((N, 3), device=device, dtype=f)
        self.commands_scale = torch.tensor(
            [cfg.obs_scales["lin_vel"], cfg.obs_scales["lin_vel"],
             cfg.obs_scales["ang_vel"]], device=device, dtype=f)
        self.actions = torch.zeros((N, self.ACT_DIM), device=device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)

    # ------------------------------------------------------------------

    def _terrain_spawn_grid(self, cfg: Go2WalkConfig) -> torch.Tensor:
        """One spawn point at the centre of each subterrain tile, cycled."""
        t = cfg.rough_terrain
        n_cols, n_rows = t.n_subterrains
        tile_w, tile_h = t.subterrain_size
        ox, oy, _ = t.position
        offsets = []
        for i in range(cfg.n_envs):
            row = (i // n_cols) % n_rows
            col = i % n_cols
            offsets.append([ox + (col + 0.5) * tile_w,
                            oy + (row + 0.5) * tile_h,
                            cfg.rough_spawn_height])
        return torch.tensor(offsets, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        state = self.robot.state

        self.actions = torch.clip(actions, -cfg.clip_actions, cfg.clip_actions)
        exec_actions = (self.last_actions if cfg.simulate_action_latency
                        else self.actions)
        target_dof_pos = (exec_actions * cfg.action_scale
                          + self.robot.default_dof_pos)
        self.robot.set_joint_targets(target_dof_pos)
        self.scene.step()

        self.episode_length_buf += 1
        self.robot.refresh()

        # Resample commands periodically
        resample_every = int(cfg.resampling_time_s / cfg.dt)
        envs_idx = ((self.episode_length_buf % resample_every == 0)
                    .nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        # Termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(state.base_euler[:, 1]) > cfg.termination_pitch
        self.reset_buf |= torch.abs(state.base_euler[:, 0]) > cfg.termination_roll
        if cfg.terrain == "flat":
            self.reset_buf |= state.base_pos[:, 2] < cfg.termination_height

        time_out_idx = ((self.episode_length_buf > self.max_episode_length)
                        .nonzero(as_tuple=False).flatten())
        self.extras["time_outs"] = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.float32)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.compute_rewards()

        self.obs_buf = torch.cat([
            state.base_ang_vel * cfg.obs_scales["ang_vel"],                    # 3
            state.projected_gravity,                                           # 3
            self.commands * self.commands_scale,                               # 3
            (state.dof_pos - self.robot.default_dof_pos)
            * cfg.obs_scales["dof_pos"],                                       # 12
            state.dof_vel * cfg.obs_scales["dof_vel"],                         # 12
            self.actions,                                                      # 12
        ], dim=-1)

        self.last_actions[:] = self.actions

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        return self.obs_buf, None

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        spawn = (self.spawn_positions[envs_idx]
                 if self.spawn_positions is not None else None)
        self.robot.reset_idx(envs_idx, base_pos=spawn)

        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.log_episode_sums(envs_idx, self.episode_length_s)
        self._resample_commands(envs_idx)

    def _resample_commands(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        cfg = self.cfg
        n = (len(envs_idx),)
        self.commands[envs_idx, 0] = rand_uniform(*cfg.lin_vel_x_range, n, self.device)
        self.commands[envs_idx, 1] = rand_uniform(*cfg.lin_vel_y_range, n, self.device)
        self.commands[envs_idx, 2] = rand_uniform(*cfg.ang_vel_range, n, self.device)

    # ------------------------------------------------------------------
    # Reward terms (official Genesis 6-term set)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(
            self.commands[:, :2] - self.robot.state.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(
            self.commands[:, 2] - self.robot.state.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.tracking_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.robot.state.base_lin_vel[:, 2])

    def _reward_base_height(self):
        return torch.square(
            self.robot.state.base_pos[:, 2] - self.cfg.base_height_target)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(
            self.robot.state.dof_pos - self.robot.default_dof_pos), dim=1)
