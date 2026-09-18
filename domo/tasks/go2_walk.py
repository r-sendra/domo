"""
Go2 velocity-tracking locomotion task (flat or rough terrain).

Functional port of the original `domo/robot/go2.py::Go2WalkEnv` (which
followed the official Genesis locomotion example) onto the layered
architecture: physics access goes through `domo.sim` handles wrapped in the
robot's sensors/actuators; all math is engine-agnostic. This is the task
`main.py` trains; the CPG variant (go2_cpg_walk.py) is a separate task with
a different action space and is deliberately NOT merged with this one.

Observation (45, float32)::

    [0:3]   base angular velocity (body frame, × obs_scales["ang_vel"]=0.25)
    [3:6]   projected gravity (body frame, unit vector; [0,0,-1] upright)
    [6:9]   velocity command (vx, vy, vyaw) × commands_scale (2, 2, 0.25)
    [9:21]  joint pos − default stance (× obs_scales["dof_pos"]=1.0)
    [21:33] joint velocities (× obs_scales["dof_vel"]=0.05)
    [33:45] current action (the one just applied, after clipping)

Action (12): joint-position offsets from the default stance, in the order
of `RobotSpec.joint_names`; target = action × action_scale + default. With
`simulate_action_latency` the PREVIOUS action is executed (matches the ~1
step delay of the real Go2 command pipeline).

Termination: |pitch| or |roll| beyond the limits, base height below
`termination_height` (flat terrain only), or the episode budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from domo.robot import GO2, Robot, RobotSpec
from domo.sim import SimConfig, TerrainConfig, ViewerConfig, create_engine

from .base import VecTask
from .common import envs_due_for_resample, fall_termination, sample_velocity_commands

__all__ = ["Go2WalkConfig", "Go2WalkTask"]


@dataclass
class Go2WalkConfig:
    """
    Go2WalkTask parameters. Serialised with `dataclasses.asdict` into every
    checkpoint (`extra["task_config"]`) so a run can be rebuilt exactly —
    field names and defaults are therefore part of the checkpoint contract.
    """

    # Vectorisation / timing
    n_envs: int = 4096
    dt: float = 0.02                         # control period [s] (50 Hz)
    max_episode_steps: int = 1000            # 20 s
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"
    terrain: str = "flat"                    # "flat" | "rough"

    # Control
    action_scale: float = 0.25               # [rad] per unit action
    clip_actions: float = 100.0              # symmetric clip on raw actions
    kp: float = 20.0                         # joint PD gains (Genesis example)
    kd: float = 0.5
    simulate_action_latency: bool = True     # real Go2 has ~1 step delay

    # Commands: (low, high) ranges resampled every `resampling_time_s`
    resampling_time_s: float = 4.0
    lin_vel_x_range: tuple[float, float] = (0.5, 0.5)   # [m/s]
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)   # [m/s]
    ang_vel_range: tuple[float, float] = (0.0, 0.0)     # [rad/s]

    # Termination
    termination_pitch: float = 1.0           # [rad]
    termination_roll: float = 1.0            # [rad]
    termination_height: float = 0.20         # [m], flat terrain only

    # Rewards: name → weight (× dt at registration); see `_reward_<name>`
    tracking_sigma: float = 0.25             # exp(-err²/σ) width for tracking
    base_height_target: float = 0.34         # [m]
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.2,
        "lin_vel_z": -1.0,
        "base_height": -50.0,
        "action_rate": -0.005,
        "similar_to_default": -0.1,
    })

    # Observation scales (lin_vel is only used for the command scaling here)
    obs_scales: dict[str, float] = field(default_factory=lambda: {
        "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05,
    })

    # Rough terrain
    rough_terrain: TerrainConfig = field(default_factory=TerrainConfig)
    rough_spawn_height: float = 0.5          # [m] drop height above the tiles


class Go2WalkTask(VecTask):
    """
    Velocity-tracking locomotion with direct joint-position actions.

    Reward terms (official Genesis 6-term set; weights in
    `Go2WalkConfig.reward_scales`):
        tracking_lin_vel   exp(-|v_cmd_xy − v_xy|² / σ)       follow the command
        tracking_ang_vel   exp(-(ω_cmd − ω_z)² / σ)           follow the yaw rate
        lin_vel_z          v_z²                               (−) no bouncing
        base_height        (h − h_target)²                    (−) keep nominal height
        action_rate        |a_t − a_{t−1}|²                   (−) smooth actions
        similar_to_default Σ|q − q_default|                   (−) stay near stance
    """

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
        self.commands = torch.zeros((N, 3), device=device, dtype=f)   # vx, vy, vyaw
        self.commands_scale = torch.tensor(
            [cfg.obs_scales["lin_vel"], cfg.obs_scales["lin_vel"],
             cfg.obs_scales["ang_vel"]], device=device, dtype=f)
        self.actions = torch.zeros((N, self.ACT_DIM), device=device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)

    # ------------------------------------------------------------------

    def _terrain_spawn_grid(self, cfg: Go2WalkConfig) -> torch.Tensor:
        """One spawn point [N, 3] at the centre of each subterrain tile, cycled."""
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
        self._resample_commands(envs_due_for_resample(
            self.episode_length_buf, cfg.resampling_time_s, cfg.dt))

        # Termination (height test only on flat ground: absolute height is
        # meaningless over rough tiles).
        self.reset_buf = self.mark_time_outs()
        self.reset_buf |= fall_termination(
            state, cfg.termination_pitch, cfg.termination_roll,
            cfg.termination_height if cfg.terrain == "flat" else None)

        # Reset BEFORE rewards/obs (legged-gym order): finished envs are
        # scored on their fresh spawn state and the policy sees it next.
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
        """Reset all envs. Returns the STALE obs_buf (legged-gym behaviour)."""
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
        cfg = self.cfg
        sample_velocity_commands(self.commands, envs_idx, cfg.lin_vel_x_range,
                                 cfg.lin_vel_y_range, cfg.ang_vel_range, self.device)

    # ------------------------------------------------------------------
    # Reward terms (official Genesis 6-term set)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel(self):
        """exp(-‖v_cmd_xy − v_xy‖² / σ): planar velocity tracking, in [0, 1]."""
        error = torch.sum(torch.square(
            self.commands[:, :2] - self.robot.state.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """exp(-(ω_cmd − ω_z)² / σ): yaw-rate tracking, in [0, 1]."""
        error = torch.square(
            self.commands[:, 2] - self.robot.state.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.tracking_sigma)

    def _reward_lin_vel_z(self):
        """v_z² (penalty): discourages vertical bouncing."""
        return torch.square(self.robot.state.base_lin_vel[:, 2])

    def _reward_base_height(self):
        """(h − h_target)² (penalty): keep the nominal standing height."""
        return torch.square(
            self.robot.state.base_pos[:, 2] - self.cfg.base_height_target)

    def _reward_action_rate(self):
        """‖a_t − a_{t−1}‖² (penalty): smooth joint targets."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Σ|q − q_default| (penalty): stay close to the default stance."""
        return torch.sum(torch.abs(
            self.robot.state.dof_pos - self.robot.default_dof_pos), dim=1)
