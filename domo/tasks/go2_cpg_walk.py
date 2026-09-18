"""
Go2 CPG-RL locomotion task (flat terrain, free space).

Functional port of scripts/house_scene/go2_cpg_rl.py onto the layered
architecture. Reproduces Bellegarda & Ijspeert, "CPG-RL: Learning Central
Pattern Generators for Quadruped Locomotion", RA-L 2022: the policy
modulates per-leg oscillators (mu, omega, psi); oscillator states map to
foot targets, closed-form IK, joint PD.

Observation (76):
  [0:3]   base linear velocity  (body, ×2.0)
  [3:6]   base angular velocity (body, ×0.25)
  [6:9]   projected gravity
  [9:12]  velocity command (scaled)
  [12:24] joint pos − default (×1.0)
  [24:36] joint velocities (×0.05)
  [36:48] previous action
  [48:52] foot contacts
  [52:76] CPG state: r, rdot, cos/sin(theta), cos/sin(phi)

Action (12): raw (mu ×4, omega ×4, psi ×4); tanh-squashed inside the CPG
controller.

The observation layout is the canonical interface of every CPG locomotion
checkpoint — `build_cpg_observation` is reused by tasks that drive a frozen
CPG policy (see go2_avoid.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# build_cpg_observation lives in domo.control.cpg (canonical home shared with
# CPGLocomotionSkill); re-exported here for backward compatibility.
from domo.control import (
    CPG_OBS_DIM,
    CPG_OBS_SCALES,
    CPGConfig,
    CPGLegController,
    LegKinematics,
    build_cpg_observation,
)
from domo.robot import GO2, Robot, RobotSpec
from domo.sim import SimConfig, ViewerConfig, create_engine

from .base import VecTask, rand_uniform

__all__ = [
    "CPG_ACT_DIM",
    "CPG_OBS_DIM",
    "CPG_OBS_SCALES",
    "Go2CPGWalkConfig",
    "Go2CPGWalkTask",
    "build_cpg_observation",
]

CPG_ACT_DIM = 12


@dataclass
class Go2CPGWalkConfig:
    # Vectorisation / timing
    n_envs: int = 4096
    dt: float = 0.02
    max_episode_steps: int = 1000
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"

    # Control (paper gains)
    kp: float = 100.0
    kd: float = 2.0
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)
    cpg: CPGConfig = field(default_factory=CPGConfig)

    # Commands
    resampling_time_s: float = 4.0
    lin_vel_x_range: tuple[float, float] = (0.3, 3.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)

    # Termination
    termination_pitch: float = 1.0
    termination_roll: float = 1.0
    termination_height: float = 0.18

    # Rewards (paper weights, signed)
    tracking_sigma: float = 0.5
    reward_clamp: float = 10.0
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        "tracking_lin_vel_x": 0.75,
        "tracking_lin_vel_y": 0.75,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "work": -0.001,
    })


class Go2CPGWalkTask(VecTask):

    OBS_DIM = CPG_OBS_DIM
    ACT_DIM = CPG_ACT_DIM

    def __init__(self, cfg: Go2CPGWalkConfig, spec: RobotSpec = GO2):
        engine = create_engine(cfg.engine, device=cfg.device)
        device = engine.device
        super().__init__(cfg.n_envs, self.OBS_DIM, self.ACT_DIM, device,
                         cfg.dt, max_episode_length=cfg.max_episode_steps)
        self.cfg = cfg
        self.episode_length_s = cfg.max_episode_steps * cfg.dt

        self.scene = engine.create_scene(SimConfig(
            dt=cfg.dt, substeps=2, device=cfg.device, headless=cfg.headless,
            solver_iterations=100,
            viewer=ViewerConfig(camera_pos=(2.0, -2.0, 1.5),
                                camera_lookat=(0.0, 0.0, 0.3),
                                camera_fov=50.0),
        ))
        self.scene.add_ground()

        self.robot = Robot(spec, self.scene, device, kp=cfg.kp, kd=cfg.kd,
                           base_init_pos=cfg.base_init_pos)
        self.scene.build(cfg.n_envs)
        self.robot.bind(cfg.n_envs)

        # CPG controller + IK self-test
        self.kinematics = LegKinematics(spec.geometry, device)
        self.controller = CPGLegController(cfg.cpg, self.kinematics,
                                           cfg.n_envs, device)
        ik_err = self.kinematics.self_test(self.robot.default_dof_pos)
        contact_mode = "force" if self.robot.contact_sensor else "phase-proxy"
        print(f"\n{'=' * 58}")
        print("  Go2 CPG-RL task")
        print(f"{'=' * 58}")
        print(f"  Envs          : {cfg.n_envs}")
        print(f"  Obs / Act     : {self.OBS_DIM} / {self.ACT_DIM}")
        print(f"  Foot contacts : {contact_mode}")
        print(f"  IK self-test  : max round-trip {ik_err:.2e} rad"
              f"  [{'OK' if ik_err < 1e-4 else 'WARN — check leg geometry'}]")
        print(f"{'=' * 58}\n")

        self.register_rewards(cfg.reward_scales)

        N, f = cfg.n_envs, torch.float32
        self.commands = torch.zeros((N, 3), device=device, dtype=f)
        self.commands_scale = torch.tensor(
            [CPG_OBS_SCALES["lin_vel"], CPG_OBS_SCALES["lin_vel"],
             CPG_OBS_SCALES["ang_vel"]], device=device, dtype=f)
        self.actions = torch.zeros((N, self.ACT_DIM), device=device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.target_dof_pos = self.robot.default_dof_pos.repeat(N, 1).clone()
        self.applied_torque = torch.zeros_like(self.actions)

    # ------------------------------------------------------------------

    def _update_foot_contacts(self):
        """Force-based if the backend/asset support it, else stance phase."""
        if self.robot.contact_sensor is None:
            self.robot.state.foot_contacts[:] = self.controller.oscillators.stance_mask()

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        state = self.robot.state

        self.actions = actions
        self.target_dof_pos = self.controller.joint_targets(actions, cfg.dt)
        self.robot.set_joint_targets(self.target_dof_pos)
        self.scene.step()

        self.episode_length_buf += 1
        self.robot.refresh()
        self.applied_torque = (cfg.kp * (self.target_dof_pos - state.dof_pos)
                               - cfg.kd * state.dof_vel)
        self._update_foot_contacts()

        resample_every = int(cfg.resampling_time_s / cfg.dt)
        envs_idx = ((self.episode_length_buf % resample_every == 0)
                    .nonzero(as_tuple=False).flatten())
        self._resample_commands(envs_idx)

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(state.base_euler[:, 1]) > cfg.termination_pitch
        self.reset_buf |= torch.abs(state.base_euler[:, 0]) > cfg.termination_roll
        self.reset_buf |= state.base_pos[:, 2] < cfg.termination_height

        time_out_idx = ((self.episode_length_buf > self.max_episode_length)
                        .nonzero(as_tuple=False).flatten())
        self.extras["time_outs"] = torch.zeros(
            self.n_envs, device=self.device, dtype=torch.float32)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.compute_rewards()
        self.rew_buf = torch.clamp(self.rew_buf, -cfg.reward_clamp, cfg.reward_clamp)

        self.obs_buf = build_cpg_observation(
            state, self.commands, self.commands_scale,
            self.robot.default_dof_pos, self.last_actions,
            state.foot_contacts, self.controller.oscillators)

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = state.dof_vel

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.robot.refresh()
        self._update_foot_contacts()
        self.obs_buf = build_cpg_observation(
            self.robot.state, self.commands, self.commands_scale,
            self.robot.default_dof_pos, self.last_actions,
            self.robot.state.foot_contacts, self.controller.oscillators)
        assert self.obs_buf.shape[-1] == self.OBS_DIM
        return self.obs_buf, None

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        self.robot.reset_idx(envs_idx)
        self.controller.reset_idx(envs_idx)

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
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
    # Reward terms (CPG-RL paper Sec III-C)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        return torch.exp(-torch.square(
            self.commands[:, 0] - self.robot.state.base_lin_vel[:, 0])
            / self.cfg.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        return torch.exp(-torch.square(
            self.commands[:, 1] - self.robot.state.base_lin_vel[:, 1])
            / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        return torch.exp(-torch.square(
            self.commands[:, 2] - self.robot.state.base_ang_vel[:, 2])
            / self.cfg.tracking_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.robot.state.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.robot.state.base_ang_vel[:, :2]), dim=1)

    def _reward_work(self):
        dqd = self.robot.state.dof_vel - self.last_dof_vel
        return torch.abs(torch.sum(self.applied_torque * dqd, dim=1))
