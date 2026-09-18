"""
Go2 CPG-RL locomotion task (flat terrain, free space).

Functional port of scripts/house_scene/go2_cpg_rl.py onto the layered
architecture. Reproduces Bellegarda & Ijspeert, "CPG-RL: Learning Central
Pattern Generators for Quadruped Locomotion", RA-L 2022: the policy
modulates per-leg oscillators (mu, omega, psi); oscillator states map to
foot targets, closed-form IK, joint PD. The oscillators and IK live in
`domo.control` so the same stack drives the frozen policy at deployment
(CPGLocomotionSkill) and inside other tasks (go2_avoid.py).

Observation (76, float32) — built by `build_cpg_observation`::

    [0:3]   base linear velocity  (body, ×2.0)
    [3:6]   base angular velocity (body, ×0.25)
    [6:9]   projected gravity
    [9:12]  velocity command (vx, vy, vyaw) × (2.0, 2.0, 0.25)
    [12:24] joint pos − default (×1.0)
    [24:36] joint velocities (×0.05)
    [36:48] previous action (the one applied at the last step)
    [48:52] foot contacts (FR, FL, RR, RL; force-based or stance-phase proxy)
    [52:76] CPG state: r, rdot, cos/sin(theta), cos/sin(phi)  (4 legs each)

Action (12): raw (mu ×4, omega ×4, psi ×4); tanh-squashed and mapped into
`CPGConfig` ranges inside the CPG controller, so the policy output is
unbounded and no clipping happens in the task.

The observation layout is the canonical interface of every CPG locomotion
checkpoint (policies/walk.pt, runs/go2_cpg/*): `build_cpg_observation` is
re-exported here so tasks that drive a frozen CPG policy share it exactly.

Termination: |pitch| / |roll| beyond the limits, base height below
`termination_height`, or the episode budget.
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

from .base import VecTask
from .common import envs_due_for_resample, fall_termination, sample_velocity_commands

__all__ = [
    "CPG_ACT_DIM",
    "CPG_OBS_DIM",
    "CPG_OBS_SCALES",
    "Go2CPGWalkConfig",
    "Go2CPGWalkTask",
    "build_cpg_observation",
]

CPG_ACT_DIM = 12

# IK round-trip error above which the leg geometry is probably wrong.
_IK_SELF_TEST_TOL = 1e-4


@dataclass
class Go2CPGWalkConfig:
    """
    Go2CPGWalkTask parameters. Serialised into every checkpoint
    (`extra["task_config"]`); field names and defaults are part of the
    checkpoint contract.
    """

    # Vectorisation / timing
    n_envs: int = 4096
    dt: float = 0.02                         # control period [s] (50 Hz)
    max_episode_steps: int = 1000            # 20 s
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"

    # Control (paper gains)
    kp: float = 100.0
    kd: float = 2.0
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.35)   # [m]
    cpg: CPGConfig = field(default_factory=CPGConfig)

    # Commands: (low, high) ranges resampled every `resampling_time_s`
    resampling_time_s: float = 4.0
    lin_vel_x_range: tuple[float, float] = (0.3, 3.0)   # [m/s]
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)  # [m/s]
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)    # [rad/s]

    # Termination
    termination_pitch: float = 1.0           # [rad]
    termination_roll: float = 1.0            # [rad]
    termination_height: float = 0.18         # [m]

    # Rewards (paper weights, signed; × dt at registration)
    tracking_sigma: float = 0.5
    reward_clamp: float = 10.0               # symmetric clamp on the total reward
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        "tracking_lin_vel_x": 0.75,
        "tracking_lin_vel_y": 0.75,
        "tracking_ang_vel": 0.75,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "work": -0.001,
    })


class Go2CPGWalkTask(VecTask):
    """
    Velocity-tracking locomotion where the policy modulates CPG oscillators.

    Reward terms (CPG-RL paper Sec III-C; weights in
    `Go2CPGWalkConfig.reward_scales`):
        tracking_lin_vel_x  exp(-(vx_cmd − vx)² / σ)     follow forward speed
        tracking_lin_vel_y  exp(-(vy_cmd − vy)² / σ)     follow lateral speed
        tracking_ang_vel    exp(-(ω_cmd − ω_z)² / σ)     follow yaw rate
        lin_vel_z           v_z²                         (−) no bouncing
        ang_vel_xy          ω_x² + ω_y²                  (−) level base
        work                |Σ τ · Δq̇|                   (−) actuator effort
    The summed reward is clamped to ±`reward_clamp`.

    Foot contacts come from the contact sensor when the backend exposes
    forces; otherwise the oscillator stance phase is the proxy (as in the
    original script) so the observation layout never changes.
    """

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
        self._print_banner()

        self.register_rewards(cfg.reward_scales)

        N, f = cfg.n_envs, torch.float32
        self.commands = torch.zeros((N, 3), device=device, dtype=f)   # vx, vy, vyaw
        self.commands_scale = torch.tensor(
            [CPG_OBS_SCALES["lin_vel"], CPG_OBS_SCALES["lin_vel"],
             CPG_OBS_SCALES["ang_vel"]], device=device, dtype=f)
        self.actions = torch.zeros((N, self.ACT_DIM), device=device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.target_dof_pos = self.robot.default_dof_pos.repeat(N, 1).clone()
        # PD torque estimate [N, 12] recomputed after each step (for `work`).
        self.applied_torque = torch.zeros_like(self.actions)

    def _print_banner(self) -> None:
        cfg = self.cfg
        ik_err = self.kinematics.self_test(self.robot.default_dof_pos)
        contact_mode = "force" if self.robot.contact_sensor else "phase-proxy"
        ik_status = "OK" if ik_err < _IK_SELF_TEST_TOL else "WARN — check leg geometry"
        print(f"\n{'=' * 58}")
        print("  Go2 CPG-RL task")
        print(f"{'=' * 58}")
        print(f"  Envs          : {cfg.n_envs}")
        print(f"  Obs / Act     : {self.OBS_DIM} / {self.ACT_DIM}")
        print(f"  Foot contacts : {contact_mode}")
        print(f"  IK self-test  : max round-trip {ik_err:.2e} rad  [{ik_status}]")
        print(f"{'=' * 58}\n")

    # ------------------------------------------------------------------

    def _update_foot_contacts(self):
        """Force-based if the backend/asset support it, else stance phase."""
        if self.robot.contact_sensor is None:
            self.robot.state.foot_contacts[:] = self.controller.oscillators.stance_mask()

    def _observe(self) -> torch.Tensor:
        """The canonical 76-dim CPG observation from the current state."""
        state = self.robot.state
        return build_cpg_observation(
            state, self.commands, self.commands_scale,
            self.robot.default_dof_pos, self.last_actions,
            state.foot_contacts, self.controller.oscillators)

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

        self._resample_commands(envs_due_for_resample(
            self.episode_length_buf, cfg.resampling_time_s, cfg.dt))

        self.reset_buf = self.mark_time_outs()
        self.reset_buf |= fall_termination(
            state, cfg.termination_pitch, cfg.termination_roll,
            cfg.termination_height)

        # Reset BEFORE rewards/obs (legged-gym order).
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.compute_rewards()
        self.rew_buf = torch.clamp(self.rew_buf, -cfg.reward_clamp, cfg.reward_clamp)

        self.obs_buf = self._observe()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = state.dof_vel

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs and return a FRESH observation of the spawn state."""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.robot.refresh()
        self._update_foot_contacts()
        self.obs_buf = self._observe()
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
        cfg = self.cfg
        sample_velocity_commands(self.commands, envs_idx, cfg.lin_vel_x_range,
                                 cfg.lin_vel_y_range, cfg.ang_vel_range, self.device)

    # ------------------------------------------------------------------
    # Reward terms (CPG-RL paper Sec III-C)
    # ------------------------------------------------------------------

    def _reward_tracking_lin_vel_x(self):
        """exp(-(vx_cmd − vx)² / σ): forward-speed tracking, in [0, 1]."""
        return torch.exp(-torch.square(
            self.commands[:, 0] - self.robot.state.base_lin_vel[:, 0])
            / self.cfg.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        """exp(-(vy_cmd − vy)² / σ): lateral-speed tracking, in [0, 1]."""
        return torch.exp(-torch.square(
            self.commands[:, 1] - self.robot.state.base_lin_vel[:, 1])
            / self.cfg.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """exp(-(ω_cmd − ω_z)² / σ): yaw-rate tracking, in [0, 1]."""
        return torch.exp(-torch.square(
            self.commands[:, 2] - self.robot.state.base_ang_vel[:, 2])
            / self.cfg.tracking_sigma)

    def _reward_lin_vel_z(self):
        """v_z² (penalty): discourages vertical bouncing."""
        return torch.square(self.robot.state.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """ω_x² + ω_y² (penalty): keep the base level."""
        return torch.sum(torch.square(self.robot.state.base_ang_vel[:, :2]), dim=1)

    def _reward_work(self):
        """
        |Σ τ · Δq̇| (penalty): mechanical effort proxy using the PD torque
        estimate and the joint-velocity change over the last step.
        """
        dqd = self.robot.state.dof_vel - self.last_dof_vel
        return torch.abs(torch.sum(self.applied_torque * dqd, dim=1))
