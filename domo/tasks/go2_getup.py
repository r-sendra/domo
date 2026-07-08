"""
Go2 get-up / recovery task — the reward-injection target for Eureka (M2/M3).

The robot spawns FALLEN (random roll near ±π/2..π, random yaw, on the
ground) and must right itself and hold a standing pose. The task ships with
NO reward: it is designed to receive an LLM-generated reward via
`set_reward_override()`. What it does fix — deliberately outside the
generated code's reach — is the SUCCESS METRIC:

    success  =  base height > 0.26 m
              ∧ |roll| < 0.4 ∧ |pitch| < 0.4
              held for `success_hold_steps` consecutive steps

Candidates are ranked by episode success rate, never by their own reward.

Observation (42):
  [0:3]   base angular velocity (body, ×0.25)
  [3:6]   projected gravity (body)
  [6:18]  joint pos − default
  [18:30] joint velocities (×0.05)
  [30:42] previous action

Action (12): joint-position offsets from the default stance × action_scale.

Supports DomainRandomization (physics resampled per reset + optional obs
noise) — the surface DrEureka optimises over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from domo.robot import GO2, RobotSpec
from domo.robot.randomization import DomainRandomization
from domo.robot.state import RobotState

from .base import VecTask, rand_uniform

__all__ = ["Go2GetUpConfig", "Go2GetUpTask", "build_getup_observation",
           "GETUP_OBS_DIM", "GETUP_ACT_DIM"]

GETUP_OBS_DIM = 42
GETUP_ACT_DIM = 12

_OBS_SCALES = {"ang_vel": 0.25, "dof_vel": 0.05}


def build_getup_observation(state: RobotState, default_dof_pos: torch.Tensor,
                            last_actions: torch.Tensor) -> torch.Tensor:
    """Shared by the task and the deployed LearnedJointSkill."""
    return torch.cat([
        state.base_ang_vel * _OBS_SCALES["ang_vel"],     # 3
        state.projected_gravity,                          # 3
        state.dof_pos - default_dof_pos,                  # 12
        state.dof_vel * _OBS_SCALES["dof_vel"],           # 12
        last_actions,                                     # 12
    ], dim=-1)


@dataclass
class Go2GetUpConfig:
    n_envs: int = 4096
    dt: float = 0.02
    max_episode_steps: int = 400          # 8 s to get up (was 5 s — too tight)
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"

    # Control
    action_scale: float = 0.5             # wider joint range for the flip-up
    clip_actions: float = 100.0
    kp: float = 100.0
    kd: float = 2.0

    # Fallen spawn. |roll| range excludes the near-inverted (belly-up) poses
    # that are nearly impossible to bootstrap from scratch; a real curriculum
    # can widen this once a base policy exists.
    spawn_height: float = 0.18
    spawn_roll_range: Tuple[float, float] = (math.pi / 3, 5 * math.pi / 6)  # 60°–150°
    spawn_joint_noise: float = 0.3        # rad, around default angles

    # Fixed success metric (reward-independent). Standing height is ~0.32 m,
    # so 0.26 m is reachable with margin. Hold is 0.5 s: a just-righted robot
    # can rarely hold a full second, and "stood up and stayed up half a
    # second" is still a legitimate success — this lets the metric register
    # for a policy that demonstrably reaches the pose.
    success_height: float = 0.26
    success_tilt: float = 0.4             # rad, roll AND pitch
    success_hold_steps: int = 25          # 0.5 s upright (was 50 / 1 s)

    # Domain randomization (serialised as dict for checkpoints/specs)
    dr: Optional[dict] = None


class Go2GetUpTask(VecTask):

    OBS_DIM = GETUP_OBS_DIM
    ACT_DIM = GETUP_ACT_DIM

    def __init__(self, cfg: Go2GetUpConfig, spec: RobotSpec = GO2):
        from domo.world import World, WorldConfig

        world = World(WorldConfig(
            engine=cfg.engine, device=cfg.device, dt=cfg.dt,
            headless=cfg.headless, scene_kind="flat",
            kp=cfg.kp, kd=cfg.kd,
            base_init_pos=(0.0, 0.0, cfg.spawn_height),
            lidar_model=None,
        ), n_envs=cfg.n_envs, spec=spec)

        super().__init__(cfg.n_envs, self.OBS_DIM, self.ACT_DIM, world.device,
                         cfg.dt, max_episode_length=cfg.max_episode_steps)
        self.cfg = cfg
        self.world = world
        self.scene = world.scene
        self.robot = world.robot
        self.episode_length_s = cfg.max_episode_steps * cfg.dt

        self.dr = DomainRandomization.from_dict(cfg.dr) if cfg.dr else None

        N, f = cfg.n_envs, torch.float32
        device = world.device
        self.actions = torch.zeros((N, self.ACT_DIM), device=device, dtype=f)
        self.last_actions = torch.zeros_like(self.actions)
        self._hold = torch.zeros((N,), device=device, dtype=torch.int32)
        # Per-episode progress trackers (dense fitness + diagnostics).
        self._max_fitness = torch.zeros((N,), device=device, dtype=f)
        self._peak_height = torch.zeros((N,), device=device, dtype=f)
        self._ever_upright = torch.zeros((N,), device=device, dtype=torch.bool)
        self._max_hold = torch.zeros((N,), device=device, dtype=torch.int32)
        # Rolling per-episode records: dict(success, fitness, peak_height,
        # ever_upright). Read by evaluation/workers.
        self.episode_outcomes: list = []

        print(f"\n{'=' * 58}")
        print(f"  Go2 Get-Up task (reward-injection target)")
        print(f"{'=' * 58}")
        print(f"  Envs     : {cfg.n_envs}")
        print(f"  Obs/Act  : {self.OBS_DIM} / {self.ACT_DIM}")
        print(f"  Success  : h>{cfg.success_height} & tilt<{cfg.success_tilt} "
              f"for {cfg.success_hold_steps} steps")
        print(f"  Reward   : "
              f"{'injected' if self._reward_override else 'NONE (inject via set_reward_override)'}")
        print(f"  DR       : {cfg.dr or 'off'}")
        print(f"{'=' * 58}\n")

    # ------------------------------------------------------------------

    def compute_success(self) -> torch.Tensor:
        state = self.robot.state
        upright = state.base_pos[:, 2] > self.cfg.success_height
        upright &= state.base_euler[:, 0].abs() < self.cfg.success_tilt
        upright &= state.base_euler[:, 1].abs() < self.cfg.success_tilt
        return upright

    def compute_fitness(self) -> torch.Tensor:
        """
        Dense, reward-independent progress in [0, 1] — Eureka's fitness F.
        The binary success metric is too sparse to rank reward candidates
        before any of them fully succeeds; this continuous proxy (how
        upright AND how high the base is) gives the evolutionary search a
        gradient to climb from the very first iteration.
          uprightness: 1 when the base z-axis points up, 0 on its side.
          height:      base height as a fraction of the success height.
        """
        state = self.robot.state
        uprightness = torch.clamp(-state.projected_gravity[:, 2], 0.0, 1.0)
        height = torch.clamp(state.base_pos[:, 2] / self.cfg.success_height,
                             0.0, 1.0)
        return uprightness * height

    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        state = self.robot.state

        self.actions = torch.clip(actions, -cfg.clip_actions, cfg.clip_actions)
        targets = self.actions * cfg.action_scale + self.robot.default_dof_pos
        self.robot.set_joint_targets(targets)
        self.scene.step()

        self.episode_length_buf += 1
        self.robot.refresh()

        # Dense progress (Eureka fitness) + binary success metric.
        fitness = self.compute_fitness()
        self._max_fitness = torch.maximum(self._max_fitness, fitness)
        self._peak_height = torch.maximum(self._peak_height, state.base_pos[:, 2])
        upright = self.compute_success()
        self._ever_upright |= upright
        self._hold = torch.where(upright, self._hold + 1,
                                 torch.zeros_like(self._hold))
        self._max_hold = torch.maximum(self._max_hold, self._hold)
        succeeded = self._hold >= cfg.success_hold_steps

        # Termination: success or timeout (no fall termination — the robot
        # STARTS fallen; thrashing simply wastes its budget).
        timeout = self.episode_length_buf > self.max_episode_length
        self.reset_buf = succeeded | timeout

        self.extras["time_outs"] = timeout.float()
        self.extras["success"] = succeeded.float()
        for idx in self.reset_buf.nonzero(as_tuple=False).flatten():
            self.episode_outcomes.append({
                "success": bool(succeeded[idx]),
                "fitness": float(self._max_fitness[idx]),
                "peak_height": float(self._peak_height[idx]),
                "ever_upright": bool(self._ever_upright[idx]),
                "max_hold": int(self._max_hold[idx]),
            })

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Reward: injected (Eureka) or zero
        if self._reward_override is not None:
            self.compute_rewards()
        else:
            self.rew_buf[:] = 0.0

        obs = build_getup_observation(state, self.robot.default_dof_pos,
                                      self.last_actions)
        if self.dr is not None and self.dr.obs_noise_std > 0:
            obs = obs + torch.randn_like(obs) * self.dr.obs_noise_std
        self.obs_buf = obs

        self.last_actions[:] = self.actions
        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.robot.refresh()
        self.obs_buf = build_getup_observation(
            self.robot.state, self.robot.default_dof_pos, self.last_actions)
        return self.obs_buf, None

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        cfg = self.cfg
        n = len(envs_idx)

        self.robot.reset_idx(envs_idx)

        # Overwrite the spawn pose: FALLEN, random side, random yaw.
        roll = rand_uniform(*cfg.spawn_roll_range, (n,), self.device)
        roll *= torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        yaw = rand_uniform(-math.pi, math.pi, (n,), self.device)
        cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        # wxyz for R = Rz(yaw) · Rx(roll)
        quat = torch.stack([cy * cr, cy * sr, sy * sr, sy * cr], dim=-1)
        pos = self.robot.base_init_pos.unsqueeze(0).repeat(n, 1)
        self.robot.articulation.set_base_pose(pos, quat, envs_idx)
        self.robot.state.base_quat[envs_idx] = quat

        # Scrambled joints around default
        joints = (self.robot.default_dof_pos.unsqueeze(0)
                  + cfg.spawn_joint_noise
                  * (2 * torch.rand((n, 12), device=self.device) - 1))
        self.robot.articulation.set_joint_positions(
            joints, self.robot.dof_idx, envs_idx, zero_velocity=True)
        self.robot.state.dof_pos[envs_idx] = joints

        if self.dr is not None:
            self.dr.apply(self.robot, envs_idx)

        self.last_actions[envs_idx] = 0.0
        self._hold[envs_idx] = 0
        self._max_hold[envs_idx] = 0
        self._max_fitness[envs_idx] = 0.0
        self._peak_height[envs_idx] = 0.0
        self._ever_upright[envs_idx] = False
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.log_episode_sums(envs_idx, self.episode_length_s)
