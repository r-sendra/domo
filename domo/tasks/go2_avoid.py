"""
Go2 obstacle avoidance on top of a frozen locomotion policy.

Functional port of scripts/house_scene/go2_cpg_rl_lidar.py (obstacle arena)
and go2_cpg_rl_avoid_house.py (ReplicaCAD house) as one task with two scene
modes.

Architecture (two decoupled layers)::

    LiDAR (36 sectors) → avoidance policy → velocity correction (Δvx, Δvy, Δvyaw)
    base command + correction → FROZEN CPG locomotion policy → joint targets

The locomotion policy is injected as a plain callable (obs[N,76] → action
[N,12]); the task never loads checkpoints or knows about network classes —
that wiring belongs to the entry point (see domo.checkpoints /
domo.policies). This is the same decoupling that made joint training
tractable in the scripts: avoidance reward is the only learning signal,
locomotion is pre-solved.

Observation (36, float32): lidar sector distances / `obs_max_range`,
clamped to [0, 1] (1 = nothing within range). Sector 0 starts at the
robot's +x and sectors run counter-clockwise (see domo.robot.sensors).

Action (3): raw corrections, tanh-squashed into ±(delta_vx_max,
delta_vy_max, delta_vyaw_max) and added to `base_command`; the final
command is clamped to the vx/vy/vyaw clamp ranges.

Termination: tilt/height fall (as the locomotion tasks), leaving the arena
(arena mode only), lidar minimum distance below `d_collision` (once a scan
exists), or the episode budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from domo.control import CPGConfig, CPGLocomotionSkill
from domo.robot import GO2, RobotSpec
from domo.robot.lidar_models import LidarModelConfig, generic_sector_lidar

# ObstacleArenaConfig re-exported for backward compatibility; the builder
# lives in domo.scenes so the goal-free World shares the same environments.
from domo.scenes.arena import ObstacleArenaConfig

from .base import VecTask
from .common import fall_termination

__all__ = [
    "AVOID_ACT_DIM",
    "AVOID_OBS_DIM",
    "Go2AvoidConfig",
    "Go2AvoidTask",
    "ObstacleArenaConfig",
]

AVOID_OBS_DIM = 36
AVOID_ACT_DIM = 3


@dataclass
class Go2AvoidConfig:
    """
    Go2AvoidTask parameters. Serialised into checkpoints via
    domo.checkpoints (nested dataclasses included); field names and
    defaults are part of that contract.
    """

    # Vectorisation / timing
    n_envs: int = 4096
    dt: float = 0.02                         # control period [s] (50 Hz)
    max_episode_steps: int = 1000            # 20 s
    device: str = "cuda"
    headless: bool = True
    engine: str = "genesis"

    # Scene: "arena" (walled arena + random obstacles) or "replica" (house)
    scene_kind: str = "arena"
    arena: ObstacleArenaConfig = field(default_factory=ObstacleArenaConfig)
    replica_scene_json: str = "scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json"
    replica_asset_root: str = "scripts/house_scene/data/replica_cad/"
    ground_height: float = 0.0           # house floor sits at 0.2 in the scripts
    base_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.42)
    base_init_yaw_deg: float = 0.0       # house replica spawned facing 180°

    # Frozen locomotion inner loop (paper gains)
    kp: float = 100.0
    kd: float = 2.0
    cpg: CPGConfig = field(default_factory=CPGConfig)

    # LiDAR device model. Default reproduces the original scripts' idealised
    # 36×5 sensor; swap for domo.robot.lidar_models.hesai_xt16() to simulate
    # the real Hesai XT16 mounted on the robot.
    lidar_model: LidarModelConfig = field(default_factory=generic_sector_lidar)
    # Policy-facing normalisation range [m]: sector distances are clamped to
    # this and divided by it for the observation, regardless of the device's
    # physical max_range (4 m for a 120 m XT16 keeps the obs distribution —
    # and trained checkpoints — unchanged).
    obs_max_range: float = 4.0

    # Reward distance zones [m], nested: collision < danger < caution < anticipate
    d_collision: float = 0.25
    d_danger: float = 0.60
    d_caution: float = 0.90
    d_anticipate: float = 1.40

    # Correction limits (how much avoidance can override the base command)
    delta_vx_max: float = 0.8            # [m/s]
    delta_vy_max: float = 0.5            # [m/s]
    delta_vyaw_max: float = 1.5          # [rad/s]

    # Base command + final clamps
    base_command: tuple[float, float, float] = (0.6, 0.0, 0.0)   # vx, vy, vyaw
    vx_clamp: tuple[float, float] = (-1.0, 2.0)
    vy_clamp: tuple[float, float] = (-0.5, 0.5)
    vyaw_clamp: tuple[float, float] = (-1.5, 1.5)

    # Termination
    termination_pitch: float = 1.0       # [rad]
    termination_roll: float = 1.0        # [rad]
    termination_height: float = 0.18     # [m]

    # Reward scales (× dt at registration); see `_reward_<name>`
    reward_scales: dict[str, float] = field(default_factory=lambda: {
        "survival": 1.0,
        "avoidance": 5.0,
        "smoothness": -0.05,
        "command_tracking": 3.0,
    })


class Go2AvoidTask(VecTask):
    """
    Learned velocity corrections around a frozen CPG locomotion skill.

    Args:
        cfg: task configuration.
        locomotion_policy: callable(obs [N, 76]) → raw CPG action [N, 12],
            evaluated under no_grad. Typically a frozen trained ActorCritic
            wrapped by the entry point; a zero-action lambda gives a
            standing robot.
        spec: robot description (default Go2).

    Reward terms (weights in `Go2AvoidConfig.reward_scales`):
        survival          1 per step                          stay alive
        avoidance         −(4-zone proximity penalty)         keep clear
        smoothness        ‖Δcorrection‖²                      (−) no jitter
        command_tracking  clearance × exp(−2‖correction‖²)    prefer the base
                                                              command when clear

    The task does not own the simulation: it is a reward-bearing lens over
    a `domo.world.World` (the goal-free digital-twin runtime), so the same
    WorldConfig spawns the twin without any task attached.
    """

    OBS_DIM = AVOID_OBS_DIM
    ACT_DIM = AVOID_ACT_DIM

    def __init__(self, cfg: Go2AvoidConfig,
                 locomotion_policy: Callable[[torch.Tensor], torch.Tensor],
                 spec: RobotSpec = GO2):
        # Imported lazily: domo.world pulls in scene loaders the RL layer
        # never needs, and domo.tasks must stay importable engine-free.
        from domo.world import World, WorldConfig

        world = World(WorldConfig(
            engine=cfg.engine, device=cfg.device, dt=cfg.dt,
            headless=cfg.headless,
            scene_kind=cfg.scene_kind, ground_height=cfg.ground_height,
            arena=cfg.arena,
            replica_scene_json=cfg.replica_scene_json,
            replica_asset_root=cfg.replica_asset_root,
            kp=cfg.kp, kd=cfg.kd,
            base_init_pos=cfg.base_init_pos,
            base_init_yaw_deg=cfg.base_init_yaw_deg,
            lidar_model=cfg.lidar_model, lidar_sectors=self.OBS_DIM,
        ), n_envs=cfg.n_envs, spec=spec)

        super().__init__(cfg.n_envs, self.OBS_DIM, self.ACT_DIM, world.device,
                         cfg.dt, max_episode_length=cfg.max_episode_steps)
        self.cfg = cfg
        self.locomotion_policy = locomotion_policy
        self.episode_length_s = cfg.max_episode_steps * cfg.dt

        self.world = world
        self.scene = world.scene
        self.robot = world.robot
        self.lidar = world.lidar
        self.arena = world.arena
        # Arena mode: leaving the walled square also ends the episode.
        self._arena_term_dist = (world.arena.termination_distance
                                 if world.arena is not None else None)
        device = world.device

        # The frozen locomotion stack as a Skill primitive; this task acts
        # as its (learned) controller, writing velocity commands into it.
        self.locomotion = CPGLocomotionSkill(locomotion_policy, cfg.cpg)
        self.locomotion.setup(self.robot)

        self.register_rewards(cfg.reward_scales)

        # ---- buffers --------------------------------------------------------
        N, f = cfg.n_envs, torch.float32
        self.commands = torch.zeros((N, 3), device=device, dtype=f)   # final vx, vy, vyaw
        self.base_command = torch.tensor(cfg.base_command, device=device, dtype=f)
        self.correction = torch.zeros((N, 3), device=device, dtype=f)
        self.last_correction = torch.zeros((N, 3), device=device, dtype=f)
        # Closest lidar return per env [N] (physical metres, not normalised).
        self._min_dist = torch.full((N,), cfg.lidar_model.max_range,
                                    device=device, dtype=f)
        self._print_banner()

    def _print_banner(self) -> None:
        cfg = self.cfg
        lm = cfg.lidar_model
        print(f"\n{'=' * 60}")
        print(f"  Go2 Avoidance task ({cfg.scene_kind})")
        print(f"{'=' * 60}")
        print(f"  Envs       : {cfg.n_envs}")
        print(f"  LiDAR      : {lm.name} ({lm.n_vertical}ch × {lm.n_horizontal}az, "
              f"{lm.fov_deg[1]:.0f}° vfov, {lm.rate_hz:.0f} Hz, "
              f"σ={lm.range_noise_std * 100:.1f} cm)")
        print(f"  Avoid obs  : {self.OBS_DIM} sectors (normalised to "
              f"{cfg.obs_max_range:.1f} m)")
        print(f"  Avoid act  : {self.ACT_DIM} (Δvx, Δvy, Δvyaw)")
        print("  Locomotion : frozen policy (injected)")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_foot_contacts(self) -> None:
        """Force-based if the backend/asset support it, else stance phase."""
        if self.robot.contact_sensor is None:
            self.robot.state.foot_contacts[:] = self.locomotion.stance_mask()

    def _observe(self) -> torch.Tensor:
        """Normalised lidar sectors [N, 36] in [0, 1] from the cached scan."""
        return torch.clamp(self.lidar.read() / self.cfg.obs_max_range, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, avoid_action: torch.Tensor):
        cfg = self.cfg
        state = self.robot.state

        # 1. Correction → final velocity command
        t = torch.tanh(avoid_action)
        dvx = cfg.delta_vx_max * t[:, 0]
        dvy = cfg.delta_vy_max * t[:, 1]
        dvyaw = cfg.delta_vyaw_max * t[:, 2]
        self.correction = torch.stack([dvx, dvy, dvyaw], dim=-1)
        self.commands[:, 0] = torch.clamp(self.base_command[0] + dvx, *cfg.vx_clamp)
        self.commands[:, 1] = torch.clamp(self.base_command[1] + dvy, *cfg.vy_clamp)
        self.commands[:, 2] = torch.clamp(self.base_command[2] + dvyaw, *cfg.vyaw_clamp)

        # 2. Frozen locomotion skill → joint targets
        self.locomotion.command[:] = self.commands
        targets = self.locomotion.update(state, cfg.dt)
        self.robot.set_joint_targets(targets)
        self.scene.step()

        # 3. State + sensors
        self.episode_length_buf += 1
        self.robot.refresh()
        self._update_foot_contacts()
        self.lidar.tick()
        sectors = self.lidar.read()
        self._min_dist = sectors.min(dim=1).values

        # 4. Termination
        self.reset_buf = self.mark_time_outs()
        self.reset_buf |= fall_termination(
            state, cfg.termination_pitch, cfg.termination_roll,
            cfg.termination_height)
        if self._arena_term_dist is not None:
            self.reset_buf |= state.base_pos[:, 0].abs() > self._arena_term_dist
            self.reset_buf |= state.base_pos[:, 1].abs() > self._arena_term_dist
        if self.lidar.has_scan:
            self.reset_buf |= self._min_dist < cfg.d_collision

        # Reset BEFORE rewards/obs (legged-gym order).
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # 5. Rewards (registry applies scales × dt)
        self.compute_rewards()
        self.last_correction = self.correction.detach()

        # 6. Observation — re-read the scan: reset_idx cleared it for the
        #    envs that just respawned, so this differs from `sectors`.
        self.obs_buf = self._observe()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs and return a fresh (max-range) lidar observation."""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.n_envs, device=self.device))
        self.robot.refresh()
        self._update_foot_contacts()
        self.obs_buf = self._observe()
        return self.obs_buf, None

    def reset_idx(self, envs_idx: torch.Tensor):
        if len(envs_idx) == 0:
            return
        self.robot.reset_idx(envs_idx)
        self.locomotion.reset_idx(envs_idx)
        self.lidar.reset_idx(envs_idx)
        self._min_dist[envs_idx] = self.cfg.lidar_model.max_range
        if self.arena is not None:
            self.arena.randomise(envs_idx, self.device)

        self.last_correction[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.log_episode_sums(envs_idx, self.episode_length_s)

    # ------------------------------------------------------------------
    # Reward terms
    # ------------------------------------------------------------------

    def _reward_survival(self):
        """Constant 1 per step: makes early termination costly."""
        return torch.ones(self.n_envs, device=self.device)

    def _reward_avoidance(self):
        """
        4-zone proximity penalty (negative), from the lidar script. Zones
        on the closest return d: anticipate (linear, gentle), caution
        (linear), danger (quadratic), collision (flat 2.0).
        """
        cfg = self.cfg
        d = torch.clamp(self._min_dist, 0.0, cfg.obs_max_range)
        zero = torch.zeros_like(d)
        anticipate = torch.where(
            (d >= cfg.d_caution) & (d < cfg.d_anticipate),
            (cfg.d_anticipate - d) * 0.05, zero)
        caution = torch.where(
            (d >= cfg.d_danger) & (d < cfg.d_caution),
            (cfg.d_caution - d) * 0.5, zero)
        danger = torch.where(
            (d >= cfg.d_collision) & (d < cfg.d_danger),
            torch.square(cfg.d_danger - d) * 3.0, zero)
        collision = torch.where(
            d < cfg.d_collision, torch.full_like(d, 2.0), zero)
        return -(anticipate + caution + danger + collision)

    def _reward_smoothness(self):
        """Squared correction rate — scale is negative (REASAN-style)."""
        return torch.sum(
            torch.square(self.correction - self.last_correction), dim=1)

    def _reward_command_tracking(self):
        """
        Rewards zero correction when clear of obstacles, gated by clearance
        so it never fights the avoidance penalty when a turn is needed.
        """
        cfg = self.cfg
        safety_margin = torch.clamp(
            (self._min_dist - cfg.d_caution) / (cfg.d_anticipate - cfg.d_caution),
            0.0, 1.0)
        correction_mag = torch.sum(torch.square(self.correction), dim=1)
        return safety_margin * torch.exp(-correction_mag * 2.0)
