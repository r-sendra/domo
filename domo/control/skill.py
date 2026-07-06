"""
Skill: the lowest programmable layer of the control hierarchy.

A Skill maps RobotState → joint targets at control rate (50 Hz). Skills are
the *primitives* of the architecture — learned policies, hand-crafted
behaviours, reflexes — and every entry in the M5 skill library implements
this interface. Layers above:

    LLM / planner (M1)   writes CONTROLLER code            (~0.01 Hz)
    Controller           selects skills + sets parameters  (1–10 Hz)
    Skill  ← this        state → joint targets             (50 Hz)
    actuators / safety   PD tracking, M7 command filter    (50–1000 Hz)

The uniform interface (setup / reset_idx / update, plus per-skill parameter
attributes such as `command`) is what makes skills composable: a Controller
can start, stop, parameterise and monitor any skill without knowing what is
inside it. Skills are pure torch — the same object runs in any simulator
and on the real robot.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch

from .cpg import CPGConfig, CPGLegController, build_cpg_observation, CPG_OBS_SCALES
from .kinematics import LegKinematics

__all__ = ["Skill", "StandSkill", "CPGLocomotionSkill",
           "CommandSkill", "LidarAvoidanceSkill", "LearnedJointSkill"]


class Skill(ABC):
    """One motor primitive. Vectorised: all tensors are [n_envs, ...]."""

    name: str = "skill"

    def setup(self, robot) -> None:
        """Bind to a robot (after robot.bind()); allocate per-env state."""
        self.robot = robot

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset internal state for the given envs (also used on activation)."""

    @abstractmethod
    def update(self, state, dt: float) -> torch.Tensor:
        """One control cycle: RobotState → joint position targets [N, D]."""


class StandSkill(Skill):
    """Hold the spec's default stance. The trivial (and safest) primitive."""

    name = "stand"

    def setup(self, robot) -> None:
        super().setup(robot)
        self._targets = robot.default_dof_pos.unsqueeze(0).repeat(
            robot.n_envs, 1).clone()

    def update(self, state, dt: float) -> torch.Tensor:
        return self._targets


class CPGLocomotionSkill(Skill):
    """
    Velocity-tracking locomotion: frozen CPG policy + oscillators + leg IK.

    This is the frozen-locomotion stack of the avoidance experiments as a
    reusable primitive. Parameterised via `command` ([N, 3]: vx, vy, vyaw in
    body frame) — the vocabulary Controllers (and higher skills like
    avoidance) use to drive it.

    policy_fn: callable(obs [N, 76]) → raw CPG action [N, 12], evaluated
    under no_grad. Inject a trained ActorCritic wrapper (see
    examples/house_scene/common.load_locomotion_policy) — or a zero lambda
    for a standing robot.
    """

    name = "cpg_locomotion"

    def __init__(self, policy_fn: Callable[[torch.Tensor], torch.Tensor],
                 cpg: Optional[CPGConfig] = None):
        self.policy_fn = policy_fn
        self.cpg_cfg = cpg or CPGConfig()

    def setup(self, robot) -> None:
        super().setup(robot)
        n, device = robot.n_envs, robot.device
        self.kinematics = LegKinematics(robot.spec.geometry, device)
        self.leg_controller = CPGLegController(self.cpg_cfg, self.kinematics,
                                               n, device)
        self.command = torch.zeros((n, 3), device=device)
        self._commands_scale = torch.tensor(
            [CPG_OBS_SCALES["lin_vel"], CPG_OBS_SCALES["lin_vel"],
             CPG_OBS_SCALES["ang_vel"]], device=device)
        self._last_action = torch.zeros((n, 12), device=device)

    @property
    def oscillators(self):
        return self.leg_controller.oscillators

    def stance_mask(self) -> torch.Tensor:
        """[N, 4] phase-proxy foot contacts (1 = stance)."""
        return self.oscillators.stance_mask()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self.leg_controller.reset_idx(envs_idx)
        self._last_action[envs_idx] = 0.0
        self.command[envs_idx] = 0.0

    def update(self, state, dt: float) -> torch.Tensor:
        obs = build_cpg_observation(
            state, self.command, self._commands_scale,
            self.robot.default_dof_pos, self._last_action,
            state.foot_contacts, self.oscillators)
        with torch.no_grad():
            action = self.policy_fn(obs)
        targets = self.leg_controller.joint_targets(action, dt)
        self._last_action[:] = action
        return targets


class LearnedJointSkill(Skill):
    """
    Generic wrapper turning an Eureka/RL-trained joint-space policy into a
    library skill: obs_builder(state, default_dof_pos, last_action) → obs,
    policy → action, targets = default + action_scale × action.
    This is how M3 outputs (e.g. the get-up policy) enter the M5 library.
    """

    def __init__(self, policy_fn: Callable[[torch.Tensor], torch.Tensor],
                 obs_builder: Callable, action_scale: float = 0.35,
                 name: str = "learned"):
        self.policy_fn = policy_fn
        self.obs_builder = obs_builder
        self.action_scale = action_scale
        self.name = name

    def setup(self, robot) -> None:
        super().setup(robot)
        self._last_action = torch.zeros(
            (robot.n_envs, robot.spec.num_dofs), device=robot.device)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self._last_action[envs_idx] = 0.0

    def update(self, state, dt: float) -> torch.Tensor:
        obs = self.obs_builder(state, self.robot.default_dof_pos,
                               self._last_action)
        with torch.no_grad():
            action = self.policy_fn(obs)
        self._last_action[:] = action
        return self.robot.default_dof_pos + self.action_scale * action


# ---------------------------------------------------------------------------
# Command skills — the layer that runs "on top of" motor skills
# ---------------------------------------------------------------------------

class CommandSkill(Skill):
    """
    A skill that emits COMMANDS for another skill instead of joint targets:
    the upper half of a layered composition (`avoid @ walk`). Its output is
    added to (additive=True) or replaces the base skill's command channel.
    """

    channel: str = "velocity"     # must match the base skill's accepted channel
    additive: bool = True

    @abstractmethod
    def update_command(self, state, dt: float) -> torch.Tensor:
        """One control cycle → command-channel output (e.g. [N, 3] Δv)."""

    def update(self, state, dt: float) -> torch.Tensor:
        raise TypeError(
            f"{type(self).__name__} emits '{self.channel}' commands, not "
            f"joint targets — layer it on a motor skill (e.g. 'this @ walk')")


class LidarAvoidanceSkill(CommandSkill):
    """
    Obstacle avoidance as a velocity-correction skill: lidar sectors →
    (Δvx, Δvy, Δvyaw), from a trained avoidance policy. Layered on a
    velocity-tracking motor skill it reproduces the two-layer architecture
    of the avoidance experiments.
    """

    name = "lidar_avoidance"
    channel = "velocity"
    additive = True

    def __init__(self, policy_fn: Callable[[torch.Tensor], torch.Tensor],
                 lidar, deltas=(0.8, 0.5, 1.5), obs_max_range: float = 4.0):
        """
        lidar: object with read() → [N, n_sectors] distances (SimulatedLidar
        in sim, the driver-fed equivalent on the real robot).
        deltas: max |Δvx|, |Δvy|, |Δvyaw| the skill may command.
        """
        self.policy_fn = policy_fn
        self.lidar = lidar
        self.deltas = deltas
        self.obs_max_range = obs_max_range

    def setup(self, robot) -> None:
        super().setup(robot)
        self._delta = torch.tensor(self.deltas, device=robot.device)

    def min_distance(self) -> torch.Tensor:
        return self.lidar.read().min(dim=1).values

    def update_command(self, state, dt: float) -> torch.Tensor:
        obs = torch.clamp(self.lidar.read() / self.obs_max_range, 0.0, 1.0)
        with torch.no_grad():
            raw = self.policy_fn(obs)
        return torch.tanh(raw) * self._delta
