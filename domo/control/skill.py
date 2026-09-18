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
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .cpg import CPG_OBS_SCALES, CPGConfig, CPGLegController, build_cpg_observation
from .kinematics import LegKinematics

__all__ = [
    "CPGLocomotionSkill",
    "CommandSkill",
    "LearnedJointSkill",
    "LidarAvoidanceSkill",
    "NavGains",
    "Skill",
    "StandSkill",
    "TrajectoryTrackingSkill",
]


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
                 cpg: CPGConfig | None = None):
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
    additive: bool = True         # True: add to base command; False: author it

    def configure(self, **params) -> None:
        """
        Apply grammar parameters (`skill(a=..., b=...)`) to this instance at
        compile time. Grammar values are floats. Default: no parameters.
        """
        if params:
            raise TypeError(
                f"{type(self).__name__} takes no parameters (got {params})")

    def success_flags(self, state) -> torch.Tensor | None:
        """
        Optional per-env completion signal [N] (bool). When a layered command
        skill returns all-True, the hosting LayerNode reports SUCCESS — this is
        how a goal-directed command skill (e.g. navigation) terminates a
        composition. Default None = "never succeeds on its own".
        """
        return None

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


@dataclass
class NavGains:
    """P gains + limits for TrajectoryTrackingSkill (merged from NavConfig)."""
    kp_par: float = 1.0       # along-track: remaining distance → forward speed
    kp_perp: float = 1.2      # cross-track: lateral deviation → sideways speed
    kp_ang: float = 1.5       # heading error → yaw rate
    min_speed: float = 0.3    # along-track floor while en route: keeps the
                              # command above the locomotion policy's deadband
                              # so it never stalls short of the target
    max_vy: float = 0.4       # |lateral correction| cap (m/s)
    max_vyaw: float = 1.0     # |yaw rate| cap (rad/s)
    tol_pos: float = 0.2      # m — target reached


class TrajectoryTrackingSkill(CommandSkill):
    """
    Straight-line navigation as a velocity-command AUTHOR.

    Layered on a velocity-tracking motor skill ('forward @ walk'), it drives
    the robot to a goal while actively minimizing lateral deviation from the
    intended straight line between the start pose and the goal. It closes the
    loop on the base pose (RobotState) every tick, so tracking error is
    corrected instead of accumulating — that is what keeps a commanded
    "go forward 2 m" straight and stops odometry drift from smearing the path.

    Override command skill (additive=False): it emits the full body-frame
    (vx, vy, vyaw); the base motor skill still clamps it to its own limits.

    Modes (fixed by the factory, goal set via configure()/grammar params):
      "forward"  — travel `distance` m along the heading held at activation
      "backward" — travel `distance` m opposite that heading (robot keeps
                   facing forward and walks in reverse)
      "goto"     — travel to absolute planar target (x, y)

    Generalises domo.control.PositionController's P-control into a line-tracking
    (along-track + cross-track + heading) law, packaged as a library skill.
    """

    name = "navigate"
    channel = "velocity"
    additive = False

    def __init__(self, mode: str = "forward", distance: float = 1.0,
                 target=(0.0, 0.0), speed: float = 0.5,
                 cfg: NavGains | None = None):
        if mode not in ("forward", "backward", "goto"):
            raise ValueError(f"unknown nav mode '{mode}'")
        self.mode = mode
        self.distance = float(distance)
        self.target = (float(target[0]), float(target[1]))
        self.speed = float(speed)
        self.cfg = cfg or NavGains()

    def configure(self, **params) -> None:
        if "distance" in params:
            self.distance = float(params["distance"])
        if "speed" in params:
            self.speed = float(params["speed"])
        if "x" in params or "y" in params:
            self.target = (float(params.get("x", self.target[0])),
                           float(params.get("y", self.target[1])))

    def setup(self, robot) -> None:
        super().setup(robot)
        n, dev = robot.n_envs, robot.device
        self._start = torch.zeros(n, 2, device=dev)   # world start position
        self._u = torch.zeros(n, 2, device=dev)        # world unit path dir
        self._face = torch.zeros(n, device=dev)        # desired yaw to hold
        self._len = torch.zeros(n, device=dev)         # full path length
        self._need_init = torch.ones(n, dtype=torch.bool, device=dev)
        self._arrived = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        # Re-latch the trajectory from wherever the robot is on (re)activation.
        self._need_init[envs_idx] = True
        self._arrived[envs_idx] = False

    def success_flags(self, state) -> torch.Tensor:
        return self._arrived

    def _init_envs(self, state, mask: torch.Tensor) -> None:
        pos = state.base_pos[mask, :2]
        yaw = state.base_euler[mask, 2]
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        if self.mode == "goto":
            tgt = torch.tensor(self.target, device=pos.device,
                               dtype=pos.dtype).expand_as(pos)
            d = tgt - pos
            length = torch.linalg.norm(d, dim=1).clamp_min(1e-6)
            u = d / length.unsqueeze(1)
            face = torch.atan2(u[:, 1], u[:, 0])       # face toward the target
        else:
            sign = 1.0 if self.mode == "forward" else -1.0
            u = sign * heading
            face = yaw                                  # keep facing forward
            length = torch.full((int(mask.sum()),), self.distance,
                                device=pos.device, dtype=pos.dtype)
        self._start[mask] = pos
        self._u[mask] = u
        self._face[mask] = face
        self._len[mask] = length
        self._need_init[mask] = False
        self._arrived[mask] = False

    def update_command(self, state, dt: float) -> torch.Tensor:
        if bool(self._need_init.any()):
            self._init_envs(state, self._need_init)

        pos = state.base_pos[:, :2]
        yaw = state.base_euler[:, 2]
        rel = pos - self._start                         # [N, 2]
        perp = torch.stack([-self._u[:, 1], self._u[:, 0]], dim=1)
        along = (rel * self._u).sum(dim=1)              # progress along path
        cross = (rel * perp).sum(dim=1)                 # signed lateral offset
        remaining = self._len - along

        c = self.cfg
        # World-frame desired velocity: advance along the line, and push back
        # onto it in proportion to the perpendicular deviation. The along-track
        # speed is floored at min_speed (capped by the leg's cruise speed) so it
        # stays above the locomotion policy's deadband and never stalls short.
        min_v = min(self.speed, c.min_speed)
        v_par = torch.clamp(c.kp_par * remaining, min=min_v, max=self.speed)
        v_perp = torch.clamp(-c.kp_perp * cross, min=-c.max_vy, max=c.max_vy)
        v_world = v_par.unsqueeze(1) * self._u + v_perp.unsqueeze(1) * perp

        # Rotate world velocity into the body frame the motor skill expects.
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        vx = cos_y * v_world[:, 0] + sin_y * v_world[:, 1]
        vy = -sin_y * v_world[:, 0] + cos_y * v_world[:, 1]
        yaw_err = self._face - yaw
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
        vyaw = torch.clamp(c.kp_ang * yaw_err, min=-c.max_vyaw, max=c.max_vyaw)

        self._arrived |= remaining <= c.tol_pos
        cmd = torch.stack([vx, vy, vyaw], dim=1)
        cmd[self._arrived] = 0.0                         # hold once arrived
        return cmd
