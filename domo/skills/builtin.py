"""
Cards + library assembly for the skills that exist today.

`make_go2_library(walk_policy, avoid_policy, lidar)` returns a SkillLibrary
with stand / walk / avoid registered — the current DOMO repertoire. As new
skills are trained (M3) they are appended with `library.register(card,
factory)`; the LLM reads `library.describe()` and writes programs against
whatever the robot knows at that moment in its development.
"""

from __future__ import annotations

from domo.control.skill import (
    CPGLocomotionSkill,
    LidarAvoidanceSkill,
    StandSkill,
    TrajectoryTrackingSkill,
)
from domo.control.slam import SlamSkill

from .card import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard
from .library import SkillLibrary

__all__ = [
    "AVOID_CARD",
    "BACKWARD_CARD",
    "FORWARD_CARD",
    "GOTO_CARD",
    "SLAM_CARD",
    "STAND_CARD",
    "WALK_CARD",
    "make_go2_library",
]


STAND_CARD = SkillCard(
    name="stand",
    description=("Hold the default standing pose, motionless. The safest "
                 "skill: use as a terminal state, a recovery fallback, or "
                 "while waiting."),
    interface=MOTOR,
    preconditions=["robot roughly upright"],
    effects=["robot stationary in nominal stance"],
    success_when=[],                      # bound with .for()/.until()
    fail_when=["fallen(0.15)"],
    safety_notes=["cannot recover from a fall — it only holds posture"],
)

WALK_CARD = SkillCard(
    name="walk",
    description=("Omnidirectional velocity-tracking locomotion (learned "
                 "CPG policy). Tracks a body-frame velocity command; walks "
                 "on flat/indoor ground. Blind — it does NOT perceive "
                 "obstacles; layer 'avoid @' on top in cluttered space."),
    interface=MOTOR,
    accepts="velocity",
    params=[
        ParamSpec("vx", "forward velocity", default=0.0, range=(-1.0, 2.0), unit="m/s"),
        ParamSpec("vy", "lateral velocity (positive = left)", default=0.0,
                  range=(-0.5, 0.5), unit="m/s"),
        ParamSpec("vyaw", "yaw rate (positive = turn left)", default=0.0,
                  range=(-1.5, 1.5), unit="rad/s"),
    ],
    preconditions=["standing on ground", "trained checkpoint loaded"],
    effects=["robot moves at the commanded velocity"],
    success_when=[],                      # tracks forever; bound its duration
    fail_when=["tipped(0.9)", "fallen(0.18)"],
    constraints={"vx": (-1.0, 2.0), "vy": (-0.5, 0.5), "vyaw": (-1.5, 1.5)},
    safety_notes=["blind to obstacles", "commands clamped to constraints"],
)

AVOID_CARD = SkillCard(
    name="avoid",
    description=("Obstacle avoidance (learned, lidar-based). Emits velocity "
                 "corrections (Δvx, Δvy, Δvyaw) that steer around obstacles "
                 "while preserving the base command when the path is clear. "
                 "Runs ON TOP of a velocity-tracking motor skill: "
                 "'avoid @ walk(vx=...)'."),
    interface=CMD_VELOCITY,
    preconditions=["lidar available", "layered on a velocity motor skill"],
    effects=["combined command steers away from obstacles"],
    fail_when=["blocked(0.22)"],          # closer than this ⇒ avoidance failed
    safety_notes=["corrections bounded to ±(0.8, 0.5, 1.5)",
                  "final command still clamped by the base skill's limits"],
)


_NAV_EFFECT = ("robot arrives at the goal having held a straight path — "
               "lateral drift from the intended line is actively nulled")
_NAV_SAFETY = ["authors the full velocity command (overrides the base "
               "command); final command still clamped by the base skill's "
               "limits", "reads base pose each tick (sim truth / odometry / "
               "mocap)"]

FORWARD_CARD = SkillCard(
    name="forward",
    description=("Walk straight forward a fixed distance along the current "
                 "heading, holding the line: lateral deviation is corrected "
                 "every tick so the path stays straight. Runs ON TOP of a "
                 "velocity motor skill: 'forward(distance=2) @ walk'."),
    interface=CMD_VELOCITY,
    params=[
        ParamSpec("distance", "metres to travel forward", default=1.0,
                  range=(0.0, 20.0), unit="m"),
        ParamSpec("speed", "cruise speed", default=0.5, range=(0.05, 1.0),
                  unit="m/s"),
    ],
    preconditions=["layered on a velocity motor skill", "roughly upright"],
    effects=[_NAV_EFFECT],
    success_when=["reached the target within tolerance"],
    safety_notes=_NAV_SAFETY,
)

BACKWARD_CARD = SkillCard(
    name="backward",
    description=("Walk straight backward a fixed distance (robot keeps facing "
                 "forward and reverses), holding the line. Runs ON TOP of a "
                 "velocity motor skill: 'backward(distance=1) @ walk'."),
    interface=CMD_VELOCITY,
    params=[
        ParamSpec("distance", "metres to travel backward", default=1.0,
                  range=(0.0, 20.0), unit="m"),
        ParamSpec("speed", "cruise speed", default=0.5, range=(0.05, 1.0),
                  unit="m/s"),
    ],
    preconditions=["layered on a velocity motor skill", "roughly upright"],
    effects=[_NAV_EFFECT],
    success_when=["reached the target within tolerance"],
    safety_notes=_NAV_SAFETY,
)

GOTO_CARD = SkillCard(
    name="goto",
    description=("Drive to an absolute planar waypoint (x, y) in the world "
                 "frame along a straight line from the current pose, holding "
                 "the line. Chain with '>>' to follow a route. Runs ON TOP of "
                 "a velocity motor skill: 'goto(x=2, y=1) @ walk'."),
    interface=CMD_VELOCITY,
    params=[
        ParamSpec("x", "target x (world frame)", default=0.0,
                  range=(-50.0, 50.0), unit="m"),
        ParamSpec("y", "target y (world frame)", default=0.0,
                  range=(-50.0, 50.0), unit="m"),
        ParamSpec("speed", "cruise speed", default=0.5, range=(0.05, 1.0),
                  unit="m/s"),
    ],
    preconditions=["layered on a velocity motor skill", "roughly upright"],
    effects=[_NAV_EFFECT],
    success_when=["reached the waypoint within tolerance"],
    safety_notes=_NAV_SAFETY,
)

SLAM_CARD = SkillCard(
    name="slam",
    description=("Passive SLAM: builds an occupancy map of the surroundings "
                 "from the lidar while the robot drives, localising against "
                 "its pose estimate. It does NOT move or steer the robot and "
                 "adds NO velocity of its own — layer it ON TOP of a driving "
                 "stack so it maps as the robot goes, e.g. "
                 "'slam @ avoid @ goto(x=8, y=5) @ walk'. Use it whenever the "
                 "mission needs a map of explored space; it never changes "
                 "where the robot goes."),
    interface=CMD_VELOCITY,
    preconditions=["lidar available",
                   "layered on a velocity motor skill (directly or via other "
                   "command skills)"],
    effects=["an occupancy grid of free / occupied / unknown space is "
             "maintained and refined as the robot explores"],
    success_when=[],           # observes; bound its duration below with the stack
    safety_notes=["contributes zero velocity — never affects control",
                  "map fidelity tracks the pose estimate feeding it"],
)


def make_go2_library(walk_policy, avoid_policy=None, lidar=None,
                     cpg=None, avoid_deltas=(0.8, 0.5, 1.5),
                     obs_max_range: float = 4.0) -> SkillLibrary:
    """
    Assemble the current Go2 repertoire.
    walk_policy / avoid_policy: policy callables (see
    examples/house_scene/common.load_locomotion_policy).
    lidar: sensor with read() → [N, n_sectors]; required for 'avoid' and
    the 'blocked'/'clear' conditions.
    cpg / avoid_deltas / obs_max_range: match the values the policies were
    trained with (defaults = the experiment-script values).
    """
    library = SkillLibrary()
    library.register(STAND_CARD, StandSkill)
    library.register(WALK_CARD, lambda: CPGLocomotionSkill(walk_policy, cpg=cpg))

    # Straight-line navigation command-skills (pose-feedback, no policy needed);
    # layered on the velocity motor skill, e.g. 'goto(x=2, y=1) @ walk'.
    library.register(FORWARD_CARD, lambda: TrajectoryTrackingSkill("forward"))
    library.register(BACKWARD_CARD, lambda: TrajectoryTrackingSkill("backward"))
    library.register(GOTO_CARD, lambda: TrajectoryTrackingSkill("goto"))

    if lidar is not None:
        def blocked(d: float = 0.25):
            def cond(state, t_s):
                return lidar.read().min(dim=1).values < d
            return cond

        def clear(d: float = 1.4):
            def cond(state, t_s):
                return lidar.read().min(dim=1).values > d
            return cond

        library.register_condition("blocked", blocked)
        library.register_condition("clear", clear)

        # SLAM needs only the lidar (no policy): a passive mapping layer.
        library.register(SLAM_CARD, lambda: SlamSkill(lidar))

        if avoid_policy is not None:
            library.register(
                AVOID_CARD,
                lambda: LidarAvoidanceSkill(avoid_policy, lidar,
                                            deltas=avoid_deltas,
                                            obs_max_range=obs_max_range))

    return library
