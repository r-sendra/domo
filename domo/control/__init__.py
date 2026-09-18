"""
domo.control — the engine-free control hierarchy.

    ControlLoop   (loop.py)        metronome: sim step / wall-clock pacing,
                                   M7 command_filter seat
    Controller    (controller.py)  orchestration: which skill, what params
    Skill         (skill.py)       primitives: RobotState → joint targets;
                                   CommandSkills drive a motor skill's command
    CPG / IK      (cpg.py, kinematics.py)  the locomotion machinery skills use
    SlamSkill     (slam.py)        passive mapping layer (command skill)
    PositionController (navigation.py)  legacy waypoint driver (evaluate_nav)

Pure torch throughout: nothing here imports a physics engine, so the same
objects run in any simulator and on the real robot.
"""

from .controller import Controller, SingleSkillController
from .cpg import (
    CPG_OBS_DIM,
    CPG_OBS_SCALES,
    CPGConfig,
    CPGLegController,
    CPGOscillators,
    build_cpg_observation,
)
from .kinematics import LegKinematics, leg_fk, leg_ik
from .loop import RealControlLoop, SimControlLoop
from .navigation import NavConfig, PositionController
from .skill import (
    CommandSkill,
    CPGLocomotionSkill,
    LearnedJointSkill,
    LidarAvoidanceSkill,
    NavGains,
    Skill,
    StandSkill,
    TrajectoryTrackingSkill,
    all_envs,
    planar_pose,
)
from .slam import SlamConfig, SlamSkill

__all__ = [
    "CPG_OBS_DIM",
    "CPG_OBS_SCALES",
    "CPGConfig",
    "CPGLegController",
    "CPGLocomotionSkill",
    "CPGOscillators",
    "CommandSkill",
    "Controller",
    "LearnedJointSkill",
    "LegKinematics",
    "LidarAvoidanceSkill",
    "NavConfig",
    "NavGains",
    "PositionController",
    "RealControlLoop",
    "SimControlLoop",
    "SingleSkillController",
    "Skill",
    "SlamConfig",
    "SlamSkill",
    "StandSkill",
    "TrajectoryTrackingSkill",
    "all_envs",
    "build_cpg_observation",
    "leg_fk",
    "leg_ik",
    "planar_pose",
]
