"""
Tasks: reward-bearing, vectorised environments over `domo.sim` / `domo.robot`.

Every task subclasses `VecTask` (see base.py for the step contract) and is
built from a dataclass config whose fields are the checkpoint / spec
contract. Tasks never import a physics engine directly.

Available tasks:
    Go2WalkTask     velocity tracking, joint-position actions (main.py)
    Go2CPGWalkTask  velocity tracking, CPG-modulating actions (76-dim obs)
    Go2AvoidTask    lidar obstacle avoidance over a frozen CPG policy
    Go2GetUpTask    recovery from a fallen pose; Eureka reward-injection target

To add a task: subclass VecTask, define OBS_DIM/ACT_DIM, a config dataclass,
`_reward_<name>` methods for every key of `reward_scales`, and implement
step/reset/reset_idx; then re-export it here.
"""

from .base import RewardFn, VecTask, rand_uniform
from .go2_avoid import (
    AVOID_ACT_DIM,
    AVOID_OBS_DIM,
    Go2AvoidConfig,
    Go2AvoidTask,
    ObstacleArenaConfig,
)
from .go2_cpg_walk import (
    CPG_ACT_DIM,
    CPG_OBS_DIM,
    Go2CPGWalkConfig,
    Go2CPGWalkTask,
    build_cpg_observation,
)
from .go2_getup import (
    GETUP_ACT_DIM,
    GETUP_OBS_DIM,
    Go2GetUpConfig,
    Go2GetUpTask,
    build_getup_observation,
)
from .go2_walk import Go2WalkConfig, Go2WalkTask

__all__ = [
    "AVOID_ACT_DIM",
    "AVOID_OBS_DIM",
    "CPG_ACT_DIM",
    "CPG_OBS_DIM",
    "GETUP_ACT_DIM",
    "GETUP_OBS_DIM",
    "Go2AvoidConfig",
    "Go2AvoidTask",
    "Go2CPGWalkConfig",
    "Go2CPGWalkTask",
    "Go2GetUpConfig",
    "Go2GetUpTask",
    "Go2WalkConfig",
    "Go2WalkTask",
    "ObstacleArenaConfig",
    "RewardFn",
    "VecTask",
    "build_cpg_observation",
    "build_getup_observation",
    "rand_uniform",
]
