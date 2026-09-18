"""
RL layer: PPO over the `domo.tasks.VecTask` step API.

Pure torch — this package imports no physics engine, so networks and
checkpoints can be loaded on machines without a simulator (deployment,
evaluation, tests). It is one optional consumer of a task; nothing in
`domo.tasks` depends on it.

    ActorCritic       shared-trunk Gaussian policy + value head
    RolloutBuffer     T×N rollout storage with GAE-λ
    PPOConfig / PPOTrainer   the training loop, logging and checkpoints
    clean_state_dict  strip compile/DataParallel prefixes from weights
"""

from .buffer import RolloutBuffer
from .networks import ActorCritic, clean_state_dict
from .ppo import PPOConfig, PPOTrainer

__all__ = [
    "ActorCritic",
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "clean_state_dict",
]
