"""
VecTask: the RL-facing environment contract.

A task composes a scene, one or more robots (with their sensors/actuators)
and reward logic. It exposes a vectorised gym-like API operating purely on
torch tensors; it knows nothing about the physics engine (only `domo.sim`
interfaces) and nothing about the learning algorithm — `domo.rl` is one
possible consumer, an LLM-driven skill executor is another.

Step API (legged-gym style 5-tuple):
    obs, privileged_obs, reward, reset_flags, extras = task.step(actions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch

__all__ = ["VecTask", "rand_uniform"]


def rand_uniform(lower: float, upper: float, shape, device) -> torch.Tensor:
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class VecTask(ABC):
    """Vectorised task environment."""

    def __init__(self, n_envs: int, num_obs: int, num_actions: int,
                 device: torch.device, dt: float, max_episode_length: int):
        self.n_envs = n_envs
        self.num_envs = n_envs          # alias used by external trainers
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.device = device
        self.dt = dt
        self.max_episode_length = max_episode_length

        f, i = torch.float32, torch.int32
        self.obs_buf = torch.zeros((n_envs, num_obs), device=device, dtype=f)
        self.rew_buf = torch.zeros((n_envs,), device=device, dtype=f)
        self.reset_buf = torch.ones((n_envs,), device=device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((n_envs,), device=device, dtype=i)
        self.extras: Dict = {}

        self._reward_functions: Dict[str, callable] = {}
        self.reward_scales: Dict[str, float] = {}
        self.episode_sums: Dict[str, torch.Tensor] = {}

        # Injected reward (Eureka / M2): fn(task) → (reward [N], components).
        self._reward_override = None
        self.reward_components: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Reward registry
    # ------------------------------------------------------------------

    def register_rewards(self, scales: Dict[str, float],
                         scale_by_dt: bool = True) -> None:
        """
        Bind reward terms by name: each name must have a `_reward_<name>`
        method returning a per-env tensor. Scales are multiplied by dt
        (legged-gym convention) so weights are timestep-independent.
        """
        for name, scale in scales.items():
            self._reward_functions[name] = getattr(self, f"_reward_{name}")
            self.reward_scales[name] = scale * self.dt if scale_by_dt else scale
            self.episode_sums[name] = torch.zeros(
                (self.n_envs,), device=self.device, dtype=torch.float32)

    def set_reward_override(self, fn) -> None:
        """
        Replace the registered reward terms with an injected function
        (typically LLM-generated, see domo.eureka):
            fn(task) → (reward [N], components: dict[str, Tensor[N]])
        The components feed reward-reflection; the fixed success metric
        (compute_success) is never affected by the override.
        """
        self._reward_override = fn

    def compute_rewards(self) -> torch.Tensor:
        """Evaluate reward into rew_buf; accumulate per-term episode sums."""
        if self._reward_override is not None:
            rew, components = self._reward_override(self)
            self.rew_buf[:] = rew
            self.reward_components = components
            for name, value in components.items():
                if name not in self.episode_sums:
                    self.episode_sums[name] = torch.zeros(
                        (self.n_envs,), device=self.device, dtype=torch.float32)
                self.episode_sums[name] += value.detach()
            return self.rew_buf

        self.rew_buf[:] = 0.0
        for name, fn in self._reward_functions.items():
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        return self.rew_buf

    # ------------------------------------------------------------------
    # Success metric — fixed, reward-independent (Eureka's fitness signal)
    # ------------------------------------------------------------------

    def compute_success(self) -> torch.Tensor:
        """
        Bool [N]: whether each env currently satisfies the task's success
        condition. Deliberately separate from rewards so that generated
        rewards are ranked against a metric they cannot game.
        """
        raise NotImplementedError(
            f"{type(self).__name__} defines no success metric")

    def log_episode_sums(self, envs_idx: torch.Tensor, episode_length_s: float) -> None:
        """Report per-term mean rewards for finished envs and zero their sums."""
        self.extras["episode"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(sums[envs_idx]).item() / episode_length_s)
            sums[envs_idx] = 0.0

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, actions: torch.Tensor
             ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor,
                        torch.Tensor, Dict]: ...

    @abstractmethod
    def reset(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...

    @abstractmethod
    def reset_idx(self, envs_idx: torch.Tensor) -> None: ...

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return None
