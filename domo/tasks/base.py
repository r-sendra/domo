"""
VecTask: the RL-facing environment contract.

A task composes a scene, one or more robots (with their sensors/actuators)
and reward logic. It exposes a vectorised gym-like API operating purely on
torch tensors; it knows nothing about the physics engine (only `domo.sim`
interfaces) and nothing about the learning algorithm — `domo.rl` is one
possible consumer, an LLM-driven skill executor is another.

Place in the architecture::

    domo.sim (engine handles) → domo.robot (sensors/actuators/state)
        → domo.tasks (THIS: reward-bearing environments, VecTask API)
            → domo.rl (PPO) / domo.eureka (reward injection) / examples

Step API (legged-gym style 5-tuple)::

    obs, privileged_obs, reward, reset_flags, extras = task.step(actions)

    obs            [N, num_obs]   float32, already normalised for the policy
    privileged_obs None           (reserved for asymmetric actor-critic)
    reward         [N]            float32, sum of scaled reward terms
    reset_flags    [N]            bool, True where the env was reset THIS step
    extras         dict           "time_outs" [N] float 0/1 (truncation, not
                                  failure — trainers may bootstrap on it),
                                  "episode" {rew_<term>: mean per-second}
                                  refreshed whenever envs finish.

Reward terms live in a registry keyed by name: a config's `reward_scales`
dict maps `<name>` → weight and the task must define `_reward_<name>()`.
Eureka bypasses the registry through `set_reward_override`, but the fixed
`compute_success` metric is never overridable — that separation is what
makes generated rewards rankable against a signal they cannot game.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

__all__ = ["RewardFn", "VecTask", "rand_uniform"]

# Injected reward (Eureka / M2): fn(task) → (reward [N], components {name: [N]}).
RewardFn = Callable[["VecTask"], tuple[torch.Tensor, dict[str, torch.Tensor]]]


def rand_uniform(lower: float, upper: float, shape, device) -> torch.Tensor:
    """Uniform samples in [lower, upper) with the given shape, on device."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class VecTask(ABC):
    """
    Vectorised task environment: N independent envs stepped in lockstep.

    Subclasses build the scene/robot in `__init__`, then implement `step`,
    `reset` and `reset_idx`. The base class owns the shared buffers, the
    reward registry and the episode bookkeeping helpers.

    Attributes:
        n_envs / num_envs: number of parallel envs (`num_envs` is the alias
            external trainers such as rsl_rl expect).
        num_obs, num_actions: observation / action dimensions.
        dt: control period [s]; reward scales are multiplied by it.
        max_episode_length: episode length in control steps.
        obs_buf [N, num_obs], rew_buf [N], reset_buf [N] bool,
        episode_length_buf [N] int32: the step-API buffers.
        extras: dict returned by `step` (see module docstring).
        reward_scales: name → weight (already × dt when registered so).
        episode_sums: name → running per-env sum of each scaled term.
        reward_components: last components dict produced by an injected
            reward (empty when the registry is in use).
    """

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
        self.extras: dict = {}

        self._reward_functions: dict[str, Callable[[], torch.Tensor]] = {}
        self.reward_scales: dict[str, float] = {}
        self.episode_sums: dict[str, torch.Tensor] = {}

        # Injected reward (Eureka / M2): fn(task) → (reward [N], components).
        self._reward_override: RewardFn | None = None
        self.reward_components: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Reward registry
    # ------------------------------------------------------------------

    def register_rewards(self, scales: dict[str, float],
                         scale_by_dt: bool = True) -> None:
        """
        Bind reward terms by name: each name must have a `_reward_<name>`
        method returning a per-env tensor. Scales are multiplied by dt
        (legged-gym convention) so weights are timestep-independent.

        Args:
            scales: term name → weight. A zero weight still registers the
                term (it is logged, just contributes nothing).
            scale_by_dt: multiply each weight by `self.dt`.

        Raises:
            AttributeError: if a name has no `_reward_<name>` method.
        """
        for name, scale in scales.items():
            self._reward_functions[name] = getattr(self, f"_reward_{name}")
            self.reward_scales[name] = scale * self.dt if scale_by_dt else scale
            self.episode_sums[name] = torch.zeros(
                (self.n_envs,), device=self.device, dtype=torch.float32)

    def set_reward_override(self, fn: RewardFn | None) -> None:
        """
        Replace the registered reward terms with an injected function
        (typically LLM-generated, see domo.eureka)::

            fn(task) → (reward [N], components: dict[str, Tensor[N]])

        The components feed reward-reflection; the fixed success metric
        (compute_success) is never affected by the override. Pass None to
        restore the registry.
        """
        self._reward_override = fn

    def compute_rewards(self) -> torch.Tensor:
        """
        Evaluate the reward into `rew_buf` and accumulate per-term episode
        sums. Registry terms are evaluated in registration order and scaled;
        an injected reward is used verbatim (no dt scaling) and its
        components are accumulated under their own names.

        Returns:
            rew_buf [N] (the same tensor, for convenience).
        """
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

        Raises:
            NotImplementedError: tasks without a fixed metric (the
                locomotion tasks) cannot be Eureka targets.
        """
        raise NotImplementedError(
            f"{type(self).__name__} defines no success metric")

    def log_episode_sums(self, envs_idx: torch.Tensor, episode_length_s: float) -> None:
        """
        Report per-term mean rewards for finished envs and zero their sums.

        Writes `extras["episode"]["rew_<name>"]` = mean over `envs_idx` of
        the accumulated term, divided by the nominal episode length in
        seconds (legged-gym convention: reward per second, comparable
        across episode lengths).
        """
        self.extras["episode"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(sums[envs_idx]).item() / episode_length_s)
            sums[envs_idx] = 0.0

    def mark_time_outs(self) -> torch.Tensor:
        """
        Flag envs whose episode budget is exhausted.

        Sets `extras["time_outs"]` to a float [N] 0/1 tensor (trainers that
        bootstrap on truncation read it) and returns the bool mask so the
        caller can OR it into `reset_buf`.
        """
        timeout = self.episode_length_buf > self.max_episode_length
        self.extras["time_outs"] = timeout.float()
        return timeout

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, actions: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor,
                        torch.Tensor, dict]:
        """Apply `actions` [N, num_actions] for one control step (5-tuple)."""

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reset every env; returns (obs [N, num_obs], privileged_obs)."""

    @abstractmethod
    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset the envs in `envs_idx` (int64 [K]); no-op when empty."""

    def get_observations(self) -> torch.Tensor:
        """Last computed observation [N, num_obs]."""
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor | None:
        """Asymmetric-critic observations; None for every current task."""
        return None
