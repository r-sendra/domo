"""Rollout storage with GAE-λ advantage computation."""

from __future__ import annotations

import torch

__all__ = ["RolloutBuffer"]


class RolloutBuffer:
    """
    Stores one rollout (T steps × N envs) and computes GAE advantages.
    All tensors are pre-allocated on device.

    Storage is split into two calls per timestep:
      store_step()    — BEFORE env.step(), with obs/action/logp/value
      store_outcome() — AFTER  env.step(), with reward/done
    """

    def __init__(self, rollout_steps: int, n_envs: int, obs_dim: int,
                 act_dim: int, device):
        self.T = rollout_steps
        self.N = n_envs
        self.device = device
        self.ptr = 0

        def buf(*shape):
            return torch.zeros(*shape, device=device)

        self.obs = buf(self.T, n_envs, obs_dim)
        self.actions = buf(self.T, n_envs, act_dim)
        self.log_probs = buf(self.T, n_envs)
        self.values = buf(self.T, n_envs)
        self.rewards = buf(self.T, n_envs)
        self.dones = buf(self.T, n_envs)
        self.advantages = buf(self.T, n_envs)
        self.returns = buf(self.T, n_envs)

    def store_step(self, obs, actions, log_probs, values):
        t = self.ptr
        self.obs[t] = obs.detach()
        self.actions[t] = actions.detach()
        self.log_probs[t] = log_probs.detach()
        self.values[t] = values.detach()

    def store_outcome(self, rewards, dones):
        t = self.ptr
        self.rewards[t] = rewards.detach().float()
        self.dones[t] = dones.detach().float()
        self.ptr += 1

    def compute_gae(self, last_value, gamma: float = 0.99, lam: float = 0.95):
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0

    def get_flat(self):
        """Flatten (T, N, ...) → (T*N, ...) for minibatch sampling."""
        T, N = self.T, self.N
        return (
            self.obs.view(T * N, -1),
            self.actions.view(T * N, -1),
            self.log_probs.view(T * N),
            self.advantages.view(T * N),
            self.returns.view(T * N),
            self.values.view(T * N),
        )
