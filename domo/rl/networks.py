"""Policy/value networks for the RL layer. Torch only — no engine imports."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ActorCritic", "clean_state_dict"]


def clean_state_dict(sd: dict) -> dict:
    """Strip torch.compile / DataParallel key prefixes so checkpoints load anywhere."""
    return {k.replace("_orig_mod.", "").replace("module.", ""): v
            for k, v in sd.items()}


class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate actor and critic heads.

    Architecture (defaults reproduce the original locomotion network)::

      Trunk    : trunk_layers × [Linear → ELU]              (shared)
      Actor    : Linear(head_hidden) → ELU → Linear(act_dim) (Gaussian mean)
      Critic   : Linear(head_hidden) → ELU → Linear(1)       (state value)
      log_std  : learned parameter, not input-dependent

    The avoidance network from the experiment scripts is the same family:
    ActorCritic(36, 3, hidden=128, trunk_layers=3, head_hidden=64).

    Orthogonal init with small gain on actor output for stable early training.

    State-dict keys (the checkpoint contract): `trunk.{2i}.weight/bias`,
    `actor_head.{0,2}.*`, `critic_head.{0,2}.*`, `log_std`.

    Args:
        obs_dim, act_dim: input / action dimensions.
        hidden: trunk width.
        trunk_layers: number of Linear→ELU trunk blocks.
        head_hidden: head width; None (or 0) → `hidden`.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512,
                 trunk_layers: int = 2, head_hidden: int | None = None):
        super().__init__()
        head_hidden = head_hidden or hidden

        trunk = []
        in_dim = obs_dim
        for _ in range(trunk_layers):
            trunk += [nn.Linear(in_dim, hidden), nn.ELU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*trunk)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ELU(),
            nn.Linear(head_hidden, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, head_hidden), nn.ELU(),
            nn.Linear(head_hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.00)

    @classmethod
    def from_state_dict(cls, sd: dict) -> ActorCritic:
        """
        Rebuild the network purely from checkpoint weight shapes (tolerates
        old script checkpoints and any config drift). Accepts a raw
        state_dict; keys are cleaned of compile/DataParallel prefixes.

        Raises:
            KeyError: if the state dict is not from this architecture.
        """
        sd = clean_state_dict(sd)
        obs_dim = sd["trunk.0.weight"].shape[1]
        act_dim = sd["log_std"].shape[0]
        hidden = sd["trunk.0.weight"].shape[0]
        trunk_layers = len([k for k in sd if k.startswith("trunk.") and k.endswith(".weight")])
        head_hidden = sd["actor_head.0.weight"].shape[0]
        net = cls(obs_dim, act_dim, hidden,
                  trunk_layers=trunk_layers, head_hidden=head_hidden)
        net.load_state_dict(sd)
        return net

    def forward(self, obs: torch.Tensor):
        """obs [N, obs_dim] → (mean [N, act_dim], std [N, act_dim], value [N])."""
        h = self.trunk(obs)
        mean = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample (or take the mean) action.

        Returns:
            (action [N, act_dim], log_prob [N], value [N]). Sampling uses
            `rsample` so gradients could flow through the action if needed.
        """
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """State value [N] (critic only)."""
        h = self.trunk(obs)
        return self.critic_head(h).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Evaluate stored actions for the PPO update.

        Returns:
            (log_prob [N], entropy [N], value [N]).
        """
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
