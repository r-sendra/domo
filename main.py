import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from domo.robot.go2 import Go2WalkEnv


# ==========================================================================
# Actor-Critic Network
# ==========================================================================

class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate actor and critic heads.

    Architecture:
      Trunk    : Linear(obs) → ELU → Linear → ELU         (shared)
      Actor    : Linear → ELU → Linear(act_dim)            (Gaussian mean)
      Critic   : Linear → ELU → Linear(1)                  (state value)
      log_std  : learned parameter, not input-dependent

    Orthogonal init with small gain on actor output for stable early training.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden,  hidden), nn.ELU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.00)

    def forward(self, obs: torch.Tensor):
        h     = self.trunk(obs)
        mean  = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        std   = self.log_std.exp().expand_as(mean)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action, return (action, log_prob, value)."""
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        action   = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return critic value estimate without sampling an action."""
        h = self.trunk(obs)
        return self.critic_head(h).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate stored actions for PPO update."""
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy, value

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ==========================================================================
# Rollout Buffer
# ==========================================================================

class RolloutBuffer:
    """
    Stores one rollout (T steps × N envs) and computes GAE advantages.
    All tensors are pre-allocated on device for efficiency.

    Storage is split into two calls per timestep:
      store_step()    — called BEFORE env.step() with obs/action/logp/value
      store_outcome() — called AFTER  env.step() with reward/done
    """

    def __init__(
        self,
        rollout_steps: int,
        n_envs:        int,
        obs_dim:       int,
        act_dim:       int,
        device:        str,
    ):
        self.T      = rollout_steps
        self.N      = n_envs
        self.device = device
        self.ptr    = 0

        def buf(*shape):
            return torch.zeros(*shape, device=device)

        self.obs       = buf(self.T, n_envs, obs_dim)
        self.actions   = buf(self.T, n_envs, act_dim)
        self.log_probs = buf(self.T, n_envs)
        self.values    = buf(self.T, n_envs)
        self.rewards   = buf(self.T, n_envs)
        self.dones     = buf(self.T, n_envs)
        self.advantages = buf(self.T, n_envs)
        self.returns    = buf(self.T, n_envs)

    def store_step(
        self,
        obs:       torch.Tensor,
        actions:   torch.Tensor,
        log_probs: torch.Tensor,
        values:    torch.Tensor,
    ):
        """Store pre-step quantities. Called BEFORE env.step()."""
        t = self.ptr
        self.obs[t]       = obs.detach()
        self.actions[t]   = actions.detach()
        self.log_probs[t] = log_probs.detach()
        self.values[t]    = values.detach()

    def store_outcome(
        self,
        rewards: torch.Tensor,
        dones:   torch.Tensor,
    ):
        """Store post-step outcome. Called AFTER env.step()."""
        t = self.ptr
        self.rewards[t] = rewards.detach().float()
        self.dones[t]   = dones.detach().float()
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        lam:   float = 0.95,
    ):
        """Generalised Advantage Estimation (GAE-λ)."""
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask     = 1.0 - self.dones[t]
            delta    = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae      = delta + gamma * lam * mask * gae
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


# ==========================================================================
# PPO Trainer
# ==========================================================================

class PPOTrainer:

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = cfg["device"]

        self.env = Go2WalkEnv(
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
            device            = cfg["device"],
            terrain           = cfg["terrain"]
        )

        self.net = ActorCritic(
            obs_dim = Go2WalkEnv.OBS_DIM,
            act_dim = Go2WalkEnv.ACT_DIM,
            hidden  = cfg["hidden_size"],
        ).to(self.device)

        print(f"Network parameters: {self.net.num_parameters:,}")

        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr  = cfg["lr"],
            eps = 1e-5,
        )

        self.buf = RolloutBuffer(
            rollout_steps = cfg["rollout_steps"],
            n_envs        = cfg["n_envs"],
            obs_dim       = Go2WalkEnv.OBS_DIM,
            act_dim       = Go2WalkEnv.ACT_DIM,
            device        = self.device,
        )

        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()

        self.ep_returns = []
        self.ep_lengths = []
        self._env_ep_return = torch.zeros(cfg["n_envs"], device=self.device)
        self._env_ep_length = torch.zeros(cfg["n_envs"], device=self.device, dtype=torch.int32)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()

        print(f"\n{'='*55}")
        print(f"  Go2 Walking PPO Training")
        print(f"{'='*55}")
        print(f"  Envs          : {cfg['n_envs']}")
        print(f"  Total steps   : {cfg['total_steps']:,}")
        print(f"  Rollout steps : {cfg['rollout_steps']}")
        print(f"  Minibatch     : {cfg['minibatch_size']}")
        print(f"  Device        : {self.device}")
        print(f"  Run dir       : {cfg['run_dir']}")
        print(f"{'='*55}\n")

        steps_per_rollout = cfg["rollout_steps"] * cfg["n_envs"]
        n_updates = cfg["total_steps"] // steps_per_rollout

        for update in range(1, n_updates + 1):
            obs = self._collect_rollout(obs)
            metrics = self._ppo_update()
            self.global_step += steps_per_rollout

            if update % cfg["log_interval"] == 0:
                elapsed   = time.time() - self.start_time
                steps_sec = self.global_step / elapsed

                mean_ret = float(np.mean(self.ep_returns[-20:])) if self.ep_returns else 0.0
                mean_len = float(np.mean(self.ep_lengths[-20:])) if self.ep_lengths else 0.0

                print(
                    f"  step {self.global_step:>10,} | "
                    f"ret {mean_ret:>7.3f} | "
                    f"len {mean_len:>5.0f} | "
                    f"ploss {metrics['policy_loss']:>7.4f} | "
                    f"vloss {metrics['value_loss']:>7.4f} | "
                    f"clip {metrics['clip_frac']:>4.2f} | "
                    f"{steps_sec:>7,.0f} sps"
                )

                self.writer.add_scalar("train/mean_return",   mean_ret,               self.global_step)
                self.writer.add_scalar("train/mean_ep_len",   mean_len,               self.global_step)
                self.writer.add_scalar("loss/policy",         metrics["policy_loss"], self.global_step)
                self.writer.add_scalar("loss/value",          metrics["value_loss"],  self.global_step)
                self.writer.add_scalar("loss/entropy",        metrics["entropy"],     self.global_step)
                self.writer.add_scalar("train/clip_fraction", metrics["clip_frac"],   self.global_step)
                self.writer.add_scalar("train/steps_per_sec", steps_sec,              self.global_step)

            if update % cfg["save_interval"] == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.writer.close()
        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Collect rollout_steps steps. obs is a torch Tensor on device.
        The env now returns torch tensors directly — no numpy conversion.
        """
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg["rollout_steps"]):

                action, log_prob, value = self.net.get_action(obs)

                # Store BEFORE step
                self.buf.store_step(obs, action, log_prob, value)

                next_obs, _, reward, reset_buf, extras = self.env.step(action)

                # Store AFTER step
                self.buf.store_outcome(reward, reset_buf.float())

                # Track episode stats per env
                self._env_ep_return += reward
                self._env_ep_length += 1
                done_idx = reset_buf.nonzero(as_tuple=False).flatten()
                for idx in done_idx:
                    self.ep_returns.append(float(self._env_ep_return[idx]))
                    self.ep_lengths.append(int(self._env_ep_length[idx]))
                self._env_ep_return[done_idx] = 0.0
                self._env_ep_length[done_idx] = 0

                obs = next_obs

            last_value = self.net.get_value(obs)
            self.buf.compute_gae(
                last_value = last_value,
                gamma      = self.cfg["gamma"],
                lam        = self.cfg["lam"],
            )

        return obs

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        self.net.train()
        cfg = self.cfg

        obs_flat, act_flat, lp_flat, adv_flat, ret_flat, val_flat = self.buf.get_flat()
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        total_samples = obs_flat.shape[0]
        metrics = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "clip_frac":   [],
        }

        for _ in range(cfg["n_epochs"]):
            idx = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, cfg["minibatch_size"]):
                mb = idx[start : start + cfg["minibatch_size"]]
                mb_old_values = val_flat[mb]
                mb_ret        = ret_flat[mb]

                new_lp, entropy, value = self.net.evaluate(
                    obs_flat[mb], act_flat[mb]
                )

                ratio = (new_lp - lp_flat[mb]).exp()

                surr1 = ratio * adv_flat[mb]
                surr2 = ratio.clamp(
                    1 - cfg["clip_eps"], 1 + cfg["clip_eps"]
                ) * adv_flat[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = mb_old_values + (value - mb_old_values).clamp(
                    -cfg["clip_eps"], cfg["clip_eps"]
                )
                value_loss = torch.max(
                    (value - mb_ret).pow(2),
                    (value_pred_clipped - mb_ret).pow(2),
                ).mean()

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg["vf_coef"]  * value_loss
                    + cfg["ent_coef"] * entropy_loss
                )

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg["max_grad_norm"]
                )
                self.opt.step()

                clip_frac = (
                    (ratio - 1.0).abs() > cfg["clip_eps"]
                ).float().mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(-entropy_loss.item())
                metrics["clip_frac"].append(clip_frac.item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str = None):
        name = f"checkpoint_step_{self.global_step:09d}"
        if tag:
            name = f"checkpoint_{tag}"
        path = os.path.join(self.cfg["run_dir"], f"{name}.pt")
        torch.save({
            "step":        self.global_step,
            "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(),
            "config":      self.cfg,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-20:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-20:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] Saved {path}")

    @classmethod
    def load_checkpoint(cls, path: str) -> "PPOTrainer":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt    = torch.load(path, weights_only=False, map_location=map_location)
        trainer = cls(ckpt["config"])
        trainer.net.load_state_dict(ckpt["model_state"])
        trainer.opt.load_state_dict(ckpt["optim_state"])
        trainer.global_step = ckpt["step"]
        print(
            f"Resumed from step {trainer.global_step:,} "
            f"(mean return: {ckpt['metrics']['mean_return']:.3f})"
        )
        return trainer


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate(checkpoint_path: str, n_episodes: int = 10):
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
    cfg  = ckpt["config"]

    env = Go2WalkEnv(
        n_envs            = 1,
        headless          = False,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
        device            = map_location,
    )
    net = ActorCritic(Go2WalkEnv.OBS_DIM, Go2WalkEnv.ACT_DIM, cfg["hidden_size"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    net = net.to(map_location)

    returns, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = torch.zeros(1, dtype=torch.bool)
        ep_ret = 0.0
        ep_len = 0

        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=False)

            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done = reset_buf.bool()

        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"  Episode {ep+1:2d} | return {ep_ret:7.2f} | length {ep_len}")

    print(f"\nMean return : {np.mean(returns):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f}")


# ==========================================================================
# Entry point
# ==========================================================================

def get_config(args) -> dict:
    n_envs = args.n_envs

    total_buffer   = n_envs * args.rollout_steps
    n_minibatches  = 4
    minibatch_size = max(total_buffer // n_minibatches, 256)

    return dict(
        # Environment
        n_envs             = n_envs,
        dt                 = 0.02,
        max_episode_steps  = 1000,
        headless           = args.headless,
        terrain            = args.terrain,

        # Network
        hidden_size        = 512,

        # PPO
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = minibatch_size,
        n_epochs           = 5,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 3e-4,
        vf_coef            = 1.0,
        ent_coef           = 0.01,
        max_grad_norm      = 1.0,

        # Logging
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",        type=int,  default=4096)
    parser.add_argument("--total-steps",   type=int,  default=100_000_000)
    parser.add_argument("--rollout-steps", type=int,  default=24)
    parser.add_argument("--device",        type=str,  default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-dir",       type=str,  default="runs/go2_walk")
    parser.add_argument("--headless",      action="store_true", default=True)
    parser.add_argument("--resume",        type=str,  default=None)
    parser.add_argument("--eval",          type=str,  default=None)
    parser.add_argument("--terrain", type=str, default="flat",
                    choices=["flat", "rough"])
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    if args.resume:
        trainer = PPOTrainer.load_checkpoint(args.resume)
    else:
        cfg     = get_config(args)
        trainer = PPOTrainer(cfg)

    trainer.train()


if __name__ == "__main__":
    main()
