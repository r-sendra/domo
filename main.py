
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# FIX 3: Updated import to match renamed class Go2WalkEnv
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

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
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
        # Small init on output layers — keeps early actions near zero
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

    # FIX 6: Dedicated value-only method avoids sampling an action when
    # only the bootstrap value is needed (cleaner and slightly faster).
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

    FIX 4: Storage is now split into two calls per timestep:
      store_step()       — called BEFORE env.step() with obs/action/logp/value
      store_outcome()    — called AFTER  env.step() with reward/done
    This ensures each field is stored at the correct moment in time.
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
        obs:       np.ndarray,
        actions:   np.ndarray,
        log_probs: np.ndarray,
        values:    np.ndarray,
    ):
        """
        Store the pre-step quantities at the current buffer position.
        Called BEFORE env.step().
        """
        t = self.ptr
        self.obs[t]       = torch.as_tensor(obs,       device=self.device)
        self.actions[t]   = torch.as_tensor(actions,   device=self.device)
        self.log_probs[t] = torch.as_tensor(log_probs, device=self.device)
        self.values[t]    = torch.as_tensor(values,    device=self.device)

    def store_outcome(
        self,
        rewards: np.ndarray,
        dones:   np.ndarray,
    ):
        """
        Store the post-step outcome at the current buffer position,
        then advance the pointer.
        Called AFTER env.step().
        """
        t = self.ptr
        self.rewards[t] = torch.as_tensor(
            rewards.astype(np.float32), device=self.device)
        self.dones[t]   = torch.as_tensor(
            dones.astype(np.float32),   device=self.device)
        self.ptr += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        lam:   float = 0.95,
    ):
        """
        Generalised Advantage Estimation (GAE-λ).

        FIX 7 (documented): The env auto-resets done environments and
        returns the new episode's first observation. The done flag at
        step t is 1 for those envs, which zeroes the bootstrap via
        `mask = 1 - done[t]`. This is the standard approach for
        auto-resetting vectorised envs and is correct.
        """
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            next_val = last_value if t == self.T - 1 else self.values[t + 1]
            mask     = 1.0 - self.dones[t]
            delta    = self.rewards[t] + gamma * next_val * mask - self.values[t]
            gae      = delta + gamma * lam * mask * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values
        self.ptr = 0   # reset for next rollout

    def get_flat(self):
        """Flatten (T, N, ...) → (T*N, ...) for minibatch sampling."""
        T, N = self.T, self.N
        return (
            self.obs.view(T * N, -1),
            self.actions.view(T * N, -1),
            self.log_probs.view(T * N),
            self.advantages.view(T * N),
            self.returns.view(T * N),
            self.values.view(T * N),    # ← add this
        )


# ==========================================================================
# PPO Trainer
# ==========================================================================

class PPOTrainer:

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = cfg["device"]

        # Environment
        # FIX 3: Use renamed Go2WalkEnv
        self.env = Go2WalkEnv(
            n_envs            = cfg["n_envs"],
            dt                = cfg["dt"],
            max_episode_steps = cfg["max_episode_steps"],
            headless          = cfg["headless"],
        )

        # Network
        self.net = ActorCritic(
            obs_dim = Go2WalkEnv.OBS_DIM,
            act_dim = Go2WalkEnv.ACT_DIM,
            hidden  = cfg["hidden_size"],
        ).to(self.device)

        print(f"Network parameters: {self.net.num_parameters:,}")

        # Optimiser
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr  = cfg["lr"],
            eps = 1e-5,
        )

        # Rollout buffer
        self.buf = RolloutBuffer(
            rollout_steps = cfg["rollout_steps"],
            n_envs        = cfg["n_envs"],
            obs_dim       = Go2WalkEnv.OBS_DIM,
            act_dim       = Go2WalkEnv.ACT_DIM,
            device        = self.device,
        )

        # Logging
        os.makedirs(cfg["run_dir"], exist_ok=True)
        self.writer      = SummaryWriter(cfg["run_dir"])
        self.global_step = 0
        self.start_time  = time.time()

        # FIX 10: Per-env episode tracking buffers so we never miss a
        # completed episode even when no env finishes in a given rollout.
        self.ep_returns = []
        self.ep_lengths = []
        self._env_ep_return = np.zeros(cfg["n_envs"], dtype=np.float32)
        self._env_ep_length = np.zeros(cfg["n_envs"], dtype=np.int32)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        obs = self.env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

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

            # Collect rollout
            t0 = time.time()
            obs_t = self._collect_rollout(obs_t)
            collect_time = time.time() - t0

            # PPO update
            t0 = time.time()
            metrics = self._ppo_update()
            update_time = time.time() - t0

            self.global_step += steps_per_rollout

            # Logging
            if update % cfg["log_interval"] == 0:
                elapsed   = time.time() - self.start_time
                steps_sec = self.global_step / elapsed

                mean_ret = np.mean(self.ep_returns[-20:]) if self.ep_returns else 0.0
                mean_len = np.mean(self.ep_lengths[-20:]) if self.ep_lengths else 0.0

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

            # Checkpoint
            if update % cfg["save_interval"] == 0:
                self._save_checkpoint()

        self._save_checkpoint(tag="final")
        self.writer.close()
        print("\nTraining complete.")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Run rollout_steps steps across all envs, storing transitions.

        FIX 4+5: store_step() is called BEFORE env.step() and
        store_outcome() AFTER, so obs/action/logp/value are stored
        at the same timestep as their corresponding reward/done.

        FIX 5+6: last_value is computed with get_value() (no action
        sampling) from the final obs_t. For done envs, obs_t is the
        reset observation — the GAE done-mask zeroes the bootstrap
        for those envs, which is correct (FIX 7).
        """
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg["rollout_steps"]):

                action_t, log_prob_t, value_t = self.net.get_action(obs_t)
                action_np = action_t.cpu().numpy()

                # FIX 4: store pre-step quantities BEFORE env.step()
                self.buf.store_step(
                    obs       = obs_t.cpu().numpy(),
                    actions   = action_np,
                    log_probs = log_prob_t.cpu().numpy(),
                    values    = value_t.cpu().numpy(),
                )

                obs_np, reward_np, done_np, info = self.env.step(action_np)

                # FIX 4: store post-step quantities AFTER env.step()
                self.buf.store_outcome(
                    rewards = reward_np,
                    dones   = done_np,
                )

                # FIX 10: Track per-env episode stats every step.
                # This guarantees we never miss a completed episode.
                self._env_ep_return += reward_np
                self._env_ep_length += 1
                finished = np.where(done_np)[0]
                for idx in finished:
                    self.ep_returns.append(float(self._env_ep_return[idx]))
                    self.ep_lengths.append(int(self._env_ep_length[idx]))
                self._env_ep_return[finished] = 0.0
                self._env_ep_length[finished] = 0

                obs_t = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=self.device
                )

            # FIX 6: Use get_value() for clean bootstrap — no action
            # sampled, no unnecessary computation.
            last_value_t = self.net.get_value(obs_t)

            self.buf.compute_gae(
                last_value = last_value_t,
                gamma      = self.cfg["gamma"],
                lam        = self.cfg["lam"],
            )

        return obs_t

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        """
        Run n_epochs passes over the rollout buffer with minibatch sampling.
        Returns dict of mean losses for logging.
        """
        self.net.train()
        cfg = self.cfg

        obs_flat, act_flat, lp_flat, adv_flat, ret_flat, val_flat = self.buf.get_flat()
        # DEBUG
        print(f"lp_flat  — mean: {lp_flat.mean():.4f}  std: {lp_flat.std():.4f}  device: {lp_flat.device}")
        print(f"adv_flat — mean: {adv_flat.mean():.4f}  std: {adv_flat.std():.4f}")
        print(f"ret_flat — mean: {ret_flat.mean():.4f}  std: {ret_flat.std():.4f}")
        print(f"obs_flat — mean: {obs_flat.mean():.4f}  device: {obs_flat.device}")

        # Normalise advantages across the full rollout
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

                # value_loss   = (ret_flat[mb] - value).pow(2).mean()
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
        # FIX 8: Removed dead variable chk_dir which was computed but
        # never used. Checkpoints always save to cfg["run_dir"].
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
        ckpt    = torch.load(path, weights_only=False)
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

    # FIX 3: Use renamed Go2WalkEnv
    env = Go2WalkEnv(
        n_envs            = 1,
        headless          = False,
        max_episode_steps = cfg["max_episode_steps"],
        dt                = cfg["dt"],
    )
    net = ActorCritic(Go2WalkEnv.OBS_DIM, Go2WalkEnv.ACT_DIM, cfg["hidden_size"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    returns, lengths = [], []
    for ep in range(n_episodes):
        obs    = env.reset()
        done   = np.array([False])
        ep_ret = 0.0
        ep_len = 0

        while not done[0]:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                act, _, _ = net.get_action(obs_t, deterministic=False)
                action = act.numpy()

            obs, reward, done, _ = env.step(action)
            ep_ret += reward[0]
            ep_len += 1

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

    # FIX 9: Always use n_minibatches=4 so each epoch has 4 gradient
    # steps regardless of scale. The original code could produce a single
    # huge minibatch per epoch (very noisy updates at large n_envs).
    total_buffer   = n_envs * args.rollout_steps
    n_minibatches  = 4
    minibatch_size = max(total_buffer // n_minibatches, 256)

    return dict(
        # Environment
        n_envs             = n_envs,
        dt                 = 0.02,
        max_episode_steps  = 2000,
        headless           = args.headless,

        # Network
        hidden_size        = 256,

        # PPO
        total_steps        = args.total_steps,
        rollout_steps      = args.rollout_steps,
        minibatch_size     = minibatch_size,
        n_epochs           = 3,
        gamma              = 0.99,
        lam                = 0.95,
        clip_eps           = 0.2,
        lr                 = 1e-4,
        vf_coef            = 0.5,
        ent_coef           = 0.005,
        max_grad_norm      = 1.0,

        # Logging
        device             = args.device,
        run_dir            = args.run_dir,
        log_interval       = 10,
        save_interval      = 100,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",        type=int,  default=100)
    parser.add_argument("--total-steps",   type=int,  default=10_000_000)
    parser.add_argument("--rollout-steps", type=int,  default=24)
    parser.add_argument("--device",        type=str,  default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-dir",       type=str,  default="runs/go2_walk")
    parser.add_argument("--headless",      action="store_true", default=True)
    parser.add_argument("--resume",        type=str,  default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval",          type=str,  default=None,
                        help="Path to checkpoint to evaluate (no training)")
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
