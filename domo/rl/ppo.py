"""
PPO trainer, decoupled from any particular task.

The trainer accepts any environment implementing the `domo.tasks.VecTask`
step API (`reset()` / `step(actions)` returning torch tensors, plus
`num_envs`, `num_obs`, `num_actions`, `device`). It never imports a physics
engine — the RL layer is optional and replaceable.

Training loop (one "update" = one rollout + one PPO optimisation)::

    for update in 1..total_steps // (rollout_steps × num_envs):
        [linear LR decay]        → _collect_rollout (T steps, no_grad, GAE)
        → _ppo_update (n_epochs × minibatches, clipped surrogate + clipped
          value loss + entropy bonus)  → update_callback(trainer, update)
        → every log_interval: console + TensorBoard scalars
        → every save_interval: checkpoint_step_<step:09d>.pt
    checkpoint_final.pt

Checkpoint format (torch.save dict), the contract read by domo.checkpoints,
domo.policies, domo.eureka.worker and every example::

    {"step":        int   global env steps so far,
     "model_state": ActorCritic state_dict,
     "optim_state": Adam state_dict,
     "ppo_config":  asdict(PPOConfig),
     "extra":       caller-owned dict (e.g. {"task_config": asdict(cfg)}),
     "metrics":     {"mean_return", "mean_length"} over the last
                    `ep_stat_window` finished episodes}

`ActorCritic.from_state_dict` rebuilds the network from "model_state" alone,
so legacy script checkpoints (which carry "config"/"obs_dim" instead of
"ppo_config"/"extra") still load for inference.

Robustness knobs from the later experiment scripts are available but
default to the plain behaviour of the original trainer:
  * target_kl        — early-stop epochs when the approx-KL exceeds 1.5×
  * lr_schedule      — "constant" (default) or "linear" decay with a floor
  * guard_nonfinite  — roll the update back atomically if a loss/grad blows up
  * vloss_skip       — roll back when a minibatch value loss exceeds a bound
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .buffer import RolloutBuffer
from .networks import ActorCritic, clean_state_dict

__all__ = ["PPOConfig", "PPOTrainer"]

# Adam epsilon of the original trainer (matters for reproducibility).
_ADAM_EPS = 1e-5
# Advantage normalisation epsilon and clamp used under the stability guards.
_ADV_EPS = 1e-8
_ADV_CLAMP = 10.0
# Early-stop threshold multiplier on target_kl (PPO-implementation convention).
_KL_STOP_FACTOR = 1.5

_ZERO_METRICS = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}


@dataclass
class PPOConfig:
    """PPO hyperparameters; serialised into every checkpoint as `ppo_config`."""

    total_steps: int = 100_000_000       # env steps (num_envs × rollout × updates)
    rollout_steps: int = 24              # T: steps per env per update
    minibatch_size: int = 8192           # samples per gradient step
    n_epochs: int = 5                    # passes over each rollout
    gamma: float = 0.99                  # discount
    lam: float = 0.95                    # GAE λ
    clip_eps: float = 0.2                # ratio AND value clipping range
    lr: float = 3e-4
    vf_coef: float = 1.0                 # value-loss weight
    ent_coef: float = 0.01               # entropy-bonus weight
    max_grad_norm: float = 1.0
    hidden_size: int = 512               # ActorCritic trunk width
    trunk_layers: int = 2
    head_hidden: int | None = None       # None → hidden_size

    # Stability guards (opt-in)
    target_kl: float | None = None       # stop epochs when KL > 1.5 × this
    lr_schedule: str = "constant"        # "constant" | "linear"
    lr_floor_frac: float = 0.05          # linear decay never goes below lr × this
    guard_nonfinite: bool = False        # roll back on non-finite loss/grad
    vloss_skip: float | None = None      # rollback update if value loss exceeds this

    # Logging / checkpoints
    run_dir: str = "runs/experiment"
    log_interval: int = 10               # in updates
    save_interval: int = 100             # in updates
    ep_stat_window: int = 20             # episodes averaged for logging


class PPOTrainer:
    """
    Proximal Policy Optimisation over a VecTask.

    Attributes:
        net: the ActorCritic being trained (on `env.device`).
        opt: Adam optimiser.
        buf: RolloutBuffer [rollout_steps, num_envs].
        global_step: env steps consumed so far (restored by `load_state`).
        ep_returns / ep_lengths: every finished episode's return/length
            (undiscounted sum of the env reward), appended in rollout order.
        update_callback: optional observer `fn(trainer, update_idx)` called
            after every PPO update (used by domo.eureka for reward reflection).
    """

    def __init__(self, env, cfg: PPOConfig, extra_checkpoint_data: dict | None = None):
        """
        Args:
            env: VecTask-compatible environment (already constructed).
            cfg: hyperparameters.
            extra_checkpoint_data: caller-owned dict stored inside every
                checkpoint (e.g. the task config needed to rebuild the env).
        """
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.extra_checkpoint_data = extra_checkpoint_data or {}

        self.net = ActorCritic(
            obs_dim=env.num_obs, act_dim=env.num_actions,
            hidden=cfg.hidden_size,
            trunk_layers=cfg.trunk_layers,
            head_hidden=cfg.head_hidden,
        ).to(self.device)
        print(f"Network parameters: {self.net.num_parameters:,}")

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=_ADAM_EPS)
        self.buf = RolloutBuffer(cfg.rollout_steps, env.num_envs,
                                 env.num_obs, env.num_actions, self.device)

        os.makedirs(cfg.run_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.run_dir)
        self.global_step = 0
        self.start_time = time.time()

        self.ep_returns: list[float] = []
        self.ep_lengths: list[int] = []
        self._env_ep_return = torch.zeros(env.num_envs, device=self.device)
        self._env_ep_length = torch.zeros(env.num_envs, device=self.device,
                                          dtype=torch.int32)

        # Optional observer: called as update_callback(trainer, update_idx)
        # after every PPO update (used by domo.eureka for reward reflection).
        self.update_callback: Callable[[PPOTrainer, int], None] | None = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Run `total_steps // (rollout_steps × num_envs)` updates, then save
        `checkpoint_final.pt`. Note the update count is NOT reduced after
        `load_state`: resuming trains `total_steps` MORE env steps.
        """
        cfg = self.cfg
        obs, _ = self.env.reset()

        steps_per_rollout = cfg.rollout_steps * self.env.num_envs
        n_updates = cfg.total_steps // steps_per_rollout

        print(f"\n{'=' * 55}")
        print("  PPO Training")
        print(f"{'=' * 55}")
        print(f"  Envs          : {self.env.num_envs}")
        print(f"  Total steps   : {cfg.total_steps:,}")
        print(f"  Rollout steps : {cfg.rollout_steps}")
        print(f"  Minibatch     : {cfg.minibatch_size}")
        print(f"  Device        : {self.device}")
        print(f"  Run dir       : {cfg.run_dir}")
        print(f"{'=' * 55}\n")

        for update in range(1, n_updates + 1):
            self._apply_lr_schedule()

            obs = self._collect_rollout(obs)
            metrics = self._ppo_update()
            self.global_step += steps_per_rollout

            if self.update_callback is not None:
                self.update_callback(self, update)

            if update % cfg.log_interval == 0:
                self._log(metrics)
            if update % cfg.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint(tag="final")
        self.writer.close()
        print("\nTraining complete.")

    def _apply_lr_schedule(self) -> None:
        """Linear decay from `lr` to `lr × lr_floor_frac` over total_steps."""
        cfg = self.cfg
        if cfg.lr_schedule == "linear":
            frac = max(1.0 - self.global_step / max(cfg.total_steps, 1),
                       cfg.lr_floor_frac)
            for g in self.opt.param_groups:
                g["lr"] = cfg.lr * frac

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Step the env `rollout_steps` times with the stochastic policy,
        filling the buffer and computing GAE. Returns the last observation
        (the next rollout starts from it — envs are never reset here).
        """
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg.rollout_steps):
                action, log_prob, value = self.net.get_action(obs)
                self.buf.store_step(obs, action, log_prob, value)
                next_obs, _, reward, reset_buf, _ = self.env.step(action)
                self.buf.store_outcome(reward, reset_buf.float())
                self._track_episode_stats(reward, reset_buf)
                obs = next_obs

            last_value = self.net.get_value(obs)
            self.buf.compute_gae(last_value, self.cfg.gamma, self.cfg.lam)
        return obs

    def _track_episode_stats(self, reward: torch.Tensor, reset_buf: torch.Tensor) -> None:
        """Accumulate per-env return/length; flush finished episodes to the lists."""
        self._env_ep_return += reward
        self._env_ep_length += 1
        done_idx = reset_buf.nonzero(as_tuple=False).flatten()
        for idx in done_idx:
            self.ep_returns.append(float(self._env_ep_return[idx]))
            self.ep_lengths.append(int(self._env_ep_length[idx]))
        self._env_ep_return[done_idx] = 0.0
        self._env_ep_length[done_idx] = 0

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _minibatch_losses(self, obs, act, old_log_prob, adv, ret, old_value):
        """
        Clipped-surrogate PPO losses for one minibatch.

        Returns:
            (loss, policy_loss, value_loss, entropy_loss, ratio, logratio)
            where loss = policy + vf_coef × value + ent_coef × entropy_loss
            and entropy_loss = −mean entropy. The value loss is clipped
            around the rollout value with the same `clip_eps`.
        """
        cfg = self.cfg
        new_lp, entropy, value = self.net.evaluate(obs, act)
        logratio = new_lp - old_log_prob
        ratio = logratio.exp()

        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_clipped = old_value + (value - old_value).clamp(
            -cfg.clip_eps, cfg.clip_eps)
        value_loss = torch.max(
            (value - ret).pow(2),
            (value_clipped - ret).pow(2)).mean()

        entropy_loss = -entropy.mean()
        loss = (policy_loss + cfg.vf_coef * value_loss
                + cfg.ent_coef * entropy_loss)
        return loss, policy_loss, value_loss, entropy_loss, ratio, logratio

    def _ppo_update(self) -> dict:
        """
        Optimise on the stored rollout for `n_epochs` × minibatches.

        Advantages are normalised over the whole rollout. With the guards
        enabled the network is snapshotted first and restored if any
        minibatch produces a non-finite loss/grad-norm (or a value loss
        above `vloss_skip`) — the whole update is then discarded.

        Returns:
            mean policy_loss / value_loss / entropy / clip_frac over the
            minibatches that were applied (zeros if none).
        """
        self.net.train()
        cfg = self.cfg

        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()

        guarded = cfg.guard_nonfinite or cfg.vloss_skip is not None
        if guarded and not (
                torch.isfinite(adv_f).all() and torch.isfinite(ret_f).all()):
            return dict(_ZERO_METRICS)

        backup = None
        if guarded:
            backup = {k: v.detach().clone()
                      for k, v in self.net.state_dict().items()}

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + _ADV_EPS)
        if guarded:
            adv_f = adv_f.clamp(-_ADV_CLAMP, _ADV_CLAMP)

        total = obs_f.shape[0]
        metrics = {"policy_loss": [], "value_loss": [],
                   "entropy": [], "clip_frac": []}
        bad = False

        for _ in range(cfg.n_epochs):
            idx = torch.randperm(total, device=self.device)
            epoch_kl = []
            for start in range(0, total, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]

                (loss, policy_loss, value_loss, entropy_loss,
                 ratio, logratio) = self._minibatch_losses(
                    obs_f[mb], act_f[mb], lp_f[mb], adv_f[mb], ret_f[mb], val_f[mb])

                if cfg.target_kl is not None:
                    # Unbiased low-variance KL estimator (Schulman's k3).
                    with torch.no_grad():
                        epoch_kl.append(((ratio - 1.0) - logratio).mean().item())

                if guarded and not torch.isfinite(loss):
                    bad = True
                    break
                if cfg.vloss_skip is not None and value_loss.item() > cfg.vloss_skip:
                    print(f"  [warn] pathological minibatch "
                          f"(vloss={value_loss.item():.2e}); rolling back update.")
                    bad = True
                    break

                self.opt.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg.max_grad_norm)
                if guarded and not torch.isfinite(gnorm):
                    bad = True
                    break
                self.opt.step()

                clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(-entropy_loss.item())
                metrics["clip_frac"].append(clip_frac.item())

            if bad:
                break
            if (cfg.target_kl is not None and epoch_kl
                    and np.mean(epoch_kl) > _KL_STOP_FACTOR * cfg.target_kl):
                break

        if bad and backup is not None:
            self.net.load_state_dict(backup)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Logging / checkpoints
    # ------------------------------------------------------------------

    def _log(self, metrics: dict):
        """Console line + TensorBoard scalars (train/*, loss/*) at global_step."""
        cfg = self.cfg
        elapsed = time.time() - self.start_time
        steps_sec = self.global_step / elapsed
        w = cfg.ep_stat_window
        mean_ret = float(np.mean(self.ep_returns[-w:])) if self.ep_returns else 0.0
        mean_len = float(np.mean(self.ep_lengths[-w:])) if self.ep_lengths else 0.0

        print(
            f"  step {self.global_step:>10,} | "
            f"ret {mean_ret:>7.3f} | "
            f"len {mean_len:>5.0f} | "
            f"ploss {metrics['policy_loss']:>7.4f} | "
            f"vloss {metrics['value_loss']:>7.4f} | "
            f"clip {metrics['clip_frac']:>4.2f} | "
            f"{steps_sec:>7,.0f} sps"
        )
        self.writer.add_scalar("train/mean_return", mean_ret, self.global_step)
        self.writer.add_scalar("train/mean_ep_len", mean_len, self.global_step)
        self.writer.add_scalar("loss/policy", metrics["policy_loss"], self.global_step)
        self.writer.add_scalar("loss/value", metrics["value_loss"], self.global_step)
        self.writer.add_scalar("loss/entropy", metrics["entropy"], self.global_step)
        self.writer.add_scalar("train/clip_fraction", metrics["clip_frac"], self.global_step)
        self.writer.add_scalar("train/steps_per_sec", steps_sec, self.global_step)
        self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], self.global_step)

    def save_checkpoint(self, tag: str | None = None) -> str:
        """
        Write `<run_dir>/checkpoint_<tag>.pt` (or `checkpoint_step_<step:09d>.pt`
        when no tag) in the format described in the module docstring.

        Returns:
            The path written.
        """
        cfg = self.cfg
        name = f"checkpoint_{tag}" if tag else f"checkpoint_step_{self.global_step:09d}"
        path = os.path.join(cfg.run_dir, f"{name}.pt")
        w = cfg.ep_stat_window
        torch.save({
            "step": self.global_step,
            "model_state": self.net.state_dict(),
            "optim_state": self.opt.state_dict(),
            "ppo_config": asdict(cfg),
            "extra": self.extra_checkpoint_data,
            "metrics": {
                "mean_return": np.mean(self.ep_returns[-w:]) if self.ep_returns else 0.0,
                "mean_length": np.mean(self.ep_lengths[-w:]) if self.ep_lengths else 0.0,
            },
        }, path)
        print(f"  [ckpt] Saved {path}")
        return path

    def load_state(self, ckpt: dict):
        """
        Restore model/optimiser/step from a loaded checkpoint dict (resume).
        The network must have been built with the same architecture; use
        `PPOConfig(**ckpt["ppo_config"])` to guarantee that.
        """
        self.net.load_state_dict(clean_state_dict(ckpt["model_state"]))
        self.opt.load_state_dict(ckpt["optim_state"])
        self.global_step = ckpt["step"]
        print(f"Resumed from step {self.global_step:,} "
              f"(mean return: {ckpt['metrics']['mean_return']:.3f})")
