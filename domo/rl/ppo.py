"""
PPO trainer, decoupled from any particular task.

The trainer accepts any environment implementing the `domo.tasks.VecTask`
step API (`reset()` / `step(actions)` returning torch tensors, plus
`num_envs`, `num_obs`, `num_actions`). It never imports a physics engine —
the RL layer is optional and replaceable.

Robustness knobs from the later experiment scripts are available but
default to the plain behaviour of the original trainer:
  * target_kl        — early-stop epochs when the approx-KL exceeds 1.5×
  * lr_schedule      — "constant" (default) or "linear" decay with a floor
  * guard_nonfinite  — roll the update back atomically if a loss/grad blows up
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .buffer import RolloutBuffer
from .networks import ActorCritic, clean_state_dict

__all__ = ["PPOConfig", "PPOTrainer"]


@dataclass
class PPOConfig:
    total_steps: int = 100_000_000
    rollout_steps: int = 24
    minibatch_size: int = 8192
    n_epochs: int = 5
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 1.0
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    hidden_size: int = 512
    trunk_layers: int = 2
    head_hidden: int | None = None    # None → hidden_size

    # Stability guards (opt-in)
    target_kl: float | None = None
    lr_schedule: str = "constant"        # "constant" | "linear"
    lr_floor_frac: float = 0.05
    guard_nonfinite: bool = False
    vloss_skip: float | None = None   # rollback update if value loss exceeds this

    # Logging / checkpoints
    run_dir: str = "runs/experiment"
    log_interval: int = 10               # in updates
    save_interval: int = 100             # in updates
    ep_stat_window: int = 20             # episodes averaged for logging


class PPOTrainer:

    def __init__(self, env, cfg: PPOConfig, extra_checkpoint_data: dict = None):
        """
        env  : VecTask-compatible environment (already constructed).
        extra_checkpoint_data : caller-owned dict stored inside every
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

        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.buf = RolloutBuffer(cfg.rollout_steps, env.num_envs,
                                 env.num_obs, env.num_actions, self.device)

        os.makedirs(cfg.run_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.run_dir)
        self.global_step = 0
        self.start_time = time.time()

        self.ep_returns, self.ep_lengths = [], []
        self._env_ep_return = torch.zeros(env.num_envs, device=self.device)
        self._env_ep_length = torch.zeros(env.num_envs, device=self.device,
                                          dtype=torch.int32)

        # Optional observer: called as update_callback(trainer, update_idx)
        # after every PPO update (used by domo.eureka for reward reflection).
        self.update_callback = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
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
            if cfg.lr_schedule == "linear":
                frac = max(1.0 - self.global_step / max(cfg.total_steps, 1),
                           cfg.lr_floor_frac)
                for g in self.opt.param_groups:
                    g["lr"] = cfg.lr * frac

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

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            for _ in range(self.cfg.rollout_steps):
                action, log_prob, value = self.net.get_action(obs)
                self.buf.store_step(obs, action, log_prob, value)
                next_obs, _, reward, reset_buf, _ = self.env.step(action)
                self.buf.store_outcome(reward, reset_buf.float())

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
            self.buf.compute_gae(last_value, self.cfg.gamma, self.cfg.lam)
        return obs

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        self.net.train()
        cfg = self.cfg

        obs_f, act_f, lp_f, adv_f, ret_f, val_f = self.buf.get_flat()

        guarded = cfg.guard_nonfinite or cfg.vloss_skip is not None
        if guarded and not (
                torch.isfinite(adv_f).all() and torch.isfinite(ret_f).all()):
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "clip_frac": 0.0}

        backup = None
        if guarded:
            backup = {k: v.detach().clone()
                      for k, v in self.net.state_dict().items()}

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        if guarded:
            adv_f = adv_f.clamp(-10.0, 10.0)

        total = obs_f.shape[0]
        metrics = {"policy_loss": [], "value_loss": [],
                   "entropy": [], "clip_frac": []}
        bad = False

        for _ in range(cfg.n_epochs):
            idx = torch.randperm(total, device=self.device)
            epoch_kl = []
            for start in range(0, total, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                mb_old_values = val_f[mb]
                mb_ret = ret_f[mb]

                new_lp, entropy, value = self.net.evaluate(obs_f[mb], act_f[mb])
                logratio = new_lp - lp_f[mb]
                ratio = logratio.exp()

                if cfg.target_kl is not None:
                    with torch.no_grad():
                        epoch_kl.append(((ratio - 1.0) - logratio).mean().item())

                surr1 = ratio * adv_f[mb]
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_f[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = mb_old_values + (value - mb_old_values).clamp(
                    -cfg.clip_eps, cfg.clip_eps)
                value_loss = torch.max(
                    (value - mb_ret).pow(2),
                    (value_clipped - mb_ret).pow(2)).mean()

                entropy_loss = -entropy.mean()
                loss = (policy_loss + cfg.vf_coef * value_loss
                        + cfg.ent_coef * entropy_loss)

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
                    and np.mean(epoch_kl) > 1.5 * cfg.target_kl):
                break

        if bad and backup is not None:
            self.net.load_state_dict(backup)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Logging / checkpoints
    # ------------------------------------------------------------------

    def _log(self, metrics: dict):
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

    def save_checkpoint(self, tag: str = None) -> str:
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
        """Restore model/optimiser/step from a loaded checkpoint dict."""
        self.net.load_state_dict(clean_state_dict(ckpt["model_state"]))
        self.opt.load_state_dict(ckpt["optim_state"])
        self.global_step = ckpt["step"]
        print(f"Resumed from step {self.global_step:,} "
              f"(mean return: {ckpt['metrics']['mean_return']:.3f})")
