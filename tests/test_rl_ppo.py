"""
PPO trainer over a fake VecTask (no physics): loop mechanics, checkpoint
format, resume, stability guards and the rollout buffer's GAE.
"""

import itertools
import os

import pytest
import torch

from domo.rl import ActorCritic, PPOConfig, PPOTrainer, RolloutBuffer, clean_state_dict

N_ENVS, OBS, ACT, EP_LEN = 8, 5, 3, 7


class FakeEnv:
    """Random observations, quadratic action cost, fixed-length episodes."""

    def __init__(self, n=N_ENVS, obs=OBS, act=ACT, device="cpu"):
        self.num_envs = self.n_envs = n
        self.num_obs, self.num_actions = obs, act
        self.device = torch.device(device)
        self.t = torch.zeros(n, dtype=torch.int32)
        self.n_steps = 0

    def reset(self):
        self.t[:] = 0
        return torch.randn(self.num_envs, self.num_obs), None

    def step(self, a):
        assert a.shape == (self.num_envs, self.num_actions)
        self.n_steps += 1
        self.t += 1
        obs = torch.randn(self.num_envs, self.num_obs)
        rew = -(a ** 2).sum(-1) + obs[:, 0]
        done = self.t >= EP_LEN
        self.t[done] = 0
        return obs, None, rew, done, {"time_outs": done.float()}


def _cfg(tmp_path, **kw):
    base = dict(total_steps=N_ENVS * 4 * 6, rollout_steps=4, minibatch_size=16,
                n_epochs=2, hidden_size=16, run_dir=str(tmp_path),
                log_interval=1, save_interval=3)
    base.update(kw)
    return PPOConfig(**base)


def test_train_runs_expected_updates_and_writes_checkpoints(tmp_path):
    torch.manual_seed(0)
    env = FakeEnv()
    cfg = _cfg(tmp_path)
    seen = []
    trainer = PPOTrainer(env, cfg, extra_checkpoint_data={"task_config": {"n_envs": N_ENVS}})
    trainer.update_callback = lambda tr, i: seen.append(i)
    trainer.train()

    n_updates = cfg.total_steps // (cfg.rollout_steps * N_ENVS)
    assert seen == list(range(1, n_updates + 1))
    assert env.n_steps == n_updates * cfg.rollout_steps
    assert trainer.global_step == cfg.total_steps
    # 6 updates, save every 3 → two periodic checkpoints + final
    files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".pt"))
    assert files == ["checkpoint_final.pt",
                     f"checkpoint_step_{3 * 4 * N_ENVS:09d}.pt",
                     f"checkpoint_step_{6 * 4 * N_ENVS:09d}.pt"]
    # episode stats: every env finishes every EP_LEN steps
    assert len(trainer.ep_lengths) == N_ENVS * (env.n_steps // EP_LEN)
    assert set(trainer.ep_lengths) == {EP_LEN}


def test_checkpoint_format_and_resume(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path)
    trainer = PPOTrainer(FakeEnv(), cfg, extra_checkpoint_data={"task_config": {"x": 1}})
    trainer.train()

    ckpt = torch.load(tmp_path / "checkpoint_final.pt", weights_only=False)
    assert set(ckpt) == {"step", "model_state", "optim_state", "ppo_config", "extra", "metrics"}
    assert ckpt["step"] == cfg.total_steps
    assert ckpt["ppo_config"] == PPOConfig(**ckpt["ppo_config"]).__dict__
    assert ckpt["extra"] == {"task_config": {"x": 1}}
    assert set(ckpt["metrics"]) == {"mean_return", "mean_length"}
    assert ckpt["metrics"]["mean_length"] == EP_LEN

    # from_state_dict reproduces the trained architecture exactly
    net = ActorCritic.from_state_dict(ckpt["model_state"])
    for k, v in trainer.net.state_dict().items():
        assert torch.equal(net.state_dict()[k], v)

    # resume restores weights, optimiser and step counter
    resumed = PPOTrainer(FakeEnv(), PPOConfig(**ckpt["ppo_config"]))
    resumed.load_state(ckpt)
    assert resumed.global_step == cfg.total_steps
    for k, v in trainer.net.state_dict().items():
        assert torch.equal(resumed.net.state_dict()[k], v)
    assert resumed.opt.state_dict()["param_groups"][0]["lr"] == cfg.lr


def test_linear_lr_schedule_decays_to_floor(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, lr_schedule="linear", lr_floor_frac=0.5)
    trainer = PPOTrainer(FakeEnv(), cfg)
    lrs = []
    trainer.update_callback = lambda tr, i: lrs.append(tr.opt.param_groups[0]["lr"])
    trainer.train()
    assert lrs[0] == pytest.approx(cfg.lr)              # frac = 1 at step 0
    assert lrs[-1] == pytest.approx(cfg.lr * 0.5)       # clipped at the floor
    assert all(a >= b for a, b in itertools.pairwise(lrs))  # monotone


def test_guarded_update_rolls_back_on_nonfinite(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, guard_nonfinite=True)
    trainer = PPOTrainer(FakeEnv(), cfg)
    trainer.env.reset()
    trainer._collect_rollout(torch.randn(N_ENVS, OBS))
    before = {k: v.clone() for k, v in trainer.net.state_dict().items()}
    trainer.buf.rewards[0, 0] = float("nan")             # poisons the returns
    trainer.buf.compute_gae(torch.zeros(N_ENVS), cfg.gamma, cfg.lam)
    metrics = trainer._ppo_update()
    assert metrics == {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    for k, v in trainer.net.state_dict().items():
        assert torch.equal(before[k], v)


def test_vloss_skip_rolls_back_whole_update(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, vloss_skip=1e-12)               # every minibatch is "pathological"
    trainer = PPOTrainer(FakeEnv(), cfg)
    trainer.env.reset()
    trainer._collect_rollout(torch.randn(N_ENVS, OBS))
    before = {k: v.clone() for k, v in trainer.net.state_dict().items()}
    trainer._ppo_update()
    for k, v in trainer.net.state_dict().items():
        assert torch.equal(before[k], v)


def test_target_kl_early_stop_still_updates(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg(tmp_path, target_kl=1e-9, n_epochs=5)
    trainer = PPOTrainer(FakeEnv(), cfg)
    trainer.env.reset()
    trainer._collect_rollout(torch.randn(N_ENVS, OBS))
    before = {k: v.clone() for k, v in trainer.net.state_dict().items()}
    metrics = trainer._ppo_update()
    assert metrics["value_loss"] > 0.0
    assert any(not torch.equal(before[k], v) for k, v in trainer.net.state_dict().items())


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

def test_gae_matches_reference_recursion():
    T, n = 4, 3
    buf = RolloutBuffer(T, n, 2, 1, "cpu")
    torch.manual_seed(1)
    for _ in range(T):
        buf.store_step(torch.randn(n, 2), torch.randn(n, 1), torch.randn(n), torch.randn(n))
        buf.store_outcome(torch.randn(n), torch.rand(n) > 0.5)
    assert buf.ptr == T
    last_value = torch.randn(n)
    gamma, lam = 0.9, 0.8
    buf.compute_gae(last_value, gamma, lam)
    assert buf.ptr == 0

    adv = torch.zeros(T, n)
    gae = torch.zeros(n)
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else buf.values[t + 1]
        m = 1.0 - buf.dones[t]
        delta = buf.rewards[t] + gamma * nv * m - buf.values[t]
        gae = delta + gamma * lam * m * gae
        adv[t] = gae
    assert torch.allclose(buf.advantages, adv)
    assert torch.allclose(buf.returns, adv + buf.values)

    flat = buf.get_flat()
    assert [x.shape for x in flat] == [(T * n, 2), (T * n, 1), (T * n,), (T * n,), (T * n,), (T * n,)]


def test_clean_state_dict_strips_prefixes():
    sd = {"_orig_mod.trunk.0.weight": 1, "module.log_std": 2, "plain": 3}
    assert clean_state_dict(sd) == {"trunk.0.weight": 1, "log_std": 2, "plain": 3}
