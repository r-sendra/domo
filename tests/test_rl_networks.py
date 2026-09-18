"""
ActorCritic architecture contract and legacy-checkpoint loading.

The checkpoints under runs/ and policies/ were trained by the original
scripts; `ActorCritic.from_state_dict` must keep rebuilding them from weight
shapes alone. Tests that need those files skip when they are absent.
"""

import importlib.util
import os

import pytest
import torch

from domo.rl import ActorCritic, PPOConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEGACY_CHECKPOINTS = [
    # (path, obs_dim, act_dim, hidden, trunk_layers, head_hidden)
    ("runs/go2_cpg/checkpoint_final_coupled.pt", 76, 12, 512, 2, 512),
    ("runs/go2_cpg/checkpoint_final.pt", 76, 12, 512, 2, 512),
    ("policies/walk.pt", 76, 12, 512, 2, 512),
    ("policies/avoid.pt", 36, 3, 128, 3, 64),
]


def test_architecture_and_state_dict_keys():
    net = ActorCritic(10, 4, hidden=32, trunk_layers=3, head_hidden=8)
    keys = set(net.state_dict())
    assert {"trunk.0.weight", "trunk.2.weight", "trunk.4.weight",
            "actor_head.0.weight", "actor_head.2.weight",
            "critic_head.0.weight", "critic_head.2.weight", "log_std"} <= keys
    assert net.trunk[0].in_features == 10 and net.trunk[0].out_features == 32
    assert net.actor_head[0].out_features == 8 and net.actor_head[2].out_features == 4
    assert net.critic_head[2].out_features == 1
    assert net.log_std.shape == (4,) and torch.all(net.log_std == 0)
    # head_hidden None → hidden
    assert ActorCritic(10, 4, hidden=32).actor_head[0].out_features == 32


def test_forward_shapes_and_deterministic_action():
    torch.manual_seed(0)
    net = ActorCritic(6, 2, hidden=16)
    obs = torch.randn(5, 6)
    mean, std, value = net(obs)
    assert mean.shape == (5, 2) and std.shape == (5, 2) and value.shape == (5,)
    assert torch.allclose(std, torch.ones(5, 2))          # exp(0)
    act, logp, val = net.get_action(obs, deterministic=True)
    assert torch.equal(act, mean) and logp.shape == (5,) and torch.equal(val, value)
    assert torch.equal(net.get_value(obs), value)
    logp2, ent, val2 = net.evaluate(obs, act)
    assert torch.allclose(logp2, logp) and ent.shape == (5,) and torch.equal(val2, value)


def test_from_state_dict_roundtrip_with_prefixes():
    torch.manual_seed(0)
    src = ActorCritic(7, 3, hidden=24, trunk_layers=3, head_hidden=12)
    sd = {"_orig_mod." + k: v for k, v in src.state_dict().items()}
    net = ActorCritic.from_state_dict(sd)
    assert net.trunk[0].in_features == 7 and len(net.trunk) == 6
    assert net.actor_head[0].out_features == 12
    for k, v in src.state_dict().items():
        assert torch.equal(net.state_dict()[k], v)


@pytest.mark.parametrize("rel, obs_dim, act_dim, hidden, layers, head", LEGACY_CHECKPOINTS)
def test_legacy_checkpoints_load(rel, obs_dim, act_dim, hidden, layers, head):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f"{rel} not present")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    net = ActorCritic.from_state_dict(ckpt["model_state"])
    assert net.trunk[0].in_features == obs_dim
    assert net.log_std.shape == (act_dim,)
    assert net.trunk[0].out_features == hidden
    assert len(net.trunk) == 2 * layers
    assert net.actor_head[0].out_features == head
    act, _, _ = net.get_action(torch.zeros(2, obs_dim), deterministic=True)
    assert act.shape == (2, act_dim) and torch.isfinite(act).all()


def _import_main():
    spec = importlib.util.spec_from_file_location("domo_main", os.path.join(ROOT, "main.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_build_policy_honours_checkpoint_architecture():
    """Eval must rebuild the net with the trunk depth / head width it was trained with."""
    main = _import_main()
    torch.manual_seed(0)
    ppo_cfg = PPOConfig(hidden_size=32, trunk_layers=3, head_hidden=8)
    trained = ActorCritic(5, 2, ppo_cfg.hidden_size, trunk_layers=3, head_hidden=8)
    ckpt = {"model_state": {"module." + k: v for k, v in trained.state_dict().items()},
            "ppo_config": ppo_cfg.__dict__}
    net = main.build_policy(ckpt, PPOConfig(**ckpt["ppo_config"]), 5, 2)
    assert not net.training
    for k, v in trained.state_dict().items():
        assert torch.equal(net.state_dict()[k], v)


def test_main_parser_flags_unchanged():
    main = _import_main()
    args = main.build_parser().parse_args([])
    assert vars(args) == {
        "n_envs": 4096, "total_steps": 100_000_000, "rollout_steps": 24,
        "device": "cuda", "run_dir": "runs/go2_walk", "headless": True,
        "resume": None, "eval": None, "terrain": "flat"}
    args = main.build_parser().parse_args(
        ["--n-envs", "8", "--device", "cpu", "--terrain", "rough", "--eval", "x.pt"])
    assert (args.n_envs, args.device, args.terrain, args.eval) == (8, "cpu", "rough", "x.pt")
    task_cfg, ppo_cfg = main.build_configs(main.build_parser().parse_args(["--n-envs", "64"]))
    assert task_cfg.n_envs == 64 and ppo_cfg.minibatch_size == max(64 * 24 // 4, 256)
