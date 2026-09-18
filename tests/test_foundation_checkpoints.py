"""
Engine-free checks for domo.checkpoints: config dict round-trips, the legacy
lidar migration, and loading of both checkpoint formats (the real legacy
files under runs/ and policies/ are used when present).
"""

import dataclasses
import glob
import json
import os

import pytest
import torch

from domo.checkpoints import (
    avoid_config_from_dict,
    configs_from_checkpoint,
    cpg_walk_config_from_dict,
    load_checkpoint,
    load_locomotion_policy,
    pick_device,
)
from domo.rl import PPOConfig
from domo.robot import LidarModelConfig
from domo.tasks import Go2AvoidConfig, Go2CPGWalkConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_FILES = sorted(glob.glob(os.path.join(REPO, "runs", "go2_cpg", "*.pt")))
WALK_PT = os.path.join(REPO, "policies", "walk.pt")


def _jsonish(cfg):
    """dataclass → dict as it comes back from JSON (tuples degraded to lists)."""
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def _normalised(cfg):
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def test_pick_device_falls_back_to_cpu_without_cuda():
    assert pick_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        assert pick_device("cuda") == "cpu"


def test_cpg_walk_config_roundtrip():
    original = Go2CPGWalkConfig(n_envs=8, dt=0.01)
    rebuilt = cpg_walk_config_from_dict(_jsonish(original))
    assert isinstance(rebuilt, Go2CPGWalkConfig)
    assert rebuilt.n_envs == 8 and rebuilt.dt == 0.01
    assert type(rebuilt.cpg) is type(original.cpg)
    assert _normalised(rebuilt) == _normalised(original)


def test_avoid_config_roundtrip_restores_nested_dataclasses_and_tuples():
    original = Go2AvoidConfig(n_envs=4, base_init_pos=(1.0, 2.0, 0.4),
                              lidar_model=LidarModelConfig(n_horizontal=72))
    rebuilt = avoid_config_from_dict(_jsonish(original))
    assert rebuilt.base_init_pos == (1.0, 2.0, 0.4)
    assert isinstance(rebuilt.lidar_model, LidarModelConfig)
    assert rebuilt.lidar_model.fov_deg == (360.0, 50.0)
    assert type(rebuilt.arena) is type(original.arena)
    assert _normalised(rebuilt) == _normalised(original)


def test_avoid_config_migrates_legacy_lidar_keys():
    d = {"n_envs": 2, "dt": 0.02,
         "lidar": {"n_horizontal": 36, "n_vertical": 5, "max_range": 3.0,
                   "fov_deg": [360.0, 50.0]},
         "lidar_interval": 10}
    cfg = avoid_config_from_dict(d)
    lm = cfg.lidar_model
    assert isinstance(lm, LidarModelConfig)
    assert lm.max_range == 3.0 and lm.fov_deg == (360.0, 50.0)
    assert lm.rate_hz == pytest.approx(5.0)                 # 10 steps @ 50 Hz
    assert lm.update_interval(0.02) == 10
    # A dict that already carries lidar_model wins over the legacy keys.
    d2 = dict(d, lidar_model={"n_horizontal": 72})
    assert avoid_config_from_dict(d2).lidar_model.n_horizontal == 72


def test_configs_from_new_format_checkpoint():
    ppo = PPOConfig(total_steps=123, hidden_size=64)
    task = Go2AvoidConfig(n_envs=3)
    ckpt = {"ppo_config": dataclasses.asdict(ppo),
            "extra": {"task_config": _jsonish(task)}}
    task2, ppo2 = configs_from_checkpoint(ckpt, "avoid")
    assert ppo2 == ppo
    assert _normalised(task2) == _normalised(task)
    task3, _ = configs_from_checkpoint(
        {"ppo_config": dataclasses.asdict(ppo),
         "extra": {"task_config": _jsonish(Go2CPGWalkConfig(n_envs=5))}}, "cpg_walk")
    assert isinstance(task3, Go2CPGWalkConfig) and task3.n_envs == 5


def test_configs_from_legacy_flat_dict():
    legacy = {"config": {"n_envs": 16, "dt": 0.02, "max_episode_steps": 500,
                         "lr": 1e-4, "lr_floor_frac": 0.1, "hidden_size": 256,
                         "target_kl": 0.02}}
    task, ppo = configs_from_checkpoint(legacy, "cpg_walk")
    assert isinstance(task, Go2CPGWalkConfig)
    assert task.n_envs == 16 and task.max_episode_steps == 500
    assert ppo.lr == 1e-4 and ppo.hidden_size == 256 and ppo.target_kl == 0.02
    assert ppo.lr_schedule == "linear" and ppo.lr_floor_frac == 0.1
    assert ppo.ep_stat_window == 50 and ppo.run_dir == "runs/legacy"
    # Without a floor fraction the scripts used a constant schedule.
    _, ppo_c = configs_from_checkpoint({"config": {}}, "avoid")
    assert ppo_c.lr_schedule == "constant" and ppo_c.total_steps == 80_000_000
    task_a, _ = configs_from_checkpoint({"config": {}}, "avoid")
    assert isinstance(task_a, Go2AvoidConfig)


@pytest.mark.skipif(not LEGACY_FILES, reason="no local legacy checkpoints")
@pytest.mark.parametrize("path", LEGACY_FILES, ids=os.path.basename)
def test_real_legacy_checkpoints_load(path):
    ckpt = load_checkpoint(path, "cpu")
    assert "config" in ckpt and "model_state" in ckpt
    kind = "avoid" if "avoid" in os.path.basename(path) else "cpg_walk"
    task, ppo = configs_from_checkpoint(ckpt, kind)
    assert isinstance(ppo, PPOConfig)
    assert task.n_envs == ckpt["config"]["n_envs"]


@pytest.mark.skipif(not os.path.exists(WALK_PT), reason="policies/walk.pt not present")
def test_stable_walk_policy_loads_and_is_frozen():
    from domo.policies import load_stable_locomotion
    policy_fn, net = load_stable_locomotion("cpu")
    assert all(not p.requires_grad for p in net.parameters())
    obs_dim = net.trunk[0].in_features
    action = policy_fn(torch.zeros(2, obs_dim))
    assert action.shape == (2, net.log_std.shape[0])
    # Deterministic: same obs → same action.
    assert torch.equal(action, policy_fn(torch.zeros(2, obs_dim)))
    fn2, _ = load_locomotion_policy(WALK_PT, "cpu")
    assert torch.equal(fn2(torch.zeros(2, obs_dim)), action)
