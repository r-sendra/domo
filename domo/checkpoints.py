"""
Checkpoint I/O and config reconstruction (used by examples, evaluation,
and the future orchestrator).

Handles checkpoint I/O for BOTH formats:
  * new library checkpoints: {"ppo_config": ..., "extra": {"task_config": ...}}
  * legacy script checkpoints (scripts/house_scene/*): {"config": <flat dict>}

Networks are always rebuilt from weight shapes (ActorCritic.from_state_dict),
so any locomotion checkpoint ever trained with the 76-dim CPG observation
loads regardless of provenance.

Config dicts come from JSON-ish serialisation (dataclasses.asdict + json),
where tuples degrade to lists; `_lists_to_tuples` restores them so the
dataclasses compare equal to freshly constructed ones.
"""

from __future__ import annotations

import torch

from domo.control import CPGConfig
from domo.rl import ActorCritic, PPOConfig
from domo.robot import LidarModelConfig
from domo.tasks import Go2AvoidConfig, Go2CPGWalkConfig, ObstacleArenaConfig

__all__ = [
    "avoid_config_from_dict",
    "configs_from_checkpoint",
    "cpg_walk_config_from_dict",
    "load_checkpoint",
    "load_locomotion_policy",
    "pick_device",
    "world_from_avoid_config",
]

# Legacy pre-lidar_model configs stored the scan interval in control steps.
_LEGACY_LIDAR_INTERVAL_STEPS = 5


def pick_device(requested: str) -> str:
    """Fall back to CPU when CUDA is requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_checkpoint(path: str, device: str):
    """torch.load with pickled configs allowed (checkpoints carry dataclasses)."""
    return torch.load(path, weights_only=False, map_location=device)


def load_locomotion_policy(path: str, device: str):
    """
    Load a frozen velocity-tracking (CPG) policy from any checkpoint format.
    Returns (policy_fn, net): policy_fn(obs) → deterministic action.
    """
    ckpt = load_checkpoint(path, device)
    net = ActorCritic.from_state_dict(ckpt["model_state"])
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    print(f"  Frozen locomotion policy: {path}")
    print(f"    obs={net.trunk[0].in_features} act={net.log_std.shape[0]} "
          f"params={net.num_parameters:,}")

    def policy_fn(obs: torch.Tensor) -> torch.Tensor:
        action, _, _ = net.get_action(obs, deterministic=True)
        return action

    return policy_fn, net


# ---------------------------------------------------------------------------
# Config (de)serialisation
# ---------------------------------------------------------------------------

def _lists_to_tuples(d: dict) -> dict:
    """Shallow copy with list values turned back into tuples (JSON round-trip)."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def world_from_avoid_config(task_cfg: Go2AvoidConfig, headless: bool = True):
    """
    Spawn the goal-free digital twin in the same environment an avoidance
    checkpoint was trained in (arena or ReplicaCAD house), n_envs=1.
    """
    from domo.world import World, WorldConfig
    return World(WorldConfig(
        engine=task_cfg.engine, device=task_cfg.device, dt=task_cfg.dt,
        headless=headless,
        scene_kind=task_cfg.scene_kind, ground_height=task_cfg.ground_height,
        arena=task_cfg.arena,
        replica_scene_json=task_cfg.replica_scene_json,
        replica_asset_root=task_cfg.replica_asset_root,
        kp=task_cfg.kp, kd=task_cfg.kd,
        base_init_pos=task_cfg.base_init_pos,
        base_init_yaw_deg=task_cfg.base_init_yaw_deg,
        lidar_model=task_cfg.lidar_model,
    ), n_envs=1)


def cpg_walk_config_from_dict(d: dict) -> Go2CPGWalkConfig:
    """Rebuild a Go2CPGWalkConfig from its serialised dict (nested cpg dict allowed)."""
    d = dict(d)
    if isinstance(d.get("cpg"), dict):
        d["cpg"] = CPGConfig(**d["cpg"])
    return Go2CPGWalkConfig(**_lists_to_tuples(d))


def _migrate_legacy_lidar(d: dict) -> None:
    """
    In place: convert pre-lidar_model keys ({"lidar": {...}, "lidar_interval": n})
    into a `lidar_model`, unless the dict already carries one.
    """
    lc = d.pop("lidar") or {}
    interval = d.pop("lidar_interval", _LEGACY_LIDAR_INTERVAL_STEPS)
    d.setdefault("lidar_model", LidarModelConfig(
        n_horizontal=lc.get("n_horizontal", 36),
        n_vertical=lc.get("n_vertical", 5),
        fov_deg=tuple(lc.get("fov_deg", (360.0, 50.0))),
        rate_hz=1.0 / (interval * d.get("dt", 0.02)),
        max_range=lc.get("max_range", 4.0),
        pos_offset=tuple(lc.get("pos_offset", (0.0, 0.0, 0.35))),
        draw_debug=lc.get("draw_debug", False)))


def avoid_config_from_dict(d: dict) -> Go2AvoidConfig:
    """Rebuild a Go2AvoidConfig from its serialised dict (nested arena/lidar/cpg dicts allowed)."""
    d = dict(d)
    if isinstance(d.get("arena"), dict):
        d["arena"] = ObstacleArenaConfig(**_lists_to_tuples(d["arena"]))
    if isinstance(d.get("lidar_model"), dict):
        d["lidar_model"] = LidarModelConfig(**_lists_to_tuples(d["lidar_model"]))
    if "lidar" in d:
        _migrate_legacy_lidar(d)
    if isinstance(d.get("cpg"), dict):
        d["cpg"] = CPGConfig(**_lists_to_tuples(d["cpg"]))
    return Go2AvoidConfig(**_lists_to_tuples(d))


def _legacy_ppo_config(c: dict) -> PPOConfig:
    """PPOConfig from a flat script config; defaults mirror the original scripts."""
    return PPOConfig(
        total_steps=c.get("total_steps", 80_000_000),
        rollout_steps=c.get("rollout_steps", 24),
        minibatch_size=c.get("minibatch_size", 8192),
        n_epochs=c.get("n_epochs", 5),
        gamma=c.get("gamma", 0.99),
        lam=c.get("lam", 0.95),
        clip_eps=c.get("clip_eps", 0.2),
        lr=c.get("lr", 3e-4),
        vf_coef=c.get("vf_coef", 1.0),
        ent_coef=c.get("ent_coef", 0.01),
        max_grad_norm=c.get("max_grad_norm", 1.0),
        hidden_size=c.get("hidden_size", 512),
        target_kl=c.get("target_kl"),
        # The scripts only had a floor fraction when they used a linear decay.
        lr_schedule="linear" if "lr_floor_frac" in c else "constant",
        lr_floor_frac=c.get("lr_floor_frac", 0.05),
        vloss_skip=c.get("vloss_skip"),
        run_dir=c.get("run_dir", "runs/legacy"),
        log_interval=c.get("log_interval", 10),
        save_interval=c.get("save_interval", 100),
        ep_stat_window=50,
    )


def _legacy_task_config(c: dict, kind: str):
    """Task config from a flat script config (only the shared fields were stored)."""
    common = dict(
        n_envs=c.get("n_envs", 4096), dt=c.get("dt", 0.02),
        max_episode_steps=c.get("max_episode_steps", 1000),
        headless=c.get("headless", True))
    return Go2CPGWalkConfig(**common) if kind == "cpg_walk" else Go2AvoidConfig(**common)


def configs_from_checkpoint(ckpt: dict, kind: str):
    """
    Reconstruct (task_config, ppo_config) from a checkpoint of either format.

    Args:
        ckpt: loaded checkpoint dict (see `load_checkpoint`).
        kind: "cpg_walk" | "avoid" — which task dataclass to build.
    """
    if "ppo_config" in ckpt:                      # new library format
        ppo = PPOConfig(**ckpt["ppo_config"])
        tc = ckpt["extra"]["task_config"]
        task = (cpg_walk_config_from_dict(tc) if kind == "cpg_walk"
                else avoid_config_from_dict(tc))
        return task, ppo

    # Legacy script format: one flat dict.
    c = ckpt["config"]
    return _legacy_task_config(c, kind), _legacy_ppo_config(c)
