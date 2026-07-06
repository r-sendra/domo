"""
Checkpoint I/O and config reconstruction (used by examples, evaluation,
and the future orchestrator).

Handles checkpoint I/O for BOTH formats:
  * new library checkpoints: {"ppo_config": ..., "extra": {"task_config": ...}}
  * legacy script checkpoints (scripts/house_scene/*): {"config": <flat dict>}

Networks are always rebuilt from weight shapes (ActorCritic.from_state_dict),
so any locomotion checkpoint ever trained with the 76-dim CPG observation
loads regardless of provenance.
"""

from __future__ import annotations

import torch

from domo.control import CPGConfig
from domo.rl import ActorCritic, PPOConfig
from domo.robot import LidarModelConfig
from domo.tasks import Go2AvoidConfig, Go2CPGWalkConfig, ObstacleArenaConfig


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_checkpoint(path: str, device: str):
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
    d = dict(d)
    if isinstance(d.get("cpg"), dict):
        d["cpg"] = CPGConfig(**d["cpg"])
    return Go2CPGWalkConfig(**{k: tuple(v) if isinstance(v, list) else v
                               for k, v in d.items()})


def avoid_config_from_dict(d: dict) -> Go2AvoidConfig:
    d = dict(d)
    if isinstance(d.get("arena"), dict):
        d["arena"] = ObstacleArenaConfig(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d["arena"].items()})
    if isinstance(d.get("lidar_model"), dict):
        d["lidar_model"] = LidarModelConfig(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d["lidar_model"].items()})
    # Migrate pre-lidar_model configs ({"lidar": {...}, "lidar_interval": n}).
    if "lidar" in d:
        lc = d.pop("lidar") or {}
        interval = d.pop("lidar_interval", 5)
        d.setdefault("lidar_model", LidarModelConfig(
            n_horizontal=lc.get("n_horizontal", 36),
            n_vertical=lc.get("n_vertical", 5),
            fov_deg=tuple(lc.get("fov_deg", (360.0, 50.0))),
            rate_hz=1.0 / (interval * d.get("dt", 0.02)),
            max_range=lc.get("max_range", 4.0),
            pos_offset=tuple(lc.get("pos_offset", (0.0, 0.0, 0.35))),
            draw_debug=lc.get("draw_debug", False)))
    if isinstance(d.get("cpg"), dict):
        d["cpg"] = CPGConfig(**{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in d["cpg"].items()})
    return Go2AvoidConfig(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in d.items()})


def configs_from_checkpoint(ckpt: dict, kind: str):
    """
    Reconstruct (task_config, ppo_config) from a checkpoint of either format.
    kind: "cpg_walk" | "avoid".
    """
    if "ppo_config" in ckpt:                      # new library format
        ppo = PPOConfig(**ckpt["ppo_config"])
        tc = ckpt["extra"]["task_config"]
        task = (cpg_walk_config_from_dict(tc) if kind == "cpg_walk"
                else avoid_config_from_dict(tc))
        return task, ppo

    # Legacy script format: one flat dict.
    c = ckpt["config"]
    ppo = PPOConfig(
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
        lr_schedule="linear" if "lr_floor_frac" in c else "constant",
        lr_floor_frac=c.get("lr_floor_frac", 0.05),
        vloss_skip=c.get("vloss_skip"),
        run_dir=c.get("run_dir", "runs/legacy"),
        log_interval=c.get("log_interval", 10),
        save_interval=c.get("save_interval", 100),
        ep_stat_window=50,
    )
    if kind == "cpg_walk":
        task = Go2CPGWalkConfig(
            n_envs=c.get("n_envs", 4096), dt=c.get("dt", 0.02),
            max_episode_steps=c.get("max_episode_steps", 1000),
            headless=c.get("headless", True))
    else:
        task = Go2AvoidConfig(
            n_envs=c.get("n_envs", 4096), dt=c.get("dt", 0.02),
            max_episode_steps=c.get("max_episode_steps", 1000),
            headless=c.get("headless", True))
    return task, ppo
