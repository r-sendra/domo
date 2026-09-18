"""
Obstacle avoidance inside a ReplicaCAD house — library replica of
scripts/house_scene/go2_cpg_rl_avoid_house.py.

Identical avoidance architecture to go2_cpg_rl_lidar.py (whose `evaluate`,
`make_trainer` and `lidar_model_from_args` are reused here); only the scene
differs — the Habitat ReplicaCAD apartment instead of the randomised arena,
with the house floor height and spawn pose from the original script.

Library pieces exercised: domo.tasks (Go2AvoidTask, scene_kind="replica"),
domo.scenes.replica (via the task), domo.rl, domo.checkpoints.

    # Train (GPU; the house takes ~2 min to build even on CPU)
    python examples/avoidance/go2_cpg_rl_avoid_house.py \\
        --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt \\
        --scene-json scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json \\
        --asset-root scripts/house_scene/data/replica_cad/ \\
        --n-envs 1024 --device cuda --headless

    # Evaluate a house checkpoint (twin-style mission, see go2_cpg_rl_lidar.py)
    python examples/avoidance/go2_cpg_rl_avoid_house.py \\
        --eval runs/go2_avoid_house/checkpoint_final.pt \\
        --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt

Assets: scripts/house_scene/data/replica_cad/ (scene apt_0 by default).
Training writes checkpoints under --run-dir (runs/go2_avoid_house).
"""

import argparse

from go2_cpg_rl_lidar import add_common_args, evaluate, lidar_model_from_args, make_trainer

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.rl import PPOConfig
from domo.tasks import Go2AvoidConfig

SCENE_JSON = ("scripts/house_scene/data/replica_cad/configs/scenes/"
              "apt_0.scene_instance.json")
ASSET_ROOT = "scripts/house_scene/data/replica_cad/"


def build_configs(args):
    """Fresh task + PPO configs for the house (same PPO recipe as the arena)."""
    task_cfg = Go2AvoidConfig(
        n_envs=args.n_envs, dt=0.02, max_episode_steps=1000,
        device=args.device, headless=args.headless,
        scene_kind="replica",
        replica_scene_json=args.scene_json,
        replica_asset_root=args.asset_root,
        ground_height=0.2,                    # house floor from the script
        base_init_pos=(3.0, -3.0, 0.44),
        base_init_yaw_deg=180.0,              # original spawn heading
        lidar_model=lidar_model_from_args(args))
    total_buffer = args.n_envs * args.rollout_steps
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256), n_epochs=5,
        lr=3e-4, ent_coef=0.02,
        hidden_size=128, trunk_layers=3, head_hidden=64,
        target_kl=0.02, guard_nonfinite=True,
        lr_schedule="linear", lr_floor_frac=0.05,
        run_dir=args.run_dir, ep_stat_window=50)
    return task_cfg, ppo_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpg-checkpoint", type=str, default=None)
    p.add_argument("--scene-json", type=str, default=SCENE_JSON)
    p.add_argument("--asset-root", type=str, default=ASSET_ROOT)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=100_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir", type=str, default="runs/go2_avoid_house")
    p.add_argument("--headless", action="store_true", default=False)
    add_common_args(p)
    return p, p.parse_args()


def main():
    p, args = parse_args()
    args.device = pick_device(args.device)

    if args.eval:
        evaluate(args.eval, args.cpg_checkpoint,
                 command_vx=args.vx, headless=args.headless,
                 lidar_model=lidar_model_from_args(args, default_none=True),
                 draw_lidar=args.draw_lidar)
        return

    if not args.cpg_checkpoint:
        p.error("--cpg-checkpoint is required for training")

    policy_fn, _ = load_locomotion_policy(args.cpg_checkpoint, args.device)
    make_trainer(args, policy_fn, build_configs).train()


if __name__ == "__main__":
    main()
