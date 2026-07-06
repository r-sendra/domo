"""
Obstacle avoidance inside a ReplicaCAD house — library replica of
scripts/house_scene/go2_cpg_rl_avoid_house.py.

Identical avoidance architecture to go2_cpg_rl_lidar.py; the scene is the
Habitat ReplicaCAD apartment instead of the randomised arena.

    python examples/avoidance/go2_cpg_rl_avoid_house.py \\
        --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt \\
        --scene-json scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json \\
        --asset-root scripts/house_scene/data/replica_cad/ \\
        --n-envs 1024 --device cuda --headless
"""

import argparse
from dataclasses import asdict

from domo.checkpoints import (configs_from_checkpoint, load_checkpoint,
                    load_locomotion_policy, pick_device)
from domo.rl import PPOConfig, PPOTrainer
from domo.tasks import Go2AvoidConfig, Go2AvoidTask

from go2_cpg_rl_lidar import evaluate, lidar_model_from_args


def build_configs(args):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpg-checkpoint", type=str, default=None)
    p.add_argument("--scene-json", type=str,
                   default="scripts/house_scene/data/replica_cad/configs/scenes/apt_0.scene_instance.json")
    p.add_argument("--asset-root", type=str,
                   default="scripts/house_scene/data/replica_cad/")
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=100_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir", type=str, default="runs/go2_avoid_house")
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval", type=str, default=None)
    p.add_argument("--vx", type=float, default=0.6)
    p.add_argument("--lidar", type=str, default=None,
                   choices=["simple", "xt16"],
                   help="Sensor model: idealised script lidar or Hesai XT16. "
                        "Train default: simple. Eval default: whatever the "
                        "checkpoint was trained with (pass to override).")
    p.add_argument("--lidar-azimuth", type=int, default=180,
                   help="XT16 azimuth samples in sim (multiple of 36; "
                        "1980 = full device fidelity)")
    p.add_argument("--draw-lidar", action="store_true", default=False,
                   help="Visualise lidar rays in the eval viewer (slow on macOS)")
    args = p.parse_args()
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

    if args.resume:
        ckpt = load_checkpoint(args.resume, args.device)
        task_cfg, ppo_cfg = configs_from_checkpoint(ckpt, "avoid")
        task_cfg.device = args.device
        env = Go2AvoidTask(task_cfg, locomotion_policy=policy_fn)
        trainer = PPOTrainer(env, ppo_cfg, extra_checkpoint_data={
            "task_config": asdict(task_cfg),
            "cpg_checkpoint": args.cpg_checkpoint})
        trainer.load_state(ckpt)
    else:
        task_cfg, ppo_cfg = build_configs(args)
        env = Go2AvoidTask(task_cfg, locomotion_policy=policy_fn)
        trainer = PPOTrainer(env, ppo_cfg, extra_checkpoint_data={
            "task_config": asdict(task_cfg),
            "cpg_checkpoint": args.cpg_checkpoint})

    trainer.train()


if __name__ == "__main__":
    main()
