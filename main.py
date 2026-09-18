"""
DOMO entry point — train / evaluate Go2 velocity-tracking locomotion.

All heavy lifting lives in the library:
    domo.tasks.Go2WalkTask  — the environment (engine-agnostic)
    domo.rl.PPOTrainer      — the learning layer

Usage:
    python main.py --n-envs 4096 --device cuda --headless
    python main.py --resume runs/go2_walk/checkpoint_step_xxx.pt
    python main.py --eval   runs/go2_walk/checkpoint_final.pt
"""

import argparse
from dataclasses import asdict

import numpy as np
import torch

from domo.rl import ActorCritic, PPOConfig, PPOTrainer, clean_state_dict
from domo.tasks import Go2WalkConfig, Go2WalkTask

# ==========================================================================
# Config
# ==========================================================================

def build_configs(args) -> tuple[Go2WalkConfig, PPOConfig]:
    task_cfg = Go2WalkConfig(
        n_envs=args.n_envs,
        dt=0.02,
        max_episode_steps=1000,
        device=args.device,
        headless=args.headless,
        terrain=args.terrain,
    )

    total_buffer = args.n_envs * args.rollout_steps
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256),
        hidden_size=512,
        run_dir=args.run_dir,
    )
    return task_cfg, ppo_cfg


def load_checkpoint(path: str):
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(path, weights_only=False, map_location=map_location), map_location


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate(checkpoint_path: str, n_episodes: int = 10, headless: bool = False):
    ckpt, device = load_checkpoint(checkpoint_path)
    ppo_cfg = PPOConfig(**ckpt["ppo_config"])
    task_cfg = Go2WalkConfig(**ckpt["extra"]["task_config"])
    task_cfg.n_envs = 1
    task_cfg.device = device
    task_cfg.headless = headless

    env = Go2WalkTask(task_cfg)
    net = ActorCritic(env.num_obs, env.num_actions, ppo_cfg.hidden_size)
    net.load_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval()
    net = net.to(device)

    returns, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len = 0.0, 0

        while not done[0]:
            with torch.no_grad():
                act, _, _ = net.get_action(obs, deterministic=False)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done = reset_buf.bool()

        returns.append(ep_ret)
        lengths.append(ep_len)
        print(f"  Episode {ep + 1:2d} | return {ep_ret:7.2f} | length {ep_len}")

    print(f"\nMean return : {np.mean(returns):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f}")


# ==========================================================================
# Entry point
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4096)
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--run-dir", type=str, default="runs/go2_walk")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--terrain", type=str, default="flat",
                        choices=["flat", "rough"])
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval)
        return

    if args.resume:
        ckpt, device = load_checkpoint(args.resume)
        ppo_cfg = PPOConfig(**ckpt["ppo_config"])
        task_cfg = Go2WalkConfig(**ckpt["extra"]["task_config"])
        task_cfg.device = device
        env = Go2WalkTask(task_cfg)
        trainer = PPOTrainer(env, ppo_cfg,
                             extra_checkpoint_data={"task_config": asdict(task_cfg)})
        trainer.load_state(ckpt)
    else:
        task_cfg, ppo_cfg = build_configs(args)
        env = Go2WalkTask(task_cfg)
        trainer = PPOTrainer(env, ppo_cfg,
                             extra_checkpoint_data={"task_config": asdict(task_cfg)})

    trainer.train()


if __name__ == "__main__":
    main()
