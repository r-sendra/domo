"""
CPG-RL locomotion for Go2 — library replica of scripts/house_scene/go2_cpg_rl.py.

Same CLI as the original:
    # Train
    python examples/locomotion/go2_cpg_rl.py --n-envs 4096 --device cuda --headless
    # Evaluate (watch the gait)
    python examples/locomotion/go2_cpg_rl.py --eval runs/go2_cpg/checkpoint_final.pt --vx 0.5
    # Resume (accepts new-format and legacy script checkpoints)
    python examples/locomotion/go2_cpg_rl.py --resume runs/go2_cpg/checkpoint_step_xxx.pt
"""

import argparse
import math
from dataclasses import asdict

import numpy as np
import torch

from domo.checkpoints import (configs_from_checkpoint, load_checkpoint,
                    load_locomotion_policy, pick_device)
from domo.rl import PPOConfig, PPOTrainer
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask


def build_configs(args):
    task_cfg = Go2CPGWalkConfig(
        n_envs=args.n_envs, dt=0.02, max_episode_steps=1000,
        device=args.device, headless=args.headless)
    total_buffer = args.n_envs * args.rollout_steps
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256), n_epochs=5,
        lr=3e-4, ent_coef=0.01, hidden_size=512,
        target_kl=0.02, vloss_skip=1e3,
        lr_schedule="linear", lr_floor_frac=0.05,
        run_dir=args.run_dir, ep_stat_window=50)
    return task_cfg, ppo_cfg


def evaluate(checkpoint_path, command=(0.5, 0.0, 0.0), n_episodes=3,
             headless=False):
    device = pick_device("cuda")
    policy_fn, _ = load_locomotion_policy(checkpoint_path, device)

    env = Go2CPGWalkTask(Go2CPGWalkConfig(
        n_envs=1, device=device, headless=headless,
        resampling_time_s=1e9))          # command held fixed for the demo
    cmd = torch.tensor([command], device=env.device)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        env.commands[:] = cmd
        done = torch.zeros(1, dtype=torch.bool)
        ep_ret, ep_len, vx_err, theta_hist = 0.0, 0, [], []
        while not done[0]:
            env.commands[:] = cmd
            with torch.no_grad():
                act = policy_fn(obs)
            obs, _, reward, reset_buf, _ = env.step(act)
            ep_ret += reward[0].item()
            ep_len += 1
            done = reset_buf.bool()
            state = env.robot.state
            vx_err.append(abs(command[0] - state.base_lin_vel[0, 0].item()))
            theta_hist.append(
                env.controller.oscillators.theta[0].cpu().numpy().copy())
            if ep_len % 50 == 0:
                r = env.controller.oscillators.r[0].cpu().numpy()
                print(f"    step {ep_len:4d}  "
                      f"vx={state.base_lin_vel[0, 0].item():+.2f}/{command[0]:.2f}  "
                      f"vy={state.base_lin_vel[0, 1].item():+.2f}  "
                      f"wz={state.base_ang_vel[0, 2].item():+.2f}  "
                      f"h={state.base_pos[0, 2].item():.2f}  "
                      f"r=[{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}]")
        th = np.array(theta_hist)[:, 0]
        wraps = int(np.sum(np.diff(th) < -math.pi))
        period = (ep_len * env.dt / wraps) if wraps > 0 else float("nan")
        print(f"  Episode {ep + 1} | return={ep_ret:7.2f} | length={ep_len:4d} | "
              f"mean |vx err|={np.mean(vx_err):.3f} m/s | gait period~{period:.2f}s\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--total-steps", type=int, default=80_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir", type=str, default="runs/go2_cpg")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--eval", type=str, default=None)
    p.add_argument("--vx", type=float, default=0.5)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--vyaw", type=float, default=0.0)
    args = p.parse_args()
    args.device = pick_device(args.device)

    if args.eval:
        evaluate(args.eval, command=(args.vx, args.vy, args.vyaw))
        return

    if args.resume:
        ckpt = load_checkpoint(args.resume, args.device)
        task_cfg, ppo_cfg = configs_from_checkpoint(ckpt, "cpg_walk")
        task_cfg.device = args.device
        env = Go2CPGWalkTask(task_cfg)
        trainer = PPOTrainer(env, ppo_cfg,
                             extra_checkpoint_data={"task_config": asdict(task_cfg)})
        trainer.load_state(ckpt)
    else:
        task_cfg, ppo_cfg = build_configs(args)
        env = Go2CPGWalkTask(task_cfg)
        trainer = PPOTrainer(env, ppo_cfg,
                             extra_checkpoint_data={"task_config": asdict(task_cfg)})

    trainer.train()


if __name__ == "__main__":
    main()
