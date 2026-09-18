"""
Obstacle avoidance in a walled arena — library replica of
scripts/house_scene/go2_cpg_rl_lidar.py.

A small avoidance network (LiDAR sectors → velocity correction) is trained
on top of a FROZEN CPG locomotion policy.

    # Train
    python examples/avoidance/go2_cpg_rl_lidar.py \\
        --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt \\
        --n-envs 4096 --device cuda --headless

    # Evaluate
    python examples/avoidance/go2_cpg_rl_lidar.py \\
        --eval runs/go2_avoidance/checkpoint_final.pt \\
        --cpg-checkpoint runs/go2_cpg/checkpoint_final.pt --vx 0.6
"""

import argparse
from dataclasses import asdict

from domo.checkpoints import (
    configs_from_checkpoint,
    load_checkpoint,
    load_locomotion_policy,
    pick_device,
    world_from_avoid_config,
)
from domo.rl import ActorCritic, PPOConfig, PPOTrainer, clean_state_dict
from domo.skills import PlanningController, make_go2_library
from domo.tasks import Go2AvoidConfig, Go2AvoidTask


def lidar_model_from_args(args, default_none=False):
    """--lidar simple → idealised script sensor; --lidar xt16 → Hesai XT16.
    With default_none=True (eval), an unset --lidar returns None (no override)."""
    from domo.robot import generic_sector_lidar, hesai_xt16
    if args.lidar == "xt16":
        return hesai_xt16(n_horizontal=args.lidar_azimuth)
    if args.lidar is None and default_none:
        return None
    return generic_sector_lidar()


def build_configs(args):
    task_cfg = Go2AvoidConfig(
        n_envs=args.n_envs, dt=0.02, max_episode_steps=1000,
        device=args.device, headless=args.headless,
        scene_kind="arena",
        lidar_model=lidar_model_from_args(args))
    total_buffer = args.n_envs * args.rollout_steps
    # AvoidanceNet from the script: 3×128 trunk, 64 head
    ppo_cfg = PPOConfig(
        total_steps=args.total_steps, rollout_steps=args.rollout_steps,
        minibatch_size=max(total_buffer // 4, 256), n_epochs=5,
        lr=3e-4, ent_coef=0.02,
        hidden_size=128, trunk_layers=3, head_hidden=64,
        target_kl=0.02, guard_nonfinite=True,
        lr_schedule="linear", lr_floor_frac=0.05,
        run_dir=args.run_dir, ep_stat_window=50)
    return task_cfg, ppo_cfg


class AvoidanceMission(PlanningController):
    """
    The evaluation mission, authored where behaviour belongs: inside the
    controller. plan() re-issues the avoidance patrol program once per
    episode (re-scattering the obstacles between runs) and idles when done.
    This is the seat an LLM/HRL planner takes over — nothing outside this
    class decides what the robot does.
    """

    def __init__(self, library, world, vx, episodes):
        super().__init__(library, decision_interval=5)
        self.world = world
        self.vx = vx
        self.episodes = episodes
        self._started = 0

    def plan(self, state, last):
        if last is not None:
            print(f"  Episode {self._started}/{self.episodes} | "
                  f"{'SUCCESS' if last.succeeded else 'FAILURE'}")
            for line in last.trace:
                print(f"      {line}")
            print()
        if self._started >= self.episodes:
            return None                          # mission complete → idle
        self._started += 1
        self.world.reset_robot()                 # independent trials (sim-only)
        self.world.randomise_obstacles()         # fresh clutter per episode
        return f"(avoid @ walk(vx={self.vx})).for(20)"


def evaluate(checkpoint_path, cpg_checkpoint, n_episodes=5, command_vx=0.6,
             headless=False, lidar_model=None, draw_lidar=False):
    """
    Twin-style evaluation: spawn the goal-free World the checkpoint trained
    in, hand the robot to an AvoidanceMission PlanningController, and let
    it author/execute skill programs. Episode termination comes from the
    SkillCards (avoid's blocked() abort, walk's tipped()/fallen() aborts,
    the program's .for horizon) — not from an RL episode counter.

    lidar_model: optional override of the trained sensor (e.g. XT16 preview).
    draw_lidar: visualise rays in the viewer (slow on macOS with XT16).
    """
    device = pick_device("cuda")
    ckpt = load_checkpoint(checkpoint_path, device)

    task_cfg, _ = configs_from_checkpoint(ckpt, "avoid")
    task_cfg.device = device
    if lidar_model is not None:
        print(f"  [eval] sensor override: {task_cfg.lidar_model.name} "
              f"(trained) → {lidar_model.name}")
        task_cfg.lidar_model = lidar_model
    task_cfg.lidar_model.draw_debug = draw_lidar and not headless

    # --- the twin: same world the checkpoint trained in, no task ----------
    world = world_from_avoid_config(task_cfg, headless=headless)

    # --- skill library from the two checkpoints ---------------------------
    cpg_path = cpg_checkpoint or ckpt.get("extra", {}).get("cpg_checkpoint")
    walk_policy, _ = load_locomotion_policy(cpg_path, device)
    avoid_net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    avoid_net.eval().to(device)
    avoid_policy = lambda obs: avoid_net.get_action(obs, deterministic=True)[0]

    library = make_go2_library(
        walk_policy, avoid_policy, world.lidar,
        cpg=task_cfg.cpg,
        avoid_deltas=(task_cfg.delta_vx_max, task_cfg.delta_vy_max,
                      task_cfg.delta_vyaw_max),
        obs_max_range=task_cfg.obs_max_range)

    mission = AvoidanceMission(library, world, command_vx, n_episodes)
    mission.setup(world.robot)
    loop = world.make_loop(mission)
    loop.reset()

    step, max_steps = 0, (n_episodes + 1) * task_cfg.max_episode_steps
    while not (mission.idle and len(mission.history) >= n_episodes):
        state = loop.step()
        step += 1
        if step % 50 == 0 and mission.program is not None:
            walk = mission.program.instances.get("walk")
            cmd = walk.command[0].cpu().numpy() if walk else (0, 0, 0)
            print(f"    step {step:5d}  "
                  f"min_lidar={world.lidar.read().min().item():.2f}m  "
                  f"vx={state.base_lin_vel[0, 0].item():+.2f}  "
                  f"cmd=({cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f})")
        if step >= max_steps:
            print("  [eval] step budget exhausted")
            break

    wins = sum(o.succeeded for o in mission.history)
    print(f"  {wins}/{n_episodes} successful runs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpg-checkpoint", type=str, default=None,
                   help="Path to trained CPG locomotion checkpoint")
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--total-steps", type=int, default=100_000_000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--run-dir", type=str, default="runs/go2_avoidance")
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
