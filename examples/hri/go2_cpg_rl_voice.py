"""
Voice-commanded CPG locomotion — library replica of the interactive part of
scripts/house_scene/go2_cpg_rl_voice.py.

(For training the locomotion policy itself, use go2_cpg_rl.py — the original
voice script duplicated the whole trainer; here it lives in the library.)

    python examples/hri/go2_cpg_rl_voice.py \\
        --eval runs/go2_cpg/checkpoint_final.pt --whisper-model tiny

Requires: pip install openai-whisper sounddevice google-genai
          export GEMINI_API_KEY=<key>
"""

import argparse

import torch

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.hri import CommandState, VoiceCommander
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask


def run_voice_eval(checkpoint_path: str, whisper_model: str = "tiny",
                   initial_vx: float = 0.0, headless: bool = False):
    device = pick_device("cuda")
    policy_fn, _ = load_locomotion_policy(checkpoint_path, device)

    env = Go2CPGWalkTask(Go2CPGWalkConfig(
        n_envs=1, device=device, headless=headless,
        max_episode_steps=1_000_000,     # run until interrupted
        resampling_time_s=1e9))          # commands come from the voice thread

    state = CommandState(vx=initial_vx)
    commander = VoiceCommander(state, whisper_model=whisper_model)
    commander.start()

    obs, _ = env.reset()
    cmd = torch.zeros(1, 3, device=env.device)
    step = 0
    try:
        while True:
            vx, vy, vyaw = state.get()
            cmd[0, 0], cmd[0, 1], cmd[0, 2] = vx, vy, vyaw
            env.commands[:] = cmd
            with torch.no_grad():
                act = policy_fn(obs)
            obs, _, _, reset_buf, _ = env.step(act)
            step += 1
            if reset_buf[0]:
                print("  [sim] robot fell — resetting")
                obs, _ = env.reset()
            if step % 250 == 0:
                s = env.robot.state
                print(f"  [sim] cmd=({vx:.2f},{vy:.2f},{vyaw:.2f})  "
                      f"vx={s.base_lin_vel[0, 0].item():+.2f}  "
                      f"h={s.base_pos[0, 2].item():.2f}")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        commander.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=str, required=True,
                   help="CPG locomotion checkpoint to drive")
    p.add_argument("--whisper-model", type=str, default="tiny",
                   choices=["tiny", "base", "small"])
    p.add_argument("--vx", type=float, default=0.0,
                   help="Initial forward speed before the first voice command")
    p.add_argument("--headless", action="store_true", default=False)
    args = p.parse_args()

    run_voice_eval(args.eval, whisper_model=args.whisper_model,
                   initial_vx=args.vx, headless=args.headless)


if __name__ == "__main__":
    main()
