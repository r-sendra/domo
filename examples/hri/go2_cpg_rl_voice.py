"""
Voice-commanded CPG locomotion — library replica of the interactive part of
scripts/house_scene/go2_cpg_rl_voice.py.

A `VoiceCommander` thread (microphone → Whisper → Gemini → (vx, vy, vyaw))
writes into a shared `CommandState`; the sim loop reads it every control
step and feeds the frozen walk policy. (For training the locomotion policy
itself, use examples/locomotion/go2_cpg_rl.py — the original voice script
duplicated the whole trainer; here it lives in the library.)

Library pieces exercised: domo.hri (VoiceCommander, CommandState),
domo.tasks (Go2CPGWalkTask as a single-env stepping harness),
domo.checkpoints.

    python examples/hri/go2_cpg_rl_voice.py \\
        --eval runs/go2_cpg/checkpoint_final.pt --whisper-model tiny
    # no display / CPU
    python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --headless

Requires: pip install openai-whisper sounddevice google-genai
          export GEMINI_API_KEY=<key>
Runs until Ctrl+C; prints the current command and measured vx/height every
250 steps and a note whenever the robot falls and is reset. Saves nothing.
Not smoke-testable without a microphone and an API key.
"""

import argparse

import torch

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.hri import CommandState, VoiceCommander
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask

REPORT_EVERY = 250          # steps between status lines


def build_env(device: str, headless: bool) -> Go2CPGWalkTask:
    """Single-env walk task used purely as a stepping harness (no episodes)."""
    return Go2CPGWalkTask(Go2CPGWalkConfig(
        n_envs=1, device=device, headless=headless,
        max_episode_steps=1_000_000,     # run until interrupted
        resampling_time_s=1e9))          # commands come from the voice thread


def drive(env, policy_fn, state: CommandState) -> None:
    """Step forever, feeding the latest voice command to the policy."""
    obs, _ = env.reset()
    cmd = torch.zeros(1, 3, device=env.device)
    step = 0
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
        if step % REPORT_EVERY == 0:
            s = env.robot.state
            print(f"  [sim] cmd=({vx:.2f},{vy:.2f},{vyaw:.2f})  "
                  f"vx={s.base_lin_vel[0, 0].item():+.2f}  "
                  f"h={s.base_pos[0, 2].item():.2f}")


def run_voice_eval(checkpoint_path: str, whisper_model: str = "tiny",
                   initial_vx: float = 0.0, headless: bool = False):
    # Device is picked independently of any flag (as in the frozen script).
    device = pick_device("cuda")
    policy_fn, _ = load_locomotion_policy(checkpoint_path, device)
    env = build_env(device, headless)

    state = CommandState(vx=initial_vx)
    commander = VoiceCommander(state, whisper_model=whisper_model)
    commander.start()
    try:
        drive(env, policy_fn, state)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        commander.stop()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", type=str, required=True,
                   help="CPG locomotion checkpoint to drive")
    p.add_argument("--whisper-model", type=str, default="tiny",
                   choices=["tiny", "base", "small"])
    p.add_argument("--vx", type=float, default=0.0,
                   help="Initial forward speed before the first voice command")
    p.add_argument("--headless", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    run_voice_eval(args.eval, whisper_model=args.whisper_model,
                   initial_vx=args.vx, headless=args.headless)


if __name__ == "__main__":
    main()
