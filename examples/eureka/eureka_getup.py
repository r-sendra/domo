"""
Use case: closing the skill gap the twin discovered.

In twin_demo.py the robot fell, exhausted its repertoire, and conceded
"SKILL GAP (get-up)". This example is the M2/M3 response: the Eureka
routine asks Gemini for reward functions, trains candidates on the
Go2GetUpTask (fixed success metric: upright & held 1 s), reflects, iterates
— optionally hardens the winner with DrEureka domain randomization — and
finally deploys the learned skill back into the twin to watch it get up.

    # Learn the skill (Gemini free tier: export GEMINI_API_KEY=...)
    python examples/eureka/eureka_getup.py \\
        --samples 4 --iterations 3 --train-steps 5000000 --device cuda

    # Add the DrEureka robustness stage
    python examples/eureka/eureka_getup.py --dr ...

    # Offline / no API key: scripted LLM with a hand-written reward
    python examples/eureka/eureka_getup.py --llm scripted ...

    # Watch a learned checkpoint get up in the twin
    python examples/eureka/eureka_getup.py --demo runs/eureka/getup/...pt
"""

import argparse
import math

import torch

from domo.checkpoints import pick_device
from domo.eureka import (DrEurekaConfig, EurekaConfig, SkillLearningRequest,
                         learn_skill)
from domo.llm.client import GeminiClient, ScriptedClient

REQUEST_DESCRIPTION = (
    "The quadruped (Unitree Go2) lies fallen on its side or back with "
    "scrambled joints. It must right itself — roll onto its feet, push up — "
    "and reach a stable standing posture at its nominal height, then hold "
    "still. Motions should be decisive but not violent (avoid wild flailing "
    "or joint-limit slamming).")

# A reasonable hand-written reward, used by --llm scripted so the whole
# pipeline runs offline. It doubles as a baseline for Gemini's candidates.
SCRIPTED_REWARD = '''\
Here is a reward function for the get-up task:

```python
def compute_reward(task):
    state = task.robot.state
    up = -state.projected_gravity[:, 2]              # 1 when upright
    uprightness = torch.clamp(up, 0.0, 1.0) ** 2
    height = torch.clamp(state.base_pos[:, 2] / 0.30, 0.0, 1.2)
    posture = torch.exp(-1.0 * (state.dof_pos
                                - task.robot.default_dof_pos).abs().sum(-1))
    still = torch.exp(-0.5 * state.base_ang_vel.norm(dim=-1)) * uprightness
    energy = -0.0005 * (task.actions - task.last_actions).pow(2).sum(-1)
    total = (1.5 * uprightness + 1.0 * height * uprightness
             + 0.5 * posture * uprightness + 0.5 * still + energy)
    return total, {
        "uprightness": 1.5 * uprightness,
        "height": 1.0 * height * uprightness,
        "posture": 0.5 * posture * uprightness,
        "stillness": 0.5 * still,
        "energy": energy,
    }
```'''

SCRIPTED_DR = '''```json
{"friction_range": [0.5, 1.5], "base_mass_range": [-0.5, 1.5],
 "kp_scale_range": [0.85, 1.15], "obs_noise_std": 0.02}
```'''


def demo(checkpoint: str, device: str, headless: bool, episodes: int = 3):
    """Deploy the learned policy as a skill in the twin: fallen → standing."""
    from domo.control import LearnedJointSkill, SimControlLoop, SingleSkillController
    from domo.rl import ActorCritic, clean_state_dict
    from domo.tasks.go2_getup import build_getup_observation
    from domo.world import World, WorldConfig

    world = World(WorldConfig(device=device, headless=headless,
                              scene_kind="flat", kp=100.0, kd=2.0,
                              base_init_pos=(0.0, 0.0, 0.18),
                              lidar_model=None))
    ckpt = torch.load(checkpoint, weights_only=False,
                      map_location=str(world.device))
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(world.device)

    skill = LearnedJointSkill(
        lambda obs: net.get_action(obs, deterministic=True)[0],
        obs_builder=build_getup_observation, action_scale=0.35, name="getup")
    controller = SingleSkillController(skill)
    controller.setup(world.robot)
    loop = world.make_loop(controller)

    for ep in range(episodes):
        loop.reset()
        # Knock the robot over (roll ~ 100–170°, random side)
        roll = (math.radians(100) + torch.rand(1).item()
                * math.radians(70)) * (1 if ep % 2 else -1)
        quat = torch.tensor([[math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]],
                            device=world.device)
        pos = torch.tensor([[0.0, 0.0, 0.18]], device=world.device)
        world.robot.articulation.set_base_pose(
            pos, quat, torch.tensor([0], device=world.device))
        world.robot.refresh()

        got_up_at = None
        for step in range(400):
            state = loop.step()
            upright = (state.base_pos[0, 2] > 0.26
                       and state.base_euler[0, :2].abs().max() < 0.4)
            if upright and got_up_at is None:
                got_up_at = step * world.dt
        print(f"  Episode {ep + 1}: "
              + (f"got up in {got_up_at:.2f}s, " if got_up_at else "did NOT get up, ")
              + f"final height {state.base_pos[0, 2]:.2f} m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--llm", type=str, default="gemini",
                   choices=["gemini", "scripted"])
    p.add_argument("--dr", action="store_true", default=False,
                   help="Run the DrEureka robustness stage on the winner")
    p.add_argument("--dr-retrain-steps", type=int, default=None)
    p.add_argument("--run-root", type=str, default="runs/eureka")
    p.add_argument("--demo", type=str, default=None,
                   help="Skip learning; deploy this checkpoint in the twin")
    p.add_argument("--headless", action="store_true", default=False)
    args = p.parse_args()
    device = pick_device(args.device)

    if args.demo:
        demo(args.demo, device, args.headless)
        return

    request = SkillLearningRequest(
        skill_name="getup",
        description=REQUEST_DESCRIPTION,
        task="go2_getup",
        eureka=EurekaConfig(
            iterations=args.iterations, samples=args.samples,
            n_envs=args.n_envs, train_steps=args.train_steps,
            device=device),
        run_dr=args.dr,
        dr=DrEurekaConfig(retrain_steps=(args.dr_retrain_steps
                                         or args.train_steps)),
        run_root=args.run_root,
        llm=args.llm,
    )

    if args.llm == "scripted":
        class OfflineClient(ScriptedClient):
            """Answers by prompt kind — reward request vs. DR proposal."""
            def generate(self, prompt, temperature=1.0):
                self.calls.append(prompt)
                if "domain randomization" in prompt:
                    return SCRIPTED_DR
                return SCRIPTED_REWARD
        llm = OfflineClient(["-"])
    else:
        llm = GeminiClient()

    skill = learn_skill(request, llm=llm)

    print("\nTo watch it in the twin:")
    print(f"  python examples/eureka/eureka_getup.py "
          f"--demo {skill.checkpoint}")


if __name__ == "__main__":
    main()
