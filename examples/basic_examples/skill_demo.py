"""
The researcher workflow for the Skill / Controller / ControlLoop paradigm.

A hand-written Controller (no RL, no rewards) orchestrates two skills:
  walk  — CPGLocomotionSkill driven by a trained checkpoint
  stand — StandSkill safety fallback

The PatrolController walks a square (4 s forward, then turn ~90°) and drops
into `stand` if the robot tips too far. This file is the template for what
LLM-generated behaviour code will look like in the M5/Paper-3 milestone.

    python examples/basic_examples/skill_demo.py runs/go2_cpg/checkpoint_final.pt
    python examples/basic_examples/skill_demo.py CKPT --headless --steps 500
"""

import argparse

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.control import (Controller, CPGLocomotionSkill, SimControlLoop,
                          StandSkill)
from domo.robot import GO2, Robot
from domo.sim import SimConfig, ViewerConfig, create_engine


class PatrolController(Controller):
    """Walk a square; stand if we're about to tip. Plain code — no learning."""

    FORWARD_S, TURN_S = 4.0, 2.0
    VX, VYAW = 0.5, 0.8
    TIP_LIMIT = 0.5          # rad of roll/pitch that triggers the reflex

    def __init__(self, walk, stand, dt):
        super().__init__({"walk": walk, "stand": stand},
                         initial="walk", decision_interval=5)
        self.dt = dt

    def decide(self, state):
        walk = self.skills["walk"]

        # Reflex: if tipping, freeze in the default stance.
        tipping = state.base_euler[:, :2].abs().max() > self.TIP_LIMIT
        if tipping and self.active == "walk":
            print("  [ctrl] tipping — switching to stand")
            self.activate("stand")
            return
        if not tipping and self.active == "stand":
            print("  [ctrl] recovered — resuming walk")
            self.activate("walk")

        # Square patrol: alternate forward and turn phases on a timer.
        t = (self.ticks * self.dt) % (self.FORWARD_S + self.TURN_S)
        if t < self.FORWARD_S:
            walk.command[:, 0], walk.command[:, 2] = self.VX, 0.0
        else:
            walk.command[:, 0], walk.command[:, 2] = 0.0, self.VYAW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="CPG locomotion checkpoint")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    device = pick_device(args.device)
    dt = 0.02

    # --- world ------------------------------------------------------------
    engine = create_engine("genesis", device=device)
    scene = engine.create_scene(SimConfig(
        dt=dt, device=device, headless=args.headless, solver_iterations=100,
        viewer=ViewerConfig(camera_pos=(3.0, -3.0, 2.0))))
    scene.add_ground()
    robot = Robot(GO2, scene, engine.device, kp=100.0, kd=2.0,
                  base_init_pos=(0.0, 0.0, 0.35))
    scene.build(n_envs=1)
    robot.bind(n_envs=1)

    # --- brain ------------------------------------------------------------
    policy_fn, _ = load_locomotion_policy(args.checkpoint, str(engine.device))
    controller = PatrolController(walk=CPGLocomotionSkill(policy_fn),
                                  stand=StandSkill(), dt=dt)
    controller.setup(robot)

    # --- run --------------------------------------------------------------
    loop = SimControlLoop(scene, robot, controller, dt=dt)
    loop.reset()

    def report(step, state):
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1:5d} | skill={controller.active:5s} | "
                  f"pos=({state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f}) "
                  f"vx={state.base_lin_vel[0, 0]:+.2f}")

    loop.run(args.steps, callback=report)
    print("Done.")


if __name__ == "__main__":
    main()
