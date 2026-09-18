"""
The researcher workflow for the Skill / Controller / ControlLoop paradigm.

Default path — composition. A straight-line navigation command-skill
(`goto`) is layered on the walk motor-skill and chained with `>>` into a
route. `goto @ walk` closes the loop on the robot's pose every tick and nulls
lateral deviation from the intended line, so the square is traced with
minimal odometry drift and the program self-terminates when the last waypoint
is reached:

    goto(x=2,y=0) @ walk >> goto(x=2,y=2) @ walk >>
    goto(x=0,y=2) @ walk >> goto(x=0,y=0) @ walk

The compiled program IS a Skill, hosted here in a SingleSkillController.
This is the template for what LLM-generated behaviour will look like in the
M5/Paper-3 milestone (the planner emits exactly such a program string).

Contrast path — `--open-loop` runs a hand-written timed PatrolController that
sets velocity commands on a clock with NO pose feedback; its error integrates
uncorrected, so the square smears. Compare the printed |dist from origin|.

    python examples/basic_examples/skill_demo.py runs/go2_cpg/checkpoint_final.pt
    python examples/basic_examples/skill_demo.py CKPT --open-loop
    python examples/basic_examples/skill_demo.py CKPT --headless --steps 500
"""

import argparse

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.control import (Controller, CPGLocomotionSkill, SimControlLoop,
                          SingleSkillController, StandSkill)
from domo.robot import GO2, Robot
from domo.sim import SimConfig, ViewerConfig, create_engine
from domo.skills import make_go2_library

# The square, as a composition program the planner would emit. Closed-loop:
# each `goto @ walk` drives to a corner along a straight line and succeeds on
# arrival; `>>` advances to the next corner.
ROUTE = ("goto(x=2, y=0) @ walk >> goto(x=2, y=2) @ walk >> "
         "goto(x=0, y=2) @ walk >> goto(x=0, y=0) @ walk")
PROGRAM = f"({ROUTE}) | stand.for(2)"     # fall back to a safe stand if walk fails


class PatrolController(Controller):
    """Open-loop contrast: walk a square on a timer; stand if about to tip.
    No pose feedback — velocity-command error integrates uncorrected."""

    FORWARD_S, TURN_S = 4.0, 2.0
    VX, VYAW = 0.5, 0.8
    TIP_LIMIT = 0.5          # rad of roll/pitch that triggers the reflex

    def __init__(self, walk, stand, dt):
        super().__init__({"walk": walk, "stand": stand},
                         initial="walk", decision_interval=5)
        self.dt = dt

    def decide(self, state):
        walk = self.skills["walk"]
        tipping = state.base_euler[:, :2].abs().max() > self.TIP_LIMIT
        if tipping and self.active == "walk":
            print("  [ctrl] tipping — switching to stand")
            self.activate("stand")
            return
        if not tipping and self.active == "stand":
            print("  [ctrl] recovered — resuming walk")
            self.activate("walk")
        t = (self.ticks * self.dt) % (self.FORWARD_S + self.TURN_S)
        if t < self.FORWARD_S:
            walk.command[:, 0], walk.command[:, 2] = self.VX, 0.0
        else:
            walk.command[:, 0], walk.command[:, 2] = 0.0, self.VYAW


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="CPG locomotion checkpoint")
    p.add_argument("--open-loop", action="store_true", default=False,
                   help="run the timed PatrolController (drifts) instead of "
                        "the closed-loop navigation composition")
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
    if args.open_loop:
        controller = PatrolController(walk=CPGLocomotionSkill(policy_fn),
                                      stand=StandSkill(), dt=dt)
        label = "open-loop PatrolController"
    else:
        library = make_go2_library(policy_fn)
        program = library.compile(PROGRAM, device=str(engine.device))
        controller = SingleSkillController(program)
        import pdb; pdb.set_trace()
        label = f"composition: {program.source}"
    controller.setup(robot)

    # --- run --------------------------------------------------------------
    loop = SimControlLoop(scene, robot, controller, dt=dt)
    loop.reset()
    print(f"Running {label}")

    def report(step, state):
        if (step + 1) % 250 == 0:
            print(f"  step {step + 1:5d} | skill={controller.active:16s} | "
                  f"pos=({state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f}) "
                  f"vx={state.base_lin_vel[0, 0]:+.2f}")

    final = loop.run(args.steps, callback=report)

    # If the composition finished (route complete), show its execution trace —
    # the log the LLM supervisor reads to see what happened.
    program = controller.skills.get(controller.active)
    if not args.open_loop and getattr(program, "finished", False):
        print("Program trace:")
        for line in program.trace:
            print("   ", line)

    origin_err = (float(final.base_pos[0, 0]) ** 2
                  + float(final.base_pos[0, 1]) ** 2) ** 0.5
    print(f"Done. final pos=({final.base_pos[0, 0]:+.2f},"
          f"{final.base_pos[0, 1]:+.2f})  |dist from origin|={origin_err:.2f} m")


if __name__ == "__main__":
    main()
