"""
The digital twin, running the DOMO way.

No task. A World is spawned (robot + arena + lidar, goal-free), the robot's
current skill repertoire is loaded into a library, and a PlanningController
governs it — authoring skill programs at runtime and reacting to their
outcomes. The ExploreMission below is placeholder intelligence for the M1
LLM supervisor: same seat, same inputs (library catalog + outcome traces),
same output (grammar programs).

    python examples/twin/twin_demo.py \\
        --walk runs/go2_cpg/checkpoint_final.pt \\
        --avoid runs/go2_avoidance/checkpoint_final.pt

    # print what the LLM planner would see
    python examples/twin/twin_demo.py --walk CKPT --catalog
"""

import argparse

import torch

from domo.checkpoints import load_locomotion_policy, pick_device
from domo.rl import ActorCritic, clean_state_dict
from domo.skills import PlanningController, make_go2_library
from domo.world import World, WorldConfig


class ExploreMission(PlanningController):
    """
    Wander legs with avoidance, recover after falls, stop when done.
    Every decision is made HERE, from state + outcome history — the shape
    of thing the LLM supervisor will generate.
    """

    LEGS = 3          # exploration legs before resting
    MAX_FAILS = 3     # consecutive failures before conceding a skill gap

    def __init__(self, library, world):
        super().__init__(library, decision_interval=10)
        self.world = world
        self.legs_done = 0
        self.fails = 0
        self.done = False
        self._recovery = "stand.for(2)"

    def plan(self, state, last):
        if last is not None:
            print(f"  [mission] '{last.program}' → "
                  f"{'ok' if last.succeeded else 'FAILED'}")
            if last.succeeded:
                self.fails = 0
                if last.program != self._recovery:
                    self.legs_done += 1
            else:
                self.fails += 1
                if self.fails >= self.MAX_FAILS:
                    # The robot cannot restore itself with what it knows —
                    # in the full system this is the M1 skill-gap moment:
                    # "I need a get-up/recovery skill" → M2/M3 training.
                    print("  [mission] repeated failures and no recovery "
                          "skill in the library — SKILL GAP (get-up). "
                          "Stopping mission.")
                    self.done = True
                    return None
                if last.program != self._recovery:
                    print("  [mission] leg failed — resting before retry")
                    return self._recovery

        if self.legs_done >= self.LEGS:
            print("  [mission] exploration complete — idling")
            self.done = True
            return None

        self.world.randomise_obstacles()
        # Alternate heading a little between legs to cover more ground.
        vyaw = 0.3 if self.legs_done % 2 == 0 else -0.3
        return (f"(avoid @ walk(vx=1, vyaw={vyaw})).until(moved(2)) "
                f">> stand.for(1)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walk", type=str, required=True)
    p.add_argument("--avoid", type=str, default=None,
                   help="Avoidance checkpoint (omit → zero corrections)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--catalog", action="store_true",
                   help="Print the planner-facing catalog and exit")
    args = p.parse_args()
    device = pick_device(args.device)

    # --- spawn the twin: a world, a robot, its sensors — no goal ----------
    world = World(WorldConfig(
        device=device, headless=args.headless, scene_kind="arena"))

    # --- its current repertoire --------------------------------------------
    walk_policy, _ = load_locomotion_policy(args.walk, str(world.device))
    if args.avoid:
        ckpt = torch.load(args.avoid, weights_only=False,
                          map_location=str(world.device))
        net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
        net.eval().to(world.device)
        avoid_policy = lambda obs: net.get_action(obs, deterministic=True)[0]
    else:
        avoid_policy = lambda obs: torch.zeros(obs.shape[0], 3)
    library = make_go2_library(walk_policy, avoid_policy, world.lidar)

    if args.catalog:
        print(library.describe())
        return

    # --- govern it ----------------------------------------------------------
    mission = ExploreMission(library, world)
    mission.setup(world.robot)
    loop = world.make_loop(mission)
    loop.reset()

    for step in range(args.steps):
        state = loop.step()
        if (step + 1) % 250 == 0:
            active = (mission.program.source if mission.program else "idle")
            print(f"  step {step + 1:5d} | "
                  f"pos=({state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f})"
                  f" | {active}")
        if mission.idle and mission.done:
            break

    print(f"\n  mission history ({len(mission.history)} programs):")
    for outcome in mission.history:
        mark = "✓" if outcome.succeeded else "✗"
        print(f"    {mark} {outcome.program}")


if __name__ == "__main__":
    main()
