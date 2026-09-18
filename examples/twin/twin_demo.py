"""
The digital twin, running the DOMO way.

No task. A World is spawned (robot + arena + lidar, goal-free), the robot's
current skill repertoire is loaded into a library, and a PlanningController
governs it — authoring skill programs at runtime and reacting to their
outcomes. The ExploreMission below is placeholder intelligence for the M1
LLM supervisor: same seat, same inputs (library catalog + outcome traces),
same output (grammar programs).

Library pieces exercised: domo.world (World/WorldConfig, scene_kind="arena"),
domo.skills (make_go2_library, PlanningController), domo.checkpoints,
domo.rl (ActorCritic for the avoid net). No frozen-script counterpart.

    # GPU, viewer open
    python examples/twin/twin_demo.py \\
        --walk runs/go2_cpg/checkpoint_final.pt \\
        --avoid runs/go2_avoidance/checkpoint_final.pt
    # CPU, headless, stable checkpoints
    python examples/twin/twin_demo.py --walk policies/walk.pt --avoid policies/avoid.pt \\
        --headless --device cpu --steps 1000

    # print what the LLM planner would see (builds the world, then exits)
    python examples/twin/twin_demo.py --walk CKPT --catalog

Prints each program outcome as the mission re-plans, a position line every
250 steps, and the mission history at the end. Saves nothing.
"""

import argparse

import torch

from domo.checkpoints import load_checkpoint, load_locomotion_policy, pick_device
from domo.rl import ActorCritic, clean_state_dict
from domo.skills import PlanningController, make_go2_library
from domo.world import World, WorldConfig

REPORT_EVERY = 250          # steps between position lines


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


def load_avoid_policy(path: str, device: str):
    """Callable obs → raw Δv from a trained avoidance checkpoint."""
    ckpt = load_checkpoint(path, device)
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(device)

    def avoid_policy(obs):
        return net.get_action(obs, deterministic=True)[0]
    return avoid_policy


def zero_avoid_policy(obs):
    """No avoidance checkpoint: the avoid skill contributes zero corrections."""
    return torch.zeros(obs.shape[0], 3, device=obs.device)


def build_library(world, walk_ckpt: str, avoid_ckpt: str | None):
    """The robot's current repertoire: walk + (trained or null) avoid."""
    device = str(world.device)
    walk_policy, _ = load_locomotion_policy(walk_ckpt, device)
    avoid_policy = (load_avoid_policy(avoid_ckpt, device) if avoid_ckpt
                    else zero_avoid_policy)
    return make_go2_library(walk_policy, avoid_policy, world.lidar)


def run_mission(mission, loop, steps: int) -> None:
    for step in range(steps):
        state = loop.step()
        if (step + 1) % REPORT_EVERY == 0:
            active = (mission.program.source if mission.program else "idle")
            print(f"  step {step + 1:5d} | "
                  f"pos=({state.base_pos[0, 0]:+.2f},{state.base_pos[0, 1]:+.2f})"
                  f" | {active}")
        if mission.idle and mission.done:
            break


def print_history(mission) -> None:
    print(f"\n  mission history ({len(mission.history)} programs):")
    for outcome in mission.history:
        mark = "✓" if outcome.succeeded else "✗"
        print(f"    {mark} {outcome.program}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--walk", type=str, required=True)
    p.add_argument("--avoid", type=str, default=None,
                   help="Avoidance checkpoint (omit → zero corrections)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--catalog", action="store_true",
                   help="Print the planner-facing catalog and exit")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)

    # --- spawn the twin: a world, a robot, its sensors — no goal ----------
    world = World(WorldConfig(
        device=device, headless=args.headless, scene_kind="arena"))

    # --- its current repertoire --------------------------------------------
    library = build_library(world, args.walk, args.avoid)
    if args.catalog:
        print(library.describe())
        return

    # --- govern it ----------------------------------------------------------
    mission = ExploreMission(library, world)
    mission.setup(world.robot)
    loop = world.make_loop(mission)
    loop.reset()
    run_mission(mission, loop, args.steps)

    # --- report -------------------------------------------------------------
    print_history(mission)


if __name__ == "__main__":
    main()
