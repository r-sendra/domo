"""
Gemini-as-level-designer: the LLM designs a whole navigation program from a
top-down map, then we run it BLIND.

The Go2 spawns at 'R' in a walled arena of '#' obstacles and must reach 'G' in
the opposite corner without touching anything. We hand the LLM three things —
the task, the skill catalogue (`library.describe()`), and a cenital ASCII view
of the arena — and ask for ONE composition program. The LLM has no control
during the mission: it only designs. We compile its program and execute it,
then replay the robot's actual path on the map to see if its plan held up.

The grid is the world: cell (col, row) sits at world (x=col, y=row) metres, so
'goto(x=5, y=3) @ walk' waypoints route straight to map cells. Because `goto`
is the pose-feedback navigation skill, each leg holds a straight line between
waypoints. The LLM can solve it two ways: a precise collision-free polyline of
waypoints, OR a coarse route with reactive lidar avoidance layered on top —
'avoid @ goto(x=.., y=..) @ walk' — which steers around obstacles the segment
would clip. Both skills are in the catalogue it sees.

(Caveat: the bundled avoid checkpoint is conservative and tends to stall in
front of obstacles rather than skirt them, so the `--scripted-avoid` beeline
often does not finish — it is here to show the composition, and its limits. A
stronger avoid policy would weave through. The default scripted route and the
goto-only strategy are reliable.)

Library pieces exercised: domo.llm (make_llm, extract_code_block — scripted /
gemini / vllm / openai providers), domo.skills (library.describe(), compile,
CompileError/GrammarError repair loop), domo.control (SimControlLoop,
SingleSkillController), domo.robot (Go2 + SimulatedLidar), domo.sim.
No frozen-script counterpart.

    # offline / CPU — canned safe route, no API key needed
    python examples/navigation/gemini_design_nav.py CKPT --llm scripted --headless --device cpu
    # offline — show reactive 'avoid @ goto @ walk' (may not reach the goal)
    python examples/navigation/gemini_design_nav.py CKPT --llm scripted --scripted-avoid --headless
    # let Gemini design it (needs GEMINI_API_KEY; GPU viewer by default)
    python examples/navigation/gemini_design_nav.py CKPT --llm gemini

Prints the arena, each design attempt, a position line every 250 steps, the
verdict (reached / crashed / ran out), the path overlaid on the map, and the
program trace. Needs the avoid checkpoint (--avoid-checkpoint, or --no-avoid
to drop the skill). Saves nothing.
"""

import argparse
import math

from domo.checkpoints import load_checkpoint, load_locomotion_policy, pick_device
from domo.control import SimControlLoop, SingleSkillController
from domo.llm import extract_code_block, make_llm
from domo.rl import ActorCritic, clean_state_dict
from domo.robot import GO2, Robot, SimulatedLidar
from domo.robot.lidar_models import generic_sector_lidar
from domo.sim import SimConfig, ViewerConfig, create_engine
from domo.skills import CompileError, GrammarError, make_go2_library

# --- the arena --------------------------------------------------------------
# '.' = free (rendered as a space for the LLM), '#' = obstacle, R = spawn,
# G = goal. Column index → world x, row index → world y (1 m cells).
# Isolated pillars placed just OFF the straight R→G diagonal, so a beeline to
# the goal grazes them (forcing a lateral dodge to a clear side) rather than
# hitting one head-on — a pillar collinear with start+goal would trap a purely
# reactive avoider. This is the avoid policy's happy case: lone obstacles in
# open space, like the training ring.
ARENA = [
    "R.........",
    "...#......",
    "..........",
    "..........",
    ".....#.#..",
    "..........",
    ".........G",
]

# What the ScriptedClient "designs" (so the demo runs without an API key).
# Default: a hand-verified collision-free polyline that routes AROUND the
# pillars with goto alone — reliable, always reaches the goal.
SCRIPTED_ROUTE = ("goto(x=1, y=4) @ walk >> goto(x=6, y=6) @ walk >> "
                  "goto(x=9, y=6) @ walk")

# Alternative (--scripted-avoid): beeline at the goal and let reactive
# avoidance handle the pillars. NOTE: the bundled avoid checkpoint is very
# conservative — it tends to stall a metre in front of an obstacle rather than
# skirt it, so this route often does NOT reach the goal. It is here to SHOW the
# 'avoid @ goto @ walk' composition running (and its limits), not as a reliable
# solver. A stronger avoid policy would weave through cleanly.
SCRIPTED_ROUTE_AVOID = "(avoid @ goto(x=9, y=6) @ walk) | stand.for(2)"

AVOID_CHECKPOINT = "runs/go2_cpg/checkpoint_final_avoid.pt"
# Correction authority granted to 'avoid' when it layers on goal-seeking goto.
# The forward/back cap is small on purpose: avoid may STEER (lateral + yaw) to
# skirt an obstacle but must not be able to reverse goto's progress toward the
# goal (with full authority the conservative policy just backs away from any
# obstacle on the goal bearing). (Δvx, Δvy, Δvyaw).
AVOID_DELTAS = (0.25, 0.5, 1.2)

CELL = 1.0                 # metres per grid cell (world x=col, y=row)
OBSTACLE_H = 0.5           # obstacle box height
HIT_RADIUS = 0.6           # robot-centre distance to an obstacle centre = hit
GOAL_TOL = 0.6             # reached the goal within this many metres
WALL_MARGIN = 2.0          # cells of clear floor between the grid and the walls
CONTROL_DT = 0.02          # control step (s)
LIDAR_SECTORS = 36         # azimuth pooling of the lidar (the avoid net's obs)
REPORT_EVERY = 250         # progress line cadence (control steps)


# ---------------------------------------------------------------------------
# Map parsing / rendering
# ---------------------------------------------------------------------------

class Arena:
    def __init__(self, rows):
        self.h = len(rows)
        self.w = max(len(r) for r in rows)
        self.obstacles, self.start, self.goal = [], None, None
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    self.obstacles.append((x, y))
                elif ch == "R":
                    self.start = (x, y)
                elif ch == "G":
                    self.goal = (x, y)
        if self.start is None or self.goal is None:
            raise ValueError("arena needs both an 'R' start and a 'G' goal")

    def render(self, path=()):
        """Top-down view; free = space. `path` cells overlaid as '*'."""
        path = set(path)
        header = "    " + "".join(str(x % 10) for x in range(self.w)) + "   (x=col)"
        lines = [header]
        for y in range(self.h):
            cells = []
            for x in range(self.w):
                if (x, y) == self.start:
                    cells.append("R")
                elif (x, y) == self.goal:
                    cells.append("G")
                elif (x, y) in self.obstacles:
                    cells.append("#")
                elif (x, y) in path:
                    cells.append("*")
                else:
                    cells.append(" ")
            lines.append(f"y={y:<2}|" + "".join(cells) + "|")
        return "\n".join(lines)

    def hit(self, x, y):
        return any(abs(x - ox) < HIT_RADIUS and abs(y - oy) < HIT_RADIUS
                   for ox, oy in self.obstacles)

    def build(self, scene):
        """Fixed obstacle boxes + a bounding wall around the grid."""
        for ox, oy in self.obstacles:
            scene.add_box(size=(CELL, CELL, OBSTACLE_H),
                          pos=(ox * CELL, oy * CELL, OBSTACLE_H / 2), fixed=True)
        # Walls enclosing the grid, pushed out by WALL_MARGIN cells so the
        # corner spawn keeps clear of them (the lidar-avoidance policy was
        # trained with obstacles metres away, not walls in its face).
        wx0, wx1 = (-0.5 - WALL_MARGIN) * CELL, (self.w - 0.5 + WALL_MARGIN) * CELL
        wy0, wy1 = (-0.5 - WALL_MARGIN) * CELL, (self.h - 0.5 + WALL_MARGIN) * CELL
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        span_x, span_y, t, h = wx1 - wx0, wy1 - wy0, 0.1, 0.6
        for size, pos in [
            ((span_x, t, h), (cx, wy0, h / 2)),
            ((span_x, t, h), (cx, wy1, h / 2)),
            ((t, span_y, h), (wx0, cy, h / 2)),
            ((t, span_y, h), (wx1, cy, h / 2)),
        ]:
            scene.add_box(size=size, pos=pos, fixed=True)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(arena, library):
    gx, gy = arena.goal
    sx, sy = arena.start
    return f"""You are a robot skill-composition planner for a Unitree Go2 quadruped.

MISSION
Design ONE program that walks the robot from START (R) to GOAL (G) across the
arena below WITHOUT touching any '#' obstacle or the surrounding wall. You get
exactly one attempt and NO control once it runs — plan the whole route now.

THE ARENA (top-down; a space is free floor, '#' is a solid obstacle)
{arena.render()}

COORDINATES
The grid IS the world. Cell column = world x, cell row = world y, in metres
({CELL:g} m per cell). START R is at (x={sx}, y={sy}); GOAL G is at (x={gx}, y={gy}).
A wall encloses the grid, so stay within it.

AVAILABLE SKILLS
{library.describe()}

HOW TO SOLVE IT
Chain 'goto(x=.., y=..) @ walk' waypoints with '>>' to move between points.
Each leg drives a straight line between consecutive waypoints. Two strategies:
  (a) Precise route — pick waypoints so every straight segment passes only
      through free cells.
  (b) Reactive avoidance — layer the lidar avoidance skill on a leg:
      'avoid @ goto(x=.., y=..) @ walk'. Then 'avoid' steers around obstacles
      the segment would otherwise clip, so you can use FEWER, coarser
      waypoints and let it dodge. Combine both as you like.
Finish at the goal cell. You may wrap the whole route in a '| stand.for(2)'
fallback.

OUTPUT
Return ONLY the program as a single fenced code block, no comments, no prose:
```
avoid @ goto(x=.., y=..) @ walk >> ... >> goto(x={gx}, y={gy}) @ walk
```"""


def extract_program(text):
    """Pull the composition program out of the LLM reply, tolerantly."""
    block = extract_code_block(text) or text
    parts = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()      # drop any stray comments
        if line and not line.lower().startswith(("here", "program", "the ")):
            parts.append(line)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Design (offline) → compile
# ---------------------------------------------------------------------------

def load_avoid_policy(path, device):
    """Trained lidar-avoidance net (obs = lidar sectors → Δv correction)."""
    ckpt = load_checkpoint(path, device)
    net = ActorCritic.from_state_dict(clean_state_dict(ckpt["model_state"]))
    net.eval().to(device)

    def avoid_policy(obs):
        return net.get_action(obs, deterministic=True)[0]
    return avoid_policy


def design_program(llm, arena, library, device, max_repairs=2):
    """Ask the LLM for a program; feed compile errors back up to `max_repairs` times."""
    prompt = build_prompt(arena, library)
    feedback = ""
    for attempt in range(max_repairs + 1):
        reply = llm.generate(prompt + feedback, temperature=0.4)
        program_text = extract_program(reply)
        print(f"\n[design attempt {attempt + 1}] LLM proposed:\n  {program_text}\n")
        try:
            program = library.compile(program_text, device=device)
            return program
        except (CompileError, GrammarError) as e:
            print(f"  compile error: {e}")
            feedback = (f"\n\nYour previous program did not compile:\n"
                        f"  {program_text}\nError: {e}\nFix it and return the "
                        f"corrected program only.")
    raise SystemExit("LLM could not produce a compilable program.")


def make_designer(args, use_avoid: bool):
    """The LLM client: a ScriptedClient replaying a canned route, or a real provider."""
    if args.llm == "scripted":
        route = (SCRIPTED_ROUTE_AVOID if (args.scripted_avoid and use_avoid)
                 else SCRIPTED_ROUTE)
        return make_llm("scripted", responses=[f"```\n{route}\n```"])
    return make_llm(args.llm)


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

def build_world(arena, device: str, headless: bool, use_avoid: bool):
    """Arena scene + Go2 (+ sector lidar when avoidance is in the library)."""
    engine = create_engine("genesis", device=device)
    scene = engine.create_scene(SimConfig(
        dt=CONTROL_DT, device=device, headless=headless, solver_iterations=100,
        viewer=ViewerConfig(camera_pos=(arena.w * 0.6, -3.0, arena.h * 0.9))))
    scene.add_ground()
    arena.build(scene)
    sx, sy = arena.start
    robot = Robot(GO2, scene, engine.device, kp=100.0, kd=2.0,
                  base_init_pos=(sx * CELL, sy * CELL, 0.35))

    # Lidar for reactive avoidance: rays against the real arena geometry,
    # pooled into 36 azimuth sectors (the avoid net's observation). The
    # handle must be attached before build(); the sensor wraps it after.
    lidar = None
    if use_avoid:
        lidar_model = generic_sector_lidar()
        lidar_handle = scene.add_lidar(robot.articulation,
                                       lidar_model.to_lidar_config())

    scene.build(n_envs=1)
    robot.bind(n_envs=1)
    if use_avoid:
        lidar = SimulatedLidar(lidar_handle, lidar_model, 1, n_sectors=LIDAR_SECTORS,
                               device=engine.device, control_dt=CONTROL_DT)
    return engine, scene, robot, lidar


def build_library(args, lidar, device: str, use_avoid: bool):
    policy_fn, _ = load_locomotion_policy(args.checkpoint, device)
    if not use_avoid:
        return make_go2_library(policy_fn)
    avoid_policy = load_avoid_policy(args.avoid_checkpoint, device)
    return make_go2_library(policy_fn, avoid_policy, lidar,
                            avoid_deltas=AVOID_DELTAS)


def run_blind(loop, program, arena, steps: int):
    """Execute the program with no further LLM input; returns (path, collided, reached)."""
    gx, gy = arena.goal
    path, collided, reached = set(), False, False
    for i in range(steps):
        state = loop.step()
        x, y = float(state.base_pos[0, 0]), float(state.base_pos[0, 1])
        path.add((round(x / CELL), round(y / CELL)))
        if arena.hit(x, y):
            collided = True
            break
        if math.hypot(x - gx * CELL, y - gy * CELL) < GOAL_TOL:
            reached = True
            break
        if program.finished:
            break
        if (i + 1) % REPORT_EVERY == 0:
            print(f"  step {i + 1:5d} | pos=({x:+.2f},{y:+.2f}) | "
                  f"skill={program.root.label()[:40]}")
    return path, collided, reached


def print_verdict(arena, program, path, collided: bool, reached: bool) -> None:
    verdict = ("REACHED THE GOAL 🎉" if reached else
               "CRASHED into an obstacle 💥" if collided else
               "ran out of steps / gave up 🥱")
    print(f"\n=== {verdict} ===")
    print("Actual path taken (robot cells = '*'):\n" + arena.render(path))
    print("\nProgram trace:")
    for line in program.trace:
        print("   ", line)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=str, help="CPG locomotion checkpoint")
    p.add_argument("--llm", default="scripted",
                   help="scripted (offline) | gemini | vllm | openai")
    p.add_argument("--avoid-checkpoint", type=str, default=AVOID_CHECKPOINT,
                   help="trained lidar-avoidance net; enables 'avoid @ ...'")
    p.add_argument("--no-avoid", action="store_true", default=False,
                   help="omit the avoidance skill (goto-only library)")
    p.add_argument("--scripted-avoid", action="store_true", default=False,
                   help="with --llm scripted, 'design' the beeline "
                        "'avoid @ goto @ walk' route instead of the safe "
                        "polyline (shows reactive avoidance; may not finish)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    use_avoid = not args.no_avoid

    arena = Arena(ARENA)
    print("Arena:\n" + arena.render())

    # --- world ------------------------------------------------------------
    engine, scene, robot, lidar = build_world(arena, device, args.headless, use_avoid)
    device = str(engine.device)

    # --- design ------------------------------------------------------------
    library = build_library(args, lidar, device, use_avoid)
    llm = make_designer(args, use_avoid)
    program = design_program(llm, arena, library, device)
    controller = SingleSkillController(program)
    controller.setup(robot)
    print(f"Compiled program:\n  {program.source}\n")

    # --- run blind ---------------------------------------------------------
    loop = SimControlLoop(scene, robot, controller, dt=CONTROL_DT,
                          sensors=[lidar] if lidar is not None else ())
    loop.reset()
    path, collided, reached = run_blind(loop, program, arena, args.steps)

    # --- verdict -----------------------------------------------------------
    print_verdict(arena, program, path, collided, reached)


if __name__ == "__main__":
    main()
