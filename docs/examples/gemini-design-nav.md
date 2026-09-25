# LLM level designer

An LLM is shown a task, a skill catalogue and a top-down ASCII map, and asked
for one composition program. It gets no control while the program runs. The
robot's actual path is then overlaid on the map with a verdict, which makes
this the cleanest test of whether a language model can plan in the grammar at
all.

## What it demonstrates

`examples/navigation/gemini_design_nav.py` is library-native. The Go2 spawns
at `R` in a 10 × 7 grid with three pillars and must reach `G` in the opposite
corner without touching anything. The grid *is* the world — cell column = x,
row = y, 1 m cells — so `goto(x=5, y=3) @ walk` routes straight to a cell.

-   [`domo.llm`](../api/llm-hri.md#providers): `make_llm`,
    `extract_code_block`, and the scripted / gemini / vllm / openai providers.
-   [`domo.skills`](../api/skills.md): `library.describe()` for the prompt,
    `library.compile` and the `CompileError` / `GrammarError` repair loop.
-   [`domo.control`](../api/control.md): `SimControlLoop`,
    `SingleSkillController`.
-   [`domo.robot`](../api/robot.md#sensors): Go2 plus a `SimulatedLidar`
    pooled into 36 sectors, only when avoidance is in the library.

Two strategies are open to the designer, and the prompt names both: a
collision-free polyline of `goto` legs, or a coarse route with
`avoid @ goto(x=.., y=..) @ walk` layering reactive lidar avoidance on each
leg. The `scripted` provider needs no API key and replays a hand-verified
route:

```
goto(x=1, y=4) @ walk >> goto(x=6, y=6) @ walk >> goto(x=9, y=6) @ walk
```

With `--scripted-avoid` it replays the beeline
`(avoid @ goto(x=9, y=6) @ walk) | stand.for(2)` instead, which shows the
composition running but often does not finish.

## Run it

=== "Offline (CPU)"

    ```bash
    python examples/navigation/gemini_design_nav.py policies/walk.pt \
        --avoid-checkpoint policies/avoid.pt --llm scripted --headless --device cpu
    ```

=== "Offline, goto only"

    ```bash
    # no avoid checkpoint needed
    python examples/navigation/gemini_design_nav.py policies/walk.pt \
        --no-avoid --headless --device cpu
    ```

=== "Offline, reactive route"

    ```bash
    # the 'avoid @ goto @ walk' beeline — may not reach the goal
    python examples/navigation/gemini_design_nav.py policies/walk.pt \
        --avoid-checkpoint policies/avoid.pt --llm scripted --scripted-avoid --headless
    ```

=== "Let Gemini design it"

    ```bash
    export GEMINI_API_KEY=<key>
    python examples/navigation/gemini_design_nav.py policies/walk.pt \
        --avoid-checkpoint policies/avoid.pt --llm gemini
    ```

## How it works

-   `Arena(rows)` parses the map, renders it (`render(path)` overlays visited
    cells as `*`), detects hits — robot centre within 0.6 m of a pillar
    centre — and builds the pillars plus a bounding wall pushed two cells out,
    so the corner spawn is not staring at a wall.
-   `build_world(arena, device, headless, use_avoid)` creates the engine, the
    scene, the Go2 at `R` and, when avoidance is in the library, the sector
    lidar. The lidar handle is attached before `scene.build()`; the
    `SimulatedLidar` wrapper comes after.
-   `build_library(args, lidar, device, use_avoid)` returns walk only, or walk
    + avoid with `avoid_deltas = (0.25, 0.5, 1.2)`. The forward cap is small
    on purpose: avoid may steer to skirt an obstacle but must not be able to
    reverse `goto`'s progress toward the goal.
-   `design_program(llm, arena, library, device, max_repairs=2)` builds the
    prompt, runs `extract_program` on the reply and calls `library.compile`.
    On a `CompileError` or `GrammarError` the error text is appended to the
    prompt and the LLM is asked again; after three failed attempts the script
    exits.
-   `run_blind(loop, program, arena, steps)` steps a `SimControlLoop`,
    records visited cells, and stops on a hit, on reaching the goal (within
    0.6 m), when the program finishes, or when the budget is spent.
-   `print_verdict(...)` prints the verdict, the map with the path overlaid,
    and the program trace.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--llm` | `scripted` | `scripted` (offline), `gemini`, `vllm`, `openai`. Any name `domo.llm.make_llm` accepts works, including `gemini-lc`; there is no `choices` restriction |
| `--avoid-checkpoint` | `runs/go2_cpg/checkpoint_final_avoid.pt` | trained lidar-avoidance net; enables `avoid @ ...`. `policies/avoid.pt` is the promoted copy of that file |
| `--no-avoid` | off | omit the avoidance skill (goto-only library) |
| `--scripted-avoid` | off | with `--llm scripted`, "design" the beeline `avoid @ goto @ walk` route instead of the safe polyline |
| `--steps` | `6000` | step budget for the blind run |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

## What you will see

```
Arena:
    0123456789   (x=col)
y=0 |R         |
y=1 |   #      |
...
[design attempt 1] LLM proposed:
  goto(x=1, y=4) @ walk >> goto(x=6, y=6) @ walk >> goto(x=9, y=6) @ walk

Compiled program:
  goto(x=1, y=4) @ walk >> ...
  step   250 | pos=(+0.40,+1.60) | skill=goto(x=1, y=4) @ walk
  ...
=== REACHED THE GOAL ===
Actual path taken (robot cells = '*'):
    ...
Program trace:
    ...
```

The verdict is one of `REACHED THE GOAL`, `CRASHED into an obstacle` or
`ran out of steps / gave up`, each followed by an emoji in the console. A
compile failure prints `compile error: ...` before the next attempt.

Nothing is saved.

## Runtime

The scripted route takes 2–3 min on CPU; a `--steps 200` smoke run finishes in
about a minute. On a GPU it is seconds. With a real provider, add the LLM
latency of up to three calls.

## Known limitations

!!! warning "The default `--avoid-checkpoint` is a stale `runs/` path"

    It points at `runs/go2_cpg/checkpoint_final_avoid.pt` from the original
    experiment. Pass `--avoid-checkpoint policies/avoid.pt`, or `--no-avoid`
    to drop the skill entirely.

-   The bundled avoid policy is conservative: it tends to stall a metre in
    front of a pillar rather than skirt it. `--scripted-avoid`, and LLM
    designs that lean on avoidance, often end with `ran out of steps`. The
    goto-only strategy is the reliable one. A stronger avoid policy would
    weave through — see [obstacle avoidance](avoidance.md) for how that
    network is trained.
-   The LLM sees no feedback during execution. A wrong plan is simply wrong;
    contrast [the twin](twin-demo.md), where the planner reads outcomes and
    re-plans.
-   `gemini` needs `google-genai` and `GEMINI_API_KEY`; `vllm` and `openai`
    need the `langchain` extra and a running endpoint. See
    [providers](../api/llm-hri.md#providers).
