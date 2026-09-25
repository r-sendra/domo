# Waypoint navigation

The legacy waypoint driver, kept alive as a library module and driven entirely
through two injected callables. Run it to drive the robot to coordinates, to
type natural-language commands at it, or to see how an engine-agnostic
controller is wired to a simulator without either knowing about the other.

## What it demonstrates

`examples/navigation/evaluate_nav.py` is the library replica of
`scripts/house_scene/evaluate_nav.py`. The P-controller itself moved into
`domo.control.navigation`; the script keeps the sim wiring, the stdin
intervention and the demos.

-   [`domo.control`](../api/control.md#legacy-navigation):
    `PositionController` and `NavConfig`, driven through `step_fn(cmd)` and
    `pose_fn()` so the controller never touches the engine.
-   [`domo.tasks`](../api/tasks.md#go2cpgwalktask): a single-environment
    `Go2CPGWalkTask` used purely as a stepping harness — no episodes, no
    command resampling.
-   A non-blocking stdin `MissionControl` that the controller polls every
    control step for pause, abort, stop and speed changes.
-   Two command parsers: a rule-based `parse_simple`, and `parse_gemini`
    (`gemini-2.5-flash`, one call per utterance) behind `--gemini`.

This is deliberately *not* the skill grammar. It is the older, imperative
interface — `go_forward`, `turn`, `go_to` — and it is useful as a baseline
against [the composed `goto @ walk` route](skill-demo.md).

## Run it

=== "Smoke test (CPU)"

    ```bash
    python examples/navigation/evaluate_nav.py policies/walk.pt \
        --demo forward --headless
    ```

=== "The default demo"

    ```bash
    # square waypoints, with the viewer
    python examples/navigation/evaluate_nav.py policies/walk.pt
    ```

=== "Type commands"

    ```bash
    python examples/navigation/evaluate_nav.py policies/walk.pt --demo interactive

    # with an LLM parser instead of the rules
    export GEMINI_API_KEY=<key>
    python examples/navigation/evaluate_nav.py policies/walk.pt --demo interactive --gemini
    ```

## How it works

-   `build_env(device, headless)` creates the `Go2CPGWalkTask` with one
    environment, `max_episode_steps = 1_000_000` and
    `resampling_time_s = 1e9`, so it never ends an episode or overwrites the
    command on its own.
-   `make_step_and_pose(env, policy_fn)` returns the two callables:
    `step_fn` writes the command, queries the frozen policy and steps the
    task, resetting the episode if the robot fell; `pose_fn` returns
    `(x, y, yaw)`.
-   `MissionControl.poll()` uses `select` on stdin with a zero timeout, so it
    never blocks the control loop. It exposes `paused`, `aborted`
    (read-and-clear, one abort per goal), `stopped` and a speed delta.
-   `execute_commands(commands, controller)` maps command dicts — `forward`,
    `backward`, `turn`, `go_to`, `stop` — onto the controller's methods.
-   The three demos: `forward` walks 5 m, turns left 90°, walks 3 m at
    0.5 m/s, turns right 90° and stops; `waypoints` traces a 3 m square
    through `(3,0) (3,3) (0,3) (0,0)`; `interactive` runs a `nav>` prompt
    until `quit`.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--demo` | `waypoints` | one of `forward`, `waypoints`, `interactive` |
| `--gemini` | off | parse interactive commands with Gemini instead of the rule-based parser |
| `--headless` | off | no Genesis viewer |

There is no `--device`: it is `cuda` when available, else `cpu`.

### Intervention keys

While a demo runs, type a key and press ++enter++ on stdin.

| Key | Effect |
|-----|--------|
| `p` / `pause` | pause |
| `r` / `resume` | resume |
| `a` / `abort` | abort the current goal (one abort per goal) |
| `s` / `stop` | stop the mission |
| `+` / `faster`, `-` / `slower` | change speed by 0.2 m/s |
| `?` / `status` | print the intervention state |

The interactive prompt accepts `forward N [at S]`, `backward N`,
`turn N [left|right]`, `go to X Y`, `stop` and `quit`. Anything the
rule-based parser does not understand becomes `stop`.

## What you will see

```
  [ctrl] Intervention active — type commands + Enter:
         p=pause r=resume a=abort s=stop +=faster -=slower ?=status

=== Demo: square waypoints ===

  Waypoint 1/4: (3.0,0.0)
  ...controller progress lines...
=== Done ===
```

In interactive mode each parsed command is echoed as `→ [{"type": ...}]`
before it runs. Nothing is saved.

## Runtime

About 60 s on CPU for the `forward` demo, a few minutes for the square.
Seconds to tens of seconds on a GPU.

## Known limitations

!!! note "stdin intervention needs a real terminal"

    `MissionControl.poll()` uses `select`, which does not work on Windows or
    with a piped stdin. The poll swallows the `OSError`/`ValueError` and
    returns, so intervention is silently unavailable rather than broken — the
    demo still runs, it just ignores you.

-   A fall resets the episode: the robot reappears at the spawn while the
    controller keeps its goal and keeps driving toward it.
-   `--gemini` needs `google-genai` and `GEMINI_API_KEY`. Without either,
    `parse_gemini` prints a one-line hint and every utterance parses to
    `stop`.

For the same job expressed in the skill grammar, with pose feedback and a
self-terminating program, see [the skill demo](skill-demo.md); for a whole
route planned by an LLM up front, see
[the LLM level designer](gemini-design-nav.md).
