# Skill demo

The shortest complete path through the library: build an engine and a scene,
put a Go2 in it, load a policy, wrap the policy in skills, hand the skills to
a controller and let a control loop tick it. Two controllers walk the same
2 m square so you can see what pose feedback buys you.

## What it demonstrates

`examples/basic_examples/skill_demo.py` is library-native — there is no frozen
script behind it — and it is the file to read first. The default path compiles
a program in the skill grammar and hosts it in a `SingleSkillController`; the
`--open-loop` path runs a hand-written `Controller` on a clock. The printed
distance from the origin after the square is the comparison.

-   [`domo.sim`](../api/sim.md) for the engine and scene,
    [`domo.robot`](../api/robot.md) for the Go2, `domo.checkpoints` for the
    policy.
-   [`domo.skills`](../api/skills.md): `make_go2_library(policy_fn)` and
    `library.compile(...)`.
-   [`domo.control`](../api/control.md): `SimControlLoop`,
    `SingleSkillController`, `Controller`, `CPGLocomotionSkill`, `StandSkill`.
-   No sensors and no task: flat ground, one environment.

The program it authors is the shape the M5 planner emits:

```
(goto(x=2, y=0) @ walk >> goto(x=2, y=2) @ walk >>
 goto(x=0, y=2) @ walk >> goto(x=0, y=0) @ walk) | stand.for(2)
```

`goto @ walk` closes the loop on the robot's pose every tick, so each leg is a
straight line and the program terminates itself when the last corner is
reached. The `| stand.for(2)` fallback runs if the route fails — a tipped or
fallen walk. See [the skill grammar](../concepts/grammar.md) for what `@`,
`>>` and `|` mean.

The `--open-loop` contrast is `PatrolController`: 4 s of forward walking, 2 s
of turning, repeat, with a reflex that switches to `StandSkill` when roll or
pitch exceeds 0.5 rad. It has no pose feedback, so velocity error integrates
and the square smears.

## Run it

=== "CPU (laptop)"

    ```bash
    # the smoke test: short run, no viewer
    python examples/basic_examples/skill_demo.py policies/walk.pt \
        --headless --device cpu --steps 300

    # the full square
    python examples/basic_examples/skill_demo.py policies/walk.pt \
        --headless --device cpu
    ```

=== "GPU with the viewer"

    ```bash
    python examples/basic_examples/skill_demo.py policies/walk.pt
    ```

=== "The open-loop contrast"

    ```bash
    python examples/basic_examples/skill_demo.py policies/walk.pt --open-loop
    ```

## How it works

-   `parse_args()` takes the checkpoint positionally, then `--open-loop`,
    `--steps`, `--headless` and `--device`; `pick_device` resolves the device.
-   `build_world(device, headless, dt)` calls `create_engine("genesis")`,
    creates a scene with `dt = 0.02` and flat ground, places one Go2 at
    `(0, 0, 0.35)` with `kp = 100`, `kd = 2`, then builds and binds a single
    environment.
-   `build_controller(policy_fn, device, open_loop, dt)` returns either the
    `PatrolController` over `CPGLocomotionSkill` + `StandSkill`, or the
    compiled program in a `SingleSkillController` — plus a label for the
    console.
-   `main()` loads the policy with `load_locomotion_policy`, calls
    `controller.setup(robot)`, builds `SimControlLoop(scene, robot,
    controller, dt=0.02)`, resets it and runs `--steps` steps with a callback
    that prints every 250.
-   `print_summary(controller, final, open_loop)` prints the program's
    execution `trace` when the composition finished — that trace is the log
    the supervisor reads in the [twin](twin-demo.md) — then the final position
    and its distance from the origin.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `checkpoint` (positional) | required | CPG locomotion checkpoint |
| `--open-loop` | off | run the timed `PatrolController` (drifts) instead of the closed-loop navigation composition |
| `--steps` | `3000` | control steps to run (60 s of simulated time) |
| `--headless` | off | no Genesis viewer |
| `--device` | `cuda` | torch/Genesis device |

## What you will see

```
  Frozen locomotion policy: policies/walk.pt
    obs=... act=12 params=...
Running composition: (goto(x=2, y=0) @ walk >> ... ) | stand.for(2)
  step   250 | skill=...              | pos=(+1.10,+0.02) vx=+0.48
  ...
Program trace:
    ...
Done. final pos=(+0.05,-0.03)  |dist from origin|=0.06 m
```

With `--open-loop` the label is `Running open-loop PatrolController`, the
reflex prints `[ctrl] tipping — switching to stand` and `[ctrl] recovered —
resuming walk` when it fires, there is no trace, and the final distance is
noticeably larger.

Nothing is saved.

## Runtime

| | Build | 3000 steps |
|---|---|---|
| CPU | about 30 s | about 1 min |
| GPU | a few seconds | seconds |

The 300-step smoke run finishes in well under a minute on CPU.

## Known limitations

-   Single environment, flat ground, no sensors: the point is the control
    workflow, not the scene. For sensors and a planner, go to the
    [twin demo](twin-demo.md).
-   The composition may finish before `--steps` runs out. The loop keeps
    stepping — the program idles on its last node — until the budget is spent.
-   The open-loop square is *meant* to drift; there is no metric beyond the
    distance printed at the end.
