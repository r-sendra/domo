# Voice commands

Speak at the robot and it walks. A background thread listens on the
microphone, transcribes with Whisper, asks Gemini to turn the sentence into a
body velocity, and writes the result where the control loop can read it. This
is the thinnest possible human-robot channel: no planner, no grammar, one
command at a time.

## What it demonstrates

`examples/hri/go2_cpg_rl_voice.py` is the library replica of the interactive
part of `scripts/house_scene/go2_cpg_rl_voice.py`. The original duplicated the
whole PPO trainer; that is gone — train the gait with
[the locomotion example](locomotion.md) instead.

-   [`domo.hri`](../api/llm-hri.md#the-voice-module-domohri):
    `VoiceCommander` and `CommandState`. The commander owns the thread, the
    audio device, the Whisper model and the LLM call.
-   [`domo.tasks`](../api/tasks.md#go2cpgwalktask): a single-environment
    `Go2CPGWalkTask` used as a stepping harness — 1 000 000-step episodes and
    no command resampling, so the voice thread is the only source of commands.
-   `domo.checkpoints` for the frozen walk policy.

The division of labour is the interesting part: `CommandState` is the only
shared object, the sim never blocks on transcription, and the commander never
touches the robot.

## Run it

=== "Setup"

    ```bash
    pip install openai-whisper sounddevice google-genai
    # or: pip install -e '.[voice,llm]'
    export GEMINI_API_KEY=<key>
    ```

=== "With the viewer"

    ```bash
    python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt \
        --whisper-model tiny
    ```

=== "No display"

    ```bash
    python examples/hri/go2_cpg_rl_voice.py --eval policies/walk.pt --headless
    ```

## How it works

-   `build_env(device, headless)` creates the task with one environment,
    `max_episode_steps = 1_000_000` and `resampling_time_s = 1e9`, so it runs
    until interrupted and never overwrites the command.
-   `run_voice_eval(checkpoint_path, whisper_model="tiny", initial_vx=0.0,
    headless=False)` loads the policy, creates a `CommandState(vx=initial_vx)`,
    starts the `VoiceCommander` thread, runs `drive` until ++ctrl+c++, and
    stops the commander in a `finally` block.
-   `drive(env, policy_fn, state)` steps forever: read `state.get()`, copy the
    triple into `env.commands`, query the frozen policy, step the task, reset
    when the robot falls, print every 250 steps.
-   The device is picked with `pick_device("cuda")` — CUDA when available,
    else CPU — independently of any flag, as in the frozen script.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--eval` | required | CPG locomotion checkpoint to drive |
| `--whisper-model` | `tiny` | one of `tiny`, `base`, `small` |
| `--vx` | `0.0` | initial forward speed before the first voice command (m/s) |
| `--headless` | off | no Genesis viewer |

There is no `--device`: it is `cuda` when available, else `cpu`.

## What you will see

The `VoiceCommander`'s own transcription and command messages, then a status
line every 250 steps and a note whenever the robot falls:

```
  [sim] cmd=(0.50,0.00,0.00)  vx=+0.48  h=0.31
  [sim] robot fell — resetting
```

++ctrl+c++ prints `Stopping.` and shuts the commander down. Nothing is saved.

## Runtime

Runs until interrupted. The first run downloads the Whisper model, which is
the only slow part of startup.

## Known limitations

!!! warning "Not smoke-testable"

    This is the one example with no offline path. It needs a working
    microphone, the `voice` extra, `google-genai` and `GEMINI_API_KEY`, so it
    cannot run in CI or on a headless box without an audio device. Every other
    example has a CPU command that runs from a clean checkout plus the two
    stable policies.

-   Commands arrive with the latency of transcription plus one LLM call, so
    the robot keeps doing the previous thing for a second or so.
-   Training was removed from this replica. Use
    [the locomotion example](locomotion.md) to produce the checkpoint you pass
    to `--eval`.

For a typed rather than spoken command channel — with the same rule-based and
Gemini parsers, and no microphone — see
[waypoint navigation](evaluate-nav.md).
