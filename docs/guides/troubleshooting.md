# Troubleshooting

Every failure the project has hit twice, written as symptom, cause and fix.
Most of them are properties of Genesis 1.0.0, of the bundled Go2 asset or
of the optional-dependency layout rather than bugs, so the fix is usually a
flag or an install and not a patch. Keep the list current: an
undocumented quirk costs the next person an afternoon.

## Installation

### `ModuleNotFoundError: No module named 'genesis'`

**Cause.** The distribution is called `genesis-world` and the import name
is `genesis`. There is an unrelated PyPI package called `genesis`, and
`pip install genesis` installs that one.

**Fix.** `pip install -e '.[genesis]'` from the repository root, which
pulls `genesis-world`. If you already installed the wrong one,
`pip uninstall genesis` first.

### `ModuleNotFoundError: tensorboard` from something that is not training

**Cause.** `domo/rl/ppo.py` does `from torch.utils.tensorboard import
SummaryWriter` at module level, and `domo/rl/__init__.py` imports it. So
`import domo.rl` fails outright without tensorboard — and that failure
propagates further than it looks, because `domo.checkpoints` imports
`domo.rl` at module level too. Loading a checkpoint for pure inference, on
a laptop that will never train anything, still needs the extra.

**Fix.** `pip install -e '.[rl]'`.

Three levels of exposure, if you are trying to keep an environment thin:

| Call | Needs tensorboard |
|------|-------------------|
| `domo.policies.stable_policy("walk")` | no — it only resolves a path, and the module keeps torch out on purpose |
| `domo.policies.load_stable_locomotion(...)`, `load_stable_avoid(...)`, `stable_go2_library(...)` | yes — they import `domo.checkpoints` and `domo.rl` inside the function |
| anything in `domo.checkpoints` or `domo.rl` | yes |

### `mkdocs: command not found`

**Cause.** The documentation toolchain is not a project dependency. It is
the `docs` extra (`mkdocs-material`, `mkdocs-glightbox`), so a normal
install does not have it.

**Fix.** `pip install -e '.[docs]'`, then build with
`mkdocs build --strict`. Do not use `mkdocs serve` in a scripted session —
it blocks until interrupted.

## Genesis

### Quaternions look wrong, or the robot spawns facing backwards

**Cause.** Genesis is `wxyz`. A `(0, 0, 0, 1)` "identity" borrowed from an
`xyzw` code base is a 180° yaw flip.

**Fix.** Use `domo.utils.rotations` for every conversion. All of its
operations are tested against `genesis.utils.geom` in
`tests/test_rotations.py`.

### Euler angles do not match what you expected

**Cause.** Genesis' `quat_to_xyz` with `rpy=False` — the default the
original scripts used — is intrinsic X-Y-Z, not aerospace roll-pitch-yaw.

**Fix.** `quat_to_euler_xyz` replicates it exactly. Do not substitute a
generic RPY conversion.

### `AttributeError: 'RigidJoint' object has no attribute 'dof_idx_local'`

**Cause.** Deprecated in Genesis 1.0.0.

**Fix.** Use `joint.dofs_idx_local`, which returns a list even for a
one-DOF joint; flatten it.

### `VisOptions(n_rendered_envs=1)` raises or is ignored

**Cause.** Also deprecated.

**Fix.** `rendered_envs_idx=[0]`.

??? warning "`FR_foot` does not exist, and what that costs"

    **Symptom.** `link_indices(["FR_foot", ...])` fails, or
    `state.foot_contacts` stays all zeros.

    **Cause.** The Go2 URDF bundled with Genesis merges the fixed foot
    links into the calves, so the foot links are simply not in the model.
    `SimContactSensor` probes `get_link_contact_forces()` at construction,
    finds nothing usable, sets `available = False` and writes nothing.

    **Consequence.** Force-based foot contact is unavailable on this
    asset, so the CPG substitutes a **stance-phase proxy**:
    `CPGOscillators.stance_mask()` reports a leg as in contact whenever
    `sin(theta) < 0`. That is a kinematic prediction, not a measurement —
    it is right about the gait cycle and wrong about the ground. A reward
    term or a skill that needs to know the foot actually touched
    something cannot be written against this asset; swap in a URDF with
    separate foot links first.

    The same asset prints a benign `qpos0 exceeds joint limits` warning at
    build time. Ignore it.

### PD-gain randomisation is not per environment

**Symptom.** You randomise `kp_scale` / `kd_scale` per env and every env in
a reset batch comes out with the same gains.

**Cause.** Genesis' `set_dofs_kp` accepts at most one-dimensional gains, so
there is no per-env axis to write into. Domain randomisation of PD gains is
therefore applied **per reset group** — every environment reset together
shares one draw.

**Fix.** None available on this backend; treat it as a known limitation
when interpreting a DR sweep. Friction and base mass *are* per env, so a
sweep over those is honest. `set_pd_gains_scaled` is an optional
`Articulation` capability, and another backend may implement it properly.

??? danger "Lidar sectors are scrambled — and older avoidance numbers are not comparable"

    **Symptom.** The avoidance policy dodges in the wrong direction, or a
    lidar panel shows obstacles at plausible distances but implausible
    bearings.

    **Cause.** Genesis' `SphericalPattern(n_points=(n_h, n_v))` returns
    distances as `[N, n_horizontal, n_vertical]` — **azimuth-major**, one
    row per azimuth with that azimuth's vertical rays contiguous. An
    earlier `read_ranges` assumed `[N, n_vertical, n_horizontal]` and
    `view()`ed the buffer accordingly, so each "sector minimum" was taken
    over a mix of unrelated beams.

    **Fix.** Every reader now goes through
    `lidar_ranges_to_grid(raw, n_v, n_h, azimuth_major=True)`, and
    `read_sector_distances()` is the minimum over `dim=1` of the
    normalised grid. This matches the pooling the original avoidance
    scripts did on the contiguous Genesis buffer, which is pinned by
    `tests/test_foundation_sim.py::test_sector_minimum_matches_script_pooling`.

    **The part that is not a code fix.** Any avoidance evaluation recorded
    before this change was measured against a scrambled observation. Those
    success counts are not comparable with anything measured after it — do
    not put them in the same table. Re-run the evaluation instead.

    `read_points()` deliberately stays in Genesis' native flattened order
    (`[N, n_h × n_v, ...]`) and returns its own ranges, so points and
    ranges stay paired. Never index it with the `read_ranges()` grid.

### Ctrl+C does nothing during a build or a long step

**Cause.** `scene.build()` and `scene.step()` are long C calls that hold
the GIL, so Python defers the `KeyboardInterrupt` until they return.

**Fix.** The examples that host the dashboard set
`signal.signal(signal.SIGINT, signal.SIG_DFL)` at the top of `main()`, so
the OS kills the process immediately. Nothing is flushed that way; the
dashboard's Stop button is the graceful path.

### The ReplicaCAD house takes minutes to build

**Cause.** A ReplicaCAD apartment loads about 114 meshes. Expect around two
minutes on CPU, tens of seconds on a GPU — per process, and the Eureka
workers are separate processes.

**Fix.** Nothing to fix; plan for it. Run those examples in the background
with a generous timeout rather than waiting on them in the foreground, and
prefer the arena for anything you are iterating on.

### Adding a camera or a lidar raises inside Genesis

**Cause.** `Scene.add_camera` and `Scene.add_lidar` are build-time
operations, like every other `add_*`.

**Fix.** Add every entity and sensor before `scene.build(n_envs)`. After
the build you may only move things.

## Checkpoints

### "Unexpected key" or a shape mismatch when loading a `.pt`

**Cause.** Two checkpoint formats coexist: the library's (config
dictionaries plus weights) and the frozen scripts' (one flat `config` and
weights, sometimes with `_orig_mod.` or `module.` prefixes on the keys).
Code that calls `load_state_dict` directly on a raw `model_state` will trip
over either the prefixes or the architecture.

**Fix.** Go through the library: `ActorCritic.from_state_dict` rebuilds the
network from the weight shapes and `clean_state_dict` strips the prefixes,
while `domo.checkpoints.configs_from_checkpoint` recovers the configs from
either format ([the two formats](running.md#the-two-formats)). If it still
fails, check you are loading the right task's checkpoint — the CPG gait
expects 76-dimensional observations, the avoidance net 36.

### `FileNotFoundError: stable policy 'walk' missing at .../policies/walk.pt`

**Cause.** Blessed checkpoints are not versioned. `policies/` holds a
`README.md` and a registry entry for each name, and the `.pt` files are
git-ignored, so a fresh clone has the names without the weights.

**Fix.** Copy one from a training run, or get it from a colleague:

```bash
cp runs/go2_cpg/checkpoint_final_coupled.pt policies/walk.pt
```

`policies/README.md` says what each name is meant to be; see
[stable policies](running.md#stable-policies) before promoting anything.

### `torch.load` refuses to unpickle

**Cause.** Checkpoints carry pickled config dataclasses, so
`domo.checkpoints.load_checkpoint` passes `weights_only=False`. Newer
PyTorch defaults to `weights_only=True` and will refuse a checkpoint loaded
by hand.

**Fix.** Use `domo.checkpoints.load_checkpoint`, and only load checkpoints
you trust — `weights_only=False` executes pickled code.

## Training

### The run finishes immediately and writes only `checkpoint_final.pt`

**Cause.** The trainer runs `total_steps // (rollout_steps × n_envs)`
updates. With `--total-steps` smaller than one rollout that integer
division is zero, so the loop body never executes and `train()` goes
straight to the final save.

**Fix.** Give a smoke run at least a few rollouts: at `--n-envs 8` and the
default 24 rollout steps, one rollout is 192 steps, so `--total-steps 4000`
is about twenty updates.

### `--resume` trains far past the budget you asked for

**Cause.** `PPOTrainer.train()` computes `n_updates` once, from
`cfg.total_steps`, and never subtracts the restored `global_step`. The
resumed run therefore performs a *full* budget of updates on top of
whatever the checkpoint already had.

**Fix.** Lower `--total-steps` by hand to the amount of extra training you
want. The step counter itself continues from the checkpoint, so file names
never collide and the TensorBoard curves join up correctly. See
[resuming](running.md#resuming).

### `go2_cpg_rl.py --eval` opens a viewer on a display-less machine

**Cause.** The script parses `--headless` but does not forward it to
`evaluate()`, which builds a windowed scene unconditionally. The flag is
honoured for training and ignored for evaluation.

**Fix.** Evaluate that gait through
`examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu`,
which loads the same checkpoint and is headless-clean. The two avoidance
scripts do forward `--headless`.

### The SLAM map is empty, or resets on every leg of a mission

**Cause.** `SlamSkill.reset_idx` wipes the occupancy grid, the odometry
seed and the accumulated cloud. `LayerNode.enter()` calls `reset_idx` when
a layer starts, so a program like `slam @ goto(x=2) @ walk >> slam @
goto(x=0) @ walk` builds a fresh map for each leg and throws away the
previous one.

**Fix.** Host SLAM persistently: keep one `SlamSkill` instance in your
controller and tick it yourself every step, instead of letting a
composition own it. Layer it into a program only when you genuinely want
per-leg mapping.

### A run diverges: NaN losses, or the value loss explodes

**Cause.** The `PPOConfig` stability guards are off by default, because the
defaults reproduce the original trainer exactly.

**Fix.** Turn them on in the script's `build_configs`, not on the command
line (they are not exposed as flags): `guard_nonfinite=True` rolls an
update back atomically when a loss or gradient norm is non-finite,
`vloss_skip=<bound>` does the same for a pathological value loss,
`target_kl` early-stops the epochs, and `lr_schedule="linear"` decays the
learning rate. Watch `loss/value` and `train/clip_fraction` to decide which
one you need ([monitoring](running.md#monitoring)).

## LLM providers

### Empty or truncated replies from Gemini

**Cause.** `gemini-2.5-flash` is a thinking model: reasoning tokens count
against `max_output_tokens`, so a long chain of thought can consume the
entire budget and leave no output.

**Fix.** The default is 16 384; raise it for complex prompts (the model
caps at 65 536). Truncated code fences are salvaged by
`extract_code_block`, and a warning is printed when a reply hit the token
limit — if you see that warning repeatedly, raise the budget rather than
retrying.

### "import not allowed" when compiling a generated reward

**Cause.** The reward validator allows `torch` and `math`, and nothing
else. This is deliberate: generated code runs in the training process.

**Fix.** Nothing to fix in the validator. If a reward genuinely needs
another quantity, expose it on the task and list it in the `TaskSpec`'s
`env_interface` so the model is told it exists.

### vLLM is not reachable

**Cause.** The `vllm` provider talks to an OpenAI-compatible server that
you start yourself.

**Fix.** `vllm serve MODEL`, then point `VLLM_BASE_URL` at it (the default
is `http://localhost:8000/v1`). `--llm-base-url` on the Eureka example
overrides it per run.

### A DrEureka run reports a DR config that randomises nothing

**Cause.** Proposals are clamped into the RAPP feasible bounds, and
clamping a range into a bound with no room yields a zero-width range such
as `[1.0, 1.0]`. That trains with the parameter pinned at its default
instead of dropping it, so the "randomised" retraining is a no-op for that
parameter. The fallback path has a related property: it always adds
`obs_noise_std`, possibly `0.0`, so Stage 3 always retrains at least once.

**Fix.** Read `skill.dr_config` before believing a robustness number, and
widen the sweep grids in `DrEurekaConfig` if every bound came back tight —
a collapsed bound usually means the policy was fragile, not that the model
proposed badly ([DrEureka](../api/eureka.md#dreureka-domoeurekadr)).

## Dashboard

### The page loads but shows no data

**Cause.** Either nothing is publishing, or the 3D panel failed alone. The
telemetry code runs in a plain `<script>` independent of the three.js
module, so a CDN failure only darkens the 3D panel and the numbers keep
updating.

**Fix.** If *everything* is dark, the simulation is not publishing: check
that the example was launched with `--dashboard PORT` or `--dashboard-url
URL`. A reconnecting page also sees only the latest snapshot — there is no
history, so a paused or finished run shows one frame and no charts.

### The 3D robot does not render

**Cause.** The viewer loads three.js and the URDF loader from a CDN and
needs internet access. The Go2 COLLADA meshes are tens of megabytes.

**Fix.** Give the first load a few seconds. Offline, expect the telemetry
panels to work and the 3D panel not to. Any change to the scene needs a
page reload; the manifest is fetched once per page load.

### `python -m domo.dashboard` raises `NameError`

**Cause.** The module is executed top to bottom and `make_server` reads the
embedded HTML constant, so the `if __name__ == "__main__":` guard must be
the **last** statement in the file, after `_DASHBOARD_HTML`.

**Fix.** Move the guard back to the end.
`test_html_constant_precedes_main_guard` enforces it, and also checks that
the page body contains no `"""` because it lives in an `r"""` literal.

## Tests

### `tests/test_lidar_models.py` fails with `AttributeError: read_points`

**Cause.** `SimulatedLidar` used to call `read_points()` unconditionally,
and `read_points` is an *optional* `LidarSensorHandle` capability.

**Fix.** Already fixed: `SimulatedLidar` probes for the method and falls
back to sector mode. A fake handle in a test needs only `read_ranges` and
`read_sector_distances`.

### A test imports Genesis

**Cause.** Someone tested a layer that should have been engine-free, or
wrote a fake that reached for a real handle.

**Fix.** It must not happen. The suite covers the engine-free layers only —
that is what makes it 228 tests in about ten seconds and runnable on any
laptop. Anything that needs physics is a CPU smoke run of an example with
`--n-envs 2`, not a unit test. `tests/test_foundation_world.py` shows how
far a fake backend gets you: a complete `World` with no physics at all.

## Still stuck

Reduce it to something that fails in seconds, then say exactly what you
ran.

A minimal reproduction is CPU, tiny, headless and seeded:

```bash
python - <<'PY'
import numpy as np, torch
torch.manual_seed(0); np.random.seed(0)     # (1)!
from domo.tasks import Go2CPGWalkConfig, Go2CPGWalkTask
task = Go2CPGWalkTask(Go2CPGWalkConfig(n_envs=2, device="cpu", headless=True))
obs, _ = task.reset()
for _ in range(10):
    obs, _, rew, reset, extras = task.step(torch.zeros(2, task.num_actions))
print(obs.shape, rew, reset)
PY
```

1.  Both RNGs: torch for commands, spawns and obstacle scatter, numpy for
    rough-terrain heightfields generated inside Genesis. See
    [reproducibility](running.md#reproducibility).

Two environments is enough to catch a broadcasting bug that one env hides,
and ten steps is enough to catch a wiring bug. Keep `headless=True` so the
report does not depend on a display.

Then run the tests that cover the layer you touched — the whole suite is
about ten seconds, so there is rarely a reason not to run all of it:

```bash
pytest tests/ -q                       # everything, ~10 s
pytest tests/test_tasks_common.py -q   # task configs, reward keys, observation layouts
pytest tests/test_foundation_world.py -q   # engine → scene → robot → lidar wiring
pytest tests/test_skills_library.py tests/test_skill_grammar.py -q   # cards, compilation, traces
pytest tests/test_rl_ppo.py -q         # the trainer loop and checkpoint round-trips
```

Include, in this order: the exact command, the Python, torch and Genesis
versions, the device, whether it reproduces at `--n-envs 2` on CPU, and the
full traceback. If it only fails at 4096 envs on CUDA, say so — that is a
different class of bug from anything above.
