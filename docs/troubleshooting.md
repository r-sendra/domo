# Troubleshooting

Known quirks of Genesis 1.0.0 and of the tooling, with the workaround the
code base uses. Most of these were discovered the hard way; keep the list
current.

## Genesis

**Quaternions look wrong.** Genesis is `wxyz`. A `(0, 0, 0, 1)` "identity"
is a 180° yaw flip. Use `domo.utils.rotations`; all of its operations are
tested against `genesis.utils.geom` in `tests/test_rotations.py`.

**Euler angles do not match my expectation.** Genesis' `quat_to_xyz` with
`rpy=False` (the default the original scripts used) is intrinsic X-Y-Z, not
aerospace roll-pitch-yaw. `quat_to_euler_xyz` replicates it exactly.

**`joint.dof_idx_local` is deprecated.** Use `joint.dofs_idx_local`, which
returns a list even for one-DOF joints; flatten it.

**`VisOptions(n_rendered_envs=1)` is deprecated.** Use `rendered_envs_idx=[0]`.

**There are no foot links.** The bundled Go2 URDF merges the fixed foot links
into the calves, so `FR_foot` and friends do not exist. Force-based foot
contact is unavailable; the CPG stance-phase proxy is used instead. The same
asset prints a benign "qpos0 exceeds joint limits" warning at build time.

**PD gains cannot be randomised per environment.** `set_dofs_kp` accepts at
most one-dimensional gains, so domain randomisation of PD gains is
per-reset-group, not per-env. Friction and mass are per-env.

**Ctrl+C does nothing during a build or a long step.** `scene.build()` and
`scene.step()` are long C calls that hold the GIL and defer Python's
`KeyboardInterrupt`. Examples that host the dashboard set
`signal.signal(signal.SIGINT, signal.SIG_DFL)` at the top of `main()` so the
OS kills the process immediately; the dashboard's Stop button remains the
graceful path.

**The Replica house takes minutes to build.** ReplicaCAD apartments load
about 114 meshes; expect around two minutes on CPU. Run those examples in the
background and give them a generous timeout.

**A camera must be added before `build()`.** `Scene.add_camera` and
`Scene.add_lidar` are build-time operations. Adding sensors after the build
raises inside Genesis.

## Checkpoints

**"Unexpected key" or shape errors when loading a `.pt`.** Two checkpoint
formats coexist: the library's (config dictionaries plus weights) and the
frozen scripts' (weights only). `ActorCritic.from_state_dict` rebuilds the
network from weight shapes and `domo/checkpoints.py::configs_from_checkpoint`
recovers the configs, so both load. If a load fails, check that the file is
the right task's checkpoint: the CPG gait expects 76-dimensional observations.

**`policies/walk.pt` is missing.** Blessed checkpoints are not versioned.
Copy one from a training run: `cp runs/go2_cpg/checkpoint_final_coupled.pt
policies/walk.pt`. `policies/README.md` lists what each name is.

## LLM providers

**Empty or truncated Gemini replies.** `gemini-2.5-flash` is a thinking
model: reasoning tokens count against `max_output_tokens`. The default is
16 384; raise it further for complex prompts (the model caps at 65 536).
Truncated code fences are salvaged by `extract_code_block`, and a warning is
printed when a reply hit the token limit.

**"import not allowed" when compiling a generated reward.** The validator
allows `torch` and `math` only. Anything else in a generated reward function
is rejected on purpose.

**vLLM is not reachable.** The `vllm` provider talks to an OpenAI-compatible
server. Start it with `vllm serve MODEL` and point `VLLM_BASE_URL` at it
(default `http://localhost:8000/v1`).

## Dashboard

**The page loads but shows no data.** The telemetry code runs in a plain
`<script>` that is independent of the three.js module, so a CDN failure only
darkens the 3D panel. If everything is dark, the sim is not publishing:
check that the example was launched with `--dashboard PORT` or
`--dashboard-url URL`.

**The 3D robot does not render.** The viewer loads three.js and the URDF
loader from a CDN and needs internet access. The Go2 COLLADA meshes are tens
of megabytes; the first load takes a few seconds.

**`python -m domo.dashboard` raises `NameError`.** The
`if __name__ == "__main__":` guard must be the last statement of the module,
after the embedded HTML constant. Keep it there when editing.

## Tests

**`tests/test_lidar_models.py` fails with `AttributeError: read_points`.**
Fixed: `SimulatedLidar` now checks whether the handle exposes `read_points`
and falls back to sector mode. Fake handles need only `read_ranges`.

**Genesis is imported by a test.** It must not be. Tests cover the
engine-free layers only; anything needing physics is a smoke run of an
example on CPU, not a unit test.
