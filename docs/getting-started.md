# Getting started

This guide takes you from an empty machine to a running digital twin and a
first training run. Everything here was verified on macOS (CPU) and Linux
(CUDA); Windows is untested.

## 1. Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python    | 3.10    | 3.12 is what the project is developed on |
| PyTorch   | 2.x     | CPU build is enough for tests, demos and smoke runs |
| Genesis   | 1.0.0   | the only physics backend today (`pip install genesis-world`) |
| GPU       | none    | needed only for RL training at scale (thousands of envs) |

Genesis builds and steps scenes on CPU. A Go2 arena scene builds in about
30 s on a laptop, which is enough to run every example with a handful of
environments. Training a locomotion policy from scratch needs a CUDA GPU
(2048–4096 environments, tens of millions of steps).

## 2. Create the environment

```bash
conda create -n domo python=3.12 -y
conda activate domo

# core
pip install torch numpy genesis-world

# the library itself, editable, with the dev tools (pytest, ruff)
pip install -e '.[dev]'
```

Optional extras, install what you need:

| Extra | Installs | Enables |
|-------|----------|---------|
| `rl`  | tensorboard | training curves for `main.py` and the RL examples |
| `llm` | google-genai | Gemini as the LLM behind Eureka / the planners |
| `langchain` | langgraph, langchain-core, langchain-openai | local models via vLLM, OpenAI, and the LangGraph Eureka driver |
| `gemini-langchain` | langchain-google-genai | Gemini through LangChain |
| `voice` | openai-whisper, sounddevice | the voice-command example |

```bash
pip install -e '.[dev,rl,llm,langchain]'
```

Everything LLM-related is imported lazily: the core library, the tests and
all simulation examples run without any of these extras.

## 3. Verify the installation

```bash
pytest tests/          # pure-math and engine-free layers, a few seconds
python -c "import domo, genesis; print(domo.__name__, 'ok')"
```

If `import genesis` fails on macOS, check that you installed `genesis-world`
(the PyPI name) and not a different package called `genesis`.

## 4. First run: the digital twin on CPU

The entry point of the system is the goal-free `World`: a robot in a scene
with its sensors, governed by whatever controller you attach. The simplest
example builds one and runs a hand-written controller that switches between
standing and walking:

```bash
python examples/basic_examples/skill_demo.py policies/walk.pt --headless --device cpu --steps 500
```

Drop `--headless` to open the Genesis viewer. Every example accepts
`--device cpu`; on a CUDA machine the default is `cuda`.

The positional argument is the trained gait. Blessed checkpoints live in the
`policies/` folder (`walk.pt`, `avoid.pt`) and are resolved by symbolic name
through `domo.policies`; the SLAM examples default to them, the others take
the path explicitly. Those files are not versioned in git: copy them from a
training run or from a colleague (see [running.md](running.md#stable-policies)).
Without them, the twin, SLAM and navigation examples cannot walk.

## 5. First training run (GPU)

```bash
python main.py --n-envs 4096 --device cuda --headless
python main.py --eval runs/go2_walk/checkpoint_final.pt
```

`main.py` trains the joint-space velocity-tracking task; the CPG-RL gait used
by the rest of the system is trained with `examples/locomotion/go2_cpg_rl.py`.
See [running.md](running.md) for budgets, resuming and evaluation.

## 6. Where to go next

* [project-overview.md](project-overview.md): what DOMO is and how the repository is organised.
* [architecture.md](architecture.md): the layer stack and the rules that keep it engine-agnostic.
* [examples.md](examples.md): every example, what it demonstrates and how to run it.
* [api/](api/README.md): module-by-module reference.
* [extending.md](extending.md): adding a task, a skill, a sensor, a physics backend, an LLM provider.
* [troubleshooting.md](troubleshooting.md): known Genesis quirks and common errors.
