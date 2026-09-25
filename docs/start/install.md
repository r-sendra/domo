# Install

How to get a working DOMO environment: the requirements, the one command
that installs the library, the extras that unlock training and the LLM
providers, and the two checks that tell you it worked.

## Requirements

| Component | Minimum | Notes |
|-----------|---------|-------|
| Python | 3.10 | 3.12 is what the project is developed on |
| PyTorch | 2.x | 2.12 here; the CPU build is enough for the tests, the demos and smoke runs |
| Genesis | 1.0.0 | the only physics backend today. PyPI name `genesis-world`, import name `genesis` |
| GPU | none | CUDA is needed only for RL training at scale (2048–4096 environments) |
| OS | macOS or Linux | verified on macOS (CPU) and Linux (CUDA); Windows is untested |

Genesis builds and steps scenes on CPU, so a laptop runs every demo. A Go2
arena scene takes about 30 s to build and a ReplicaCAD apartment about two
minutes; stepping a single robot then runs at roughly 50–100 control steps
per second. Training a policy from scratch is a different budget entirely and
needs a CUDA GPU — see [running and training](../guides/running.md).

## Create the environment

=== "conda"

    ```bash
    conda create -n domo python=3.12 -y
    conda activate domo
    ```

=== "venv"

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

Both work; conda is the path the project is developed and tested on, and the
one the rest of the documentation assumes when it says `conda activate domo`.

## Install the library

From the repository root, editable, with the physics backend and the dev
tools:

```bash
pip install -e '.[genesis,dev]'   # (1)!
```

1.  The project's own dependencies are just `torch` and `numpy`. The
    `genesis` extra adds `genesis-world`, the `dev` extra adds `pytest` and
    `ruff`.

### Extras

Everything beyond the core is an extra, so a machine that only runs the
simulator never installs an LLM SDK.

| Extra | Installs | Enables |
|-------|----------|---------|
| `genesis` | genesis-world | the Genesis physics backend: every simulation example |
| `rl` | tensorboard | training runs and their curves (`PPOTrainer` imports tensorboard unconditionally) |
| `llm` | google-genai | Gemini as the LLM behind Eureka and the planners |
| `langchain` | langgraph, langchain-core, langchain-openai | local models via vLLM, OpenAI, and the LangGraph Eureka driver |
| `gemini-langchain` | langchain-google-genai | Gemini through LangChain |
| `voice` | openai-whisper, sounddevice | the voice-command example |
| `dev` | pytest, ruff | the test suite and the linter |

Combine what you need:

```bash
pip install -e '.[genesis,dev,rl,llm,langchain]'
```

!!! note "The LLM layers are lazy"

    `domo.llm`, `domo.hri` and `domo.eureka` import their providers inside
    the functions that use them. The core library, the whole test suite and
    every simulation example run with none of `llm`, `langchain`,
    `gemini-langchain` or `voice` installed.

## Verify the installation

```bash
pytest tests/                                        # (1)!
python -c "import domo, genesis; print(domo.__name__, 'ok')"
ruff check domo examples tests main.py               # (2)!
```

1.  228 engine-free tests, about 10 s. They cover the pure-torch layers;
    nothing in `tests/` imports Genesis, by design.
2.  Optional, and only with the `dev` extra. The rules live in
    `pyproject.toml`.

!!! warning "`import genesis` fails"

    Check that you installed `genesis-world` and not the unrelated PyPI
    package called `genesis`. The import name and the distribution name
    differ, and `pip install genesis` gives you the wrong library.

Anything else that goes wrong at this stage — Genesis quirks, checkpoint
load errors, provider problems — is collected in
[troubleshooting](../guides/troubleshooting.md).

## Next step

[Run the robot](first-run.md). One command, one minute, no GPU.
