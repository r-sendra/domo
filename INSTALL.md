# Installing DOMO

The full installation guide lives in [docs/getting-started.md](docs/getting-started.md).

Short version:

```bash
conda create -n domo python=3.12 -y
conda activate domo
pip install torch numpy genesis-world
pip install -e '.[dev]'          # + '.[rl]' '.[llm]' '.[langchain]' '.[voice]' as needed
pytest tests/                    # engine-free layers, ~5 s
```
