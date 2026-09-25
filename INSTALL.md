# Installing DOMO

The full installation guide lives in [docs/start/install.md](docs/start/install.md),
or in the rendered documentation site (`pip install -e '.[docs]'` then
`mkdocs serve`).

Short version:

```bash
conda create -n domo python=3.12 -y
conda activate domo
pip install -e '.[genesis,dev]'   # + '.[rl]' '.[llm]' '.[langchain]' '.[voice]' '.[docs]'
pytest tests/                    # engine-free layers, ~10 s
```
