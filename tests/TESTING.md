# Testing policy

**AI agents:** Follow the step-by-step roadmap in [`AI_COVERAGE_IMPLEMENTATION.md`](AI_COVERAGE_IMPLEMENTATION.md) (environments, phases A–E, Apple Silicon vs GPU, progress log).

## Principles

1. **Real I/O** — Tests use `sample_data/`, `tmp_path`, and real HuggingFace tiny checkpoints (`sshleifer/tiny-gpt2`), not `unittest.mock`.
2. **Simulated data** — Synthetic rows are built through the same code paths as production (e.g. `data.prepare_training_data`, `log_interaction`).
3. **Markers** — `unit`, `quality`, `e2e`, `app`, `real`, `slow`, `gpu`.
4. **Coverage** — CI gate **75%** on `app/`, `data/`, and `train/` (see `.gitea/workflows/tests.yml`). Long-term goal remains higher coverage on core modules (see `AI_COVERAGE_IMPLEMENTATION.md`).

## Apple Silicon (MPS)

Native Mac install uses **`requirements-mps.txt`** (see README and `run_local.sh`). Run the dedicated MPS suite on a Mac with Metal enabled:

```bash
pip install -r requirements-mps.txt
pytest tests/real/test_apple_silicon_mps.py -v -m mps
```

On Linux/CI without MPS, these tests skip cleanly. A **macOS** job in `.gitea/workflows/tests.yml` runs this file on every push/PR.

## Running tests

```bash
# Full suite with coverage gate
pytest tests/ --cov=app --cov=data --cov=train --cov-report=term-missing

# Real integration only (models + sample_data)
pytest tests/real/ -m real

# Fast unit + quality (no slow train)
pytest tests/ -m "unit or quality" --ignore=tests/real
```

## Live services

- **Ollama** — `tests/real/test_ollama_live.py` runs when `localhost:11434` is up.
- **GPU / Unsloth** — mark `@pytest.mark.gpu` for CUDA-only training smoke tests.
