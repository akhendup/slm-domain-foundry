# Testing policy

**AI agents:** Follow the step-by-step roadmap in [`AI_COVERAGE_IMPLEMENTATION.md`](AI_COVERAGE_IMPLEMENTATION.md) (environments, phases A–E, Apple Silicon vs GPU, progress log).

## Principles

1. **Real I/O** — Tests use `sample_data/`, `tmp_path`, and real HuggingFace tiny checkpoints (`sshleifer/tiny-gpt2`), not `unittest.mock`.
2. **Simulated data** — Synthetic rows are built through the same code paths as production (e.g. `data.prepare_training_data`, `log_interaction`).
3. **Markers** — `unit`, `quality`, `e2e`, `app`, `real`, `slow`, `gpu`.
4. **Coverage** — Target **100%** line coverage on `app/`, `data/`, and `train/` (see `.coveragerc`).

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
