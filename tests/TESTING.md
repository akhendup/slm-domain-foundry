# Testing policy

**Start here for hardware:** README [Hardware & platforms](../README.md#hardware--platforms) · this file for pytest commands.

**AI agents:** Follow the step-by-step roadmap in [`AI_COVERAGE_IMPLEMENTATION.md`](AI_COVERAGE_IMPLEMENTATION.md) (environments, phases A–E, Apple Silicon vs GPU, progress log).

## Principles

1. **Real I/O** — Tests use `sample_data/`, `tmp_path`, and real HuggingFace tiny checkpoints (`sshleifer/tiny-gpt2`), not `unittest.mock`.
2. **Simulated data** — Synthetic rows are built through the same code paths as production (e.g. `data.prepare_training_data`, `log_interaction`).
3. **Markers** — `unit`, `quality`, `e2e`, `app`, `real`, `slow`, `gpu`, `mps`.
4. **Coverage** — CI gate **75%** on `app/`, `data/`, and `train/` (see `.github/workflows/tests.yml`). Long-term goal remains higher coverage on core modules (see `AI_COVERAGE_IMPLEMENTATION.md`).

## Hardware matrix

> **Docker on Mac:** CPU-only inside containers — **no MPS**. Use native `requirements-mps.txt` / `run_local.sh` for Apple Silicon GPU work.

| Environment | Marker / file | Install | Command |
|-------------|---------------|---------|---------|
| **CPU only** | default suite (no `gpu` / `mps`) | `requirements.txt` or `requirements-mps.txt` | `pytest tests/ --ignore=tests/real/test_apple_silicon_mps.py --ignore=tests/real/test_cuda_gpu.py` |
| **Apple Silicon (MPS)** | `mps` → `tests/real/test_apple_silicon_mps.py` | `requirements-mps.txt` | `pytest tests/real/test_apple_silicon_mps.py -m mps` |
| **NVIDIA CUDA** | `gpu` → `tests/real/test_cuda_gpu.py` | `requirements.txt` (+ optional `unsloth`) | `pytest tests/real/test_cuda_gpu.py -m gpu` |

Platform-specific files **skip cleanly** when the required device is unavailable, so one checkout runs everywhere.

## Apple Silicon (MPS)

Native Mac install uses **`requirements-mps.txt`** (see README and `run_local.sh`):

```bash
pip install -r requirements-mps.txt
pytest tests/real/test_apple_silicon_mps.py -v -m mps
```

A **macOS** job in `.github/workflows/tests.yml` runs this file on every push/PR.

## NVIDIA CUDA (remote workstation)

Use **`scripts/run_tests_amdworkstation.sh`** to rsync the repo to `amdworkstation` and run CUDA tests:

```bash
./scripts/run_tests_amdworkstation.sh           # GPU tests only
./scripts/run_tests_amdworkstation.sh --full    # full suite minus MPS file
./scripts/run_tests_amdworkstation.sh --sync-only
```

Override host/path: `AMDWORKSTATION_HOST`, `AMDWORKSTATION_DIR`.

Local CUDA (when `nvidia-smi` works on the same machine):

```bash
pip install -r requirements.txt
pytest tests/real/test_cuda_gpu.py -v -m gpu
```

## Running tests

```bash
# Full suite with coverage gate (excludes platform-specific files)
pytest tests/ \
  --ignore=tests/real/test_apple_silicon_mps.py \
  --ignore=tests/real/test_cuda_gpu.py \
  --cov=app --cov=data --cov=train --cov-report=term-missing

# Real integration only (models + sample_data)
pytest tests/real/ -m real

# Fast unit + quality (no slow train)
pytest tests/ -m "unit or quality" --ignore=tests/real
```

## Live services

- **Ollama** — `tests/real/test_ollama_live.py` runs when `localhost:11434` is up.
- **GPU / Unsloth** — `@pytest.mark.gpu` in `tests/real/test_cuda_gpu.py`; optional Unsloth install on CUDA hosts.
