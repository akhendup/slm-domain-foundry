# AI implementation guide: test coverage and real-data testing

**Audience:** AI coding agents, CI maintainers, and human reviewers completing the test/coverage roadmap.

**Goal:** Reach **100% line coverage** on `app/`, `data/`, and `train/` using **real code paths and real or simulated data**—not `unittest.mock`, fake tensors, or hardcoded assertion strings that skip production logic.

**Companion docs:** `tests/TESTING.md` (how to run tests), `.coveragerc` (coverage config), `.gitea/workflows/tests.yml` (CI).

---

## How to use this document

1. Read **Current baseline** and **Execution environments** before changing tests.
2. Work **one phase at a time**; do not skip acceptance criteria.
3. After each step, update the **Progress log** at the bottom (date, agent/human, what changed, why, coverage %).
4. If scope grows, **append new steps** under the phase or add **Phase E+**—do not delete history.
5. Run the **Verification commands** for that phase before marking it done.

---

## Principles (non-negotiable)

| Rule | Meaning |
|------|---------|
| No `unittest.mock` | Do not use `MagicMock`, `patch`, or `monkeypatch` to replace model/tokenizer/network behavior. Setting module paths (`ui._MEMORY_DIR = tmp_path`) is allowed. |
| Real I/O | Prefer `sample_data/`, `tmp_path`, and subprocess or in-process calls to real modules. |
| Simulated data | Data built by production helpers (`prepare_training_data`, `log_interaction`, `save_to_library`) is OK. |
| Tiny real models | Default HF model for tests: `sshleifer/tiny-gpt2` (small, fast, CPU/MPS/CUDA). |
| Live services | Ollama tests run only when `http://localhost:11434` responds; must **skip** cleanly if not. |
| Markers | `@pytest.mark.real`, `slow`, `gpu`, `unit`, `quality`, `e2e`, `app`—use consistently. |

---

## Current baseline (as of 2026-06-13)

**Why this baseline exists:** Phase 1 cleanup and MPS documentation are complete; CI holds at 75% until Gradio UI and CUDA-only Unsloth paths are fully exercised.

| Item | Status |
|------|--------|
| Test count | ~1,207 tests passing (full `pytest tests/`) |
| Overall coverage | ~76% (`app`+`data`+`train`); `data/` ~92%; `app/gradio_ui.py` ~46% |
| CI coverage gate | 75% (`.gitea/workflows/tests.yml`; long-term goal 100% on core modules) |
| Domain fixtures | Medical default in `tests/conftest.py`; no SQL/Teradata test data |
| `tests/real/` | Model load, PEFT, `finetune_cpu`, MPS suite, memory tab, swarm, Ollama live |
| Apple Silicon | `tests/real/test_apple_silicon_mps.py` — 7 tests; macOS CI job |
| Removed | Mock-heavy loader tests; all `sql_vocabulary` / Teradata fixtures |
| Not installed locally | `unsloth` (Unsloth training body not executed on Mac without install + CUDA) |

**Key files already added:**

- `tests/real/conftest.py` — session fixtures: `tiny_lm_dir`, `real_sharegpt_jsonl`, `real_memory_dir`, `ollama_available`
- `tests/conftest.py` — `ollama_available` session fixture
- `app/ollama_client.py`, `app/model_loader.py` (attention mask + template fallback)
- `train/finetune_cpu.py` (`_lora_target_modules`, template fallback)

---

## Execution environments (where tests must run)

Use this matrix when placing or skipping tests. **Apple Silicon is a first-class target** for CPU/MPS paths, not an afterthought.

| Environment | OS / hardware | What to run here | What to skip / mark |
|-------------|---------------|----------------|---------------------|
| **Local dev (Apple Silicon)** | macOS, MPS or CPU | `tests/real/` except `gpu`; `finetune_cpu` integration; `load_model` / `generate_response`; Gradio handler tests (in-process); Ollama if installed | `@pytest.mark.gpu` (CUDA-only Unsloth full train) |
| **Local dev (Linux/Windows CPU)** | CPU only | Same as Mac CPU column | `gpu`, MPS-only assumptions |
| **Local dev (NVIDIA GPU)** | CUDA | Everything including `@pytest.mark.gpu` Unsloth smoke | — |
| **CI default (Gitea/GitHub)** | `ubuntu-latest`, CPU | Full suite minus MPS file on Linux; coverage `--cov-fail-under=75` | `@pytest.mark.gpu`, `slow` optional |
| **CI macOS** | `macos-latest` | `pytest tests/real/test_apple_silicon_mps.py -m mps` | Linux MPS assumptions |
| **CI GPU worker (recommended)** | `ubuntu` + CUDA + `unsloth` | `pytest -m gpu` minimal Unsloth 1-step train on tiny JSONL | Run only in dedicated job |
| **Ollama optional** | Any host with `ollama serve` | `tests/real/test_ollama_live.py`, live branches in `test_ollama_client.py` | `pytest.skip` if `/api/tags` fails |

### Apple Silicon–specific requirements

1. **`train/finetune_cpu`** must remain testable on MPS:
   - Use `--model-name sshleifer/tiny-gpt2`, `--no-eval`, `--epochs 1`, small `--max-seq-length` (128).
   - Do not assume CUDA or `bfloat16` in assertions.
2. **`app/model_loader`** tests must pass on MPS when MPS is available (device auto-detected).
3. **Do not** require Unsloth or CUDA for the default PR CI job on Mac runners.
4. **`tests/real/test_apple_silicon_mps.py`** — implemented; runs on Mac CI and locally when MPS available.

### Phase B clarification: CPU vs GPU vs Unsloth

| Component | Apple Silicon / CPU | NVIDIA CUDA |
|-----------|---------------------|-------------|
| `train/finetune_cpu.py` | **Yes** — primary training test path (`tests/real/test_train_cpu_integration.py`) | Yes (faster) |
| `train/sharegpt_format.py` | **Yes** — tokenizer formatting only | Yes |
| `train/finetune_unsloth.py` **callbacks / CLI guards** | **Yes** — no GPU (`test_finetune_unsloth_callbacks.py`, `test_finetune_unsloth_module.py`) | Yes |
| `train/finetune_unsloth.py` **full `main()` train loop** | **No** — requires `unsloth` + CUDA in practice | **Yes** — `tests/real/test_finetune_unsloth_gpu.py` (to create) |

---

## Prerequisites (what must be provided)

### Repository inputs (already in tree)

- `sample_data/sample_qa.csv` (generic SLM intro Q&A)
- `sample_data/medical_qa.csv` (clinical Q&A default profile)
- `sample_data/patternexamples/` (clinical YAML patterns)
- `requirements.txt` and split requirement files (see README)

### Runtime inputs (agent must ensure)

| Input | Purpose | How to obtain |
|-------|---------|---------------|
| Python 3.12 venv | All tests | `python -m venv .venv && pip install -r requirements.txt` |
| HuggingFace cache | Tiny model + optional base models | First run downloads `sshleifer/tiny-gpt2`; needs network |
| `pytest`, `pytest-cov` | Coverage | `pip install pytest pytest-cov` |
| Ollama (optional) | Local LLM tests | Install Ollama; `ollama pull <model>`; `ollama serve` |
| CUDA + `unsloth` (GPU job only) | Unsloth training coverage | Linux NVIDIA runner; `pip install unsloth` per project Dockerfile.gpu |

### Outputs agents should produce

- New tests under `tests/real/` or `tests/e2e/` (prefer real over new mocks)
- Updates to `pytest.ini` markers if new categories added
- Progress log entry at bottom of this file
- Raise `.coveragerc` `fail_under` and CI `--cov-fail-under` only when metrics justify it

---

## Phase A — `data/` → 100% coverage

**Target package:** `data/`  
**Baseline:** ~92% line coverage  
**Run on:** Apple Silicon, Linux CI (CPU)—no GPU required.

### A.1 Measure gaps

```bash
pytest tests/unit tests/quality tests/e2e -q \
  --cov=data --cov-report=term-missing:skip-covered
```

Record missing modules in the progress log (highest miss first).

### A.2 Add real-data tests (step-by-step)

For each module below **&lt; 100%**, add or extend tests using **real files** under `sample_data/` or `tmp_path`:

| Module | Suggested action | Test location |
|--------|------------------|---------------|
| `prepare_training_data.py` | CLI subprocess: `--csv sample_data/medical_qa.csv`, `--yaml-dir sample_data/patternexamples` | `tests/e2e/` |
| `manual_extractor.py` | Run against synthetic manual PDF fixtures in `tests/e2e/test_pipeline_pdf.py` | Extend `tests/e2e/test_pipeline_pdf.py` |
| `pattern_embedder.py` | Real YAML from `sample_data/patternexamples/hypertension.yaml` | `tests/real/test_pattern_embedder_real.py` if present |
| `template_expander.py` | Load real `data/question_templates.yaml` + `medical_vocabulary.yaml` | `tests/real/test_template_expander_real.py` |
| `knowledge_capture.py` | Round-trip save/load under `tmp_path` knowledge library | `tests/real/test_knowledge_capture_real.py` |
| `judge_llm.py` | Use real `app.model_loader` + tiny model if judge path testable without cloud API | Only if no external API; otherwise test rule-based `judge.py` paths |

**Acceptance criteria Phase A:**

- [ ] `pytest tests/ --cov=data --cov-fail-under=100` passes on CPU/Mac CI
- [ ] No new `unittest.mock` imports in added tests
- [ ] Progress log updated

---

## Phase B — `train/` → 100% coverage

**Target package:** `train/`  
**Baseline:** `finetune_cpu` ~78%; `finetune_unsloth` ~23%; `sharegpt_format` ~89%  
**Run on:** Apple Silicon (CPU/MPS) for CPU trainer; **CUDA GPU worker** for Unsloth full loop.

### B.1 `train/sharegpt_format.py` → 100%

**Where:** Any CPU/Mac/Linux.

1. Extend `tests/real/test_sharegpt_format.py`:
   - Empty conversations list
   - Batch with multiple conversations in one examples dict
   - Tokenizer **with** chat template (use a small instruct model only if download size acceptable; else document skip)
2. Verify: `pytest tests/real/test_sharegpt_format.py --cov=train/sharegpt_format --cov-fail-under=100`

### B.2 `train/finetune_cpu.py` → 100%

**Where:** **Apple Silicon (MPS)** and Linux CPU—mandatory.

1. Keep / maintain in-process integration test:
   - `tests/real/test_train_cpu_integration.py` (calls `main()` via `sys.argv`, tiny-gpt2, `--no-eval`)
2. Add tests in `tests/real/test_finetune_cpu_module.py` or new file:
   - `main()` exit `1` when train file missing (in-process)
   - `main()` with `--no-eval` vs eval branches (read `train/finetune_cpu.py` missing lines)
   - `_lora_target_modules` on Llama-style config if a second tiny fixture is added later
   - All `_PrintProgressCallback` hooks (already partially done)
   - Resume path: create minimal checkpoint dir under `tmp_path` if feasible with 1-step train
3. **Apple Silicon check** (create if missing):

```python
# tests/real/test_apple_silicon_mps.py
@pytest.mark.real
def test_finetune_cpu_runs_on_mps(real_sharegpt_jsonl, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    # run finetune_cpu.main() with tiny-gpt2, 1 epoch, --no-eval (same as integration)
```

4. Verify:

```bash
pytest tests/real/test_train_cpu_integration.py tests/real/test_finetune_cpu_module.py -q \
  --cov=train/finetune_cpu --cov-report=term-missing
```

**Acceptance B.2:**

- [ ] Passes on Mac Apple Silicon without CUDA
- [ ] `finetune_cpu.py` coverage 100%

### B.3 `train/finetune_unsloth.py` — split responsibilities

**Do not claim Phase B complete without separating these two tracks.**

#### Track B.3a — No GPU (Apple Silicon + CPU CI)

**Where:** Mac / Linux CPU CI.

| Step | Action |
|------|--------|
| 1 | Keep `tests/real/test_finetune_unsloth_module.py` (`_make_formatting_func`, missing train file → `SystemExit(1)`) |
| 2 | Keep `tests/real/test_finetune_unsloth_callbacks.py` (`_PrintProgressCallback`, missing val file) |
| 3 | Add test: missing val with **existing** train file (in-process `main()`) |
| 4 | Add test: `USE_EVAL_STRATEGY` branch coverage via importing module and asserting constant exists |

This track will **not** cover lines inside `main()` after model load (~88–159) without Unsloth installed.

#### Track B.3b — CUDA GPU (required for full `finetune_unsloth` coverage)

**Where:** Linux NVIDIA runner only. **Not required on Apple Silicon.**

**Prerequisites:**

- `torch.cuda.is_available()`
- `pip install unsloth` (see `Dockerfile.gpu`)
- Real `train_sharegpt.jsonl` / `val_sharegpt.jsonl` from fixture `real_sharegpt_jsonl`

**Create:** `tests/real/test_finetune_unsloth_gpu.py`

```python
pytestmark = [pytest.mark.real, pytest.mark.gpu, pytest.mark.slow]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Unsloth")
def test_finetune_unsloth_one_step(real_sharegpt_jsonl, tmp_path):
    # sys.argv: tiny model e.g. unsloth/TinyLlama-1.1b-Chat-v1.0 OR smallest available unsloth model
    # --epochs 1 --save-steps 9999 --batch-size 1 --max-seq-length 128
    # in-process finetune_unsloth.main()
    ...
```

**CI:** Add job `test-gpu` in `.gitea/workflows/tests.yml` that runs only `pytest -m gpu`.

**Acceptance B.3:**

- [ ] B.3a complete on Mac CI (partial file coverage documented in log)
- [ ] B.3b hits 100% on `finetune_unsloth.py` in GPU job OR file documents `pragma: no cover` with justification (prefer GPU test)

### B.4 Phase B final verification

```bash
pytest tests/ --cov=train --cov-report=term-missing --cov-fail-under=100
```

(Update CI `fail_under` for `train` only when this passes.)

---

## Phase C — `app/` → 100% coverage

**Target:** `app/chat.py`, `app/model_loader.py`, `app/ollama_client.py`, `app/swarm.py`, **`app/gradio_ui.py`**  
**Baseline:** `gradio_ui` ~46% is the main gap  
**Run on:** Apple Silicon + Linux CI; Ollama optional; no GPU required for most handler tests.

### C.1 Smaller app modules → 100%

**Where:** CPU/Mac.

| Module | Steps |
|--------|--------|
| `app/model_loader.py` | Extend `tests/real/test_model_loader.py`: ONNX path only if `optimum` installed else skip; Unsloth CUDA branch skip on Mac; `load_model` error paths (missing dir) |
| `app/chat.py` | Extend `tests/real/test_chat_real.py`: `--ollama` path with live Ollama; `--interactive` with stdin simulation via `pytest` input fixture only if no mock—prefer testing `_chat_with_ollama` via real HTTP |
| `app/ollama_client.py` | Live tests in `tests/unit/test_ollama_client.py` + `tests/real/test_ollama_live.py`; no patches |
| `app/swarm.py` | Extend `tests/real/test_swarm_real.py`: unload, error strings, `generate_all` with 0 models |

### C.2 `app/gradio_ui.py` — systematic handler coverage

**Where:** In-process only (do not start Gradio server for every test). Use pattern:

```python
import app.gradio_ui as ui

def test_handler(tmp_path):
    old = ui._DATA_DIR
    ui._DATA_DIR = tmp_path / "data"
    try:
        result = ui._some_handler(...)
    finally:
        ui._DATA_DIR = old
```

**Order of implementation (check off in progress log):**

1. [ ] **Memory** — extend `tests/real/test_gradio_memory.py` (reject, export empty, list edge cases)
2. [ ] **Ollama tab** — `_ollama_chat`, `_ollama_test_connection` with live Ollama; Gradio-shaped list message for `_parse_interaction_id`
3. [ ] **Swarm tab** — `_swarm_add_model`, `_swarm_unload_model`, `_swarm_query` with `tiny_lm_dir` under `saved_models`
4. [ ] **Knowledge library** — `_make_library_html`, `save_to_library`, `delete_from_library` with real `tmp_path` library
5. [ ] **Pipeline helpers** — upload/extract/train **status HTML** functions with real `tmp_path` JSONL from `real_sharegpt_jsonl`
6. [ ] **Chat tab** — `_load_model_ui` / chat response path using `tiny_lm_dir` (no Gradio server)
7. [ ] **`build_app()`** — keep smoke in `tests/app/test_app_smoke.py`; add tab-count / component existence
8. [ ] **`main()` / CLI** — `python -m app.gradio_ui --help` subprocess (already in smoke)

Generate coverage after each group:

```bash
pytest tests/real/test_gradio_*.py tests/app/ \
  --cov=app/gradio_ui --cov-report=term-missing
```

**Acceptance Phase C:**

- [ ] `app/` coverage 100% except documented optional CUDA-only lines (if any)
- [ ] All new tests in `tests/real/` or `tests/app/` follow no-mock policy

---

## Phase D — Remove remaining `unittest.mock` usage

**Where:** Audit on CPU/Mac.

### D.1 Inventory

```bash
rg "unittest\.mock|MagicMock|@patch|monkeypatch" tests/ --glob "*.py" -l
```

### D.2 Migration order (replace, then delete mock tests)

1. `tests/unit/test_chat.py` → merge into `tests/real/test_chat_real.py`
2. `tests/unit/test_swarm.py` → merge into `tests/real/test_swarm_real.py`
3. `tests/unit/test_finetune_cpu.py` → merge into `tests/real/test_finetune_cpu_module.py`
4. `tests/unit/test_model_loader.py` → keep only tests that use real dirs; drop patches
5. `tests/app/test_app_smoke.py` — replace `MagicMock` generate_response with tiny real model or skip GPU paths
6. PDF tests using mocks — prefer real small PDFs from `sample_data/`

### D.3 Optional enforcement

Add `tests/test_no_mock_imports.py` that fails if banned imports appear outside an allowlist file.

**Acceptance Phase D:**

- [ ] Inventory empty or allowlist documented
- [ ] Progress log lists migrated files

---

## Phase E — Raise coverage gate to 100%

**Where:** CI + `.coveragerc`.

Only when Phases A–D pass locally:

1. Set `.coveragerc` `fail_under = 100`
2. Set `.gitea/workflows/tests.yml` `--cov-fail-under=100`
3. Add GPU job separately so Unsloth does not block Mac/CPU PRs

```yaml
# Example split jobs
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - run: pytest tests/ -m "not gpu" --cov=app --cov=data --cov=train --cov-fail-under=100
  test-gpu:
    runs-on: [self-hosted, cuda]  # or cloud GPU label
    steps:
      - run: pytest tests/ -m gpu --cov=train --cov-append ...
```

---

## Verification commands (full checklist)

```bash
# 1. Full suite
pytest tests/ -q

# 2. Real tests only (Mac-friendly)
pytest tests/real/ -m "real and not gpu" -q

# 3. Coverage report
pytest tests/ --cov=app --cov=data --cov=train --cov-report=html
open htmlcov/index.html

# 4. Apple Silicon spot check
pytest tests/real/test_train_cpu_integration.py tests/real/test_model_loader.py -q

# 5. GPU job (when available)
pytest tests/ -m gpu -q
```

---

## Progress log (append only — do not delete rows)

| Date | Author | Phase / step | What changed | Why | Coverage (`data` / `train` / `app`) |
|------|--------|--------------|--------------|-----|-------------------------------------|
| 2026-06-02 | AI + human | Baseline | Added `tests/real/*`, sharegpt_format, model_loader fixes, removed mock extended loader tests | Ollama 404 clarity, memory dropdown crash, PEFT load | ~92% / ~65% / ~46% |
| 2026-06-13 | AI + human | Phase 1 docs/tests | Domain-neutral fixtures; MPS tests; SQL/Teradata removed from codebase and tests; CI gate 75% | 100% coverage deferred (Gradio + Unsloth gaps) | ~76% overall |

---

## Adding new work without losing history

When new gaps are discovered:

1. **Do not** remove or rewrite completed rows in **Progress log**.
2. Add a new subsection under the relevant phase, e.g. `### A.3 New: data/judge_llm cloud API`.
3. State **environment** (Mac / CUDA / Ollama), **inputs**, and **acceptance criteria**.
4. If a step is obsolete, strike through the step and add a one-line **Superseded by …** note—keep the old text.

---

## Quick reference: test file map

| Area | Existing real tests | To create / extend |
|------|---------------------|-------------------|
| Model load / generate | `tests/real/test_model_loader.py` | ONNX, error paths |
| Chat CLI | `tests/real/test_chat_real.py` | Ollama branch |
| Swarm | `tests/real/test_swarm_real.py` | unload / errors |
| Memory UI | `tests/real/test_gradio_memory.py` | reject, export |
| Ollama | `tests/real/test_ollama_live.py`, `test_ollama_client.py` | — |
| CPU train | `tests/real/test_train_cpu_integration.py` | resume, errors |
| CPU train unit | `tests/real/test_finetune_cpu_module.py` | main() branches |
| Unsloth no-GPU | `test_finetune_unsloth_module.py`, `test_finetune_unsloth_callbacks.py` | val-missing |
| Unsloth GPU | — | `test_finetune_unsloth_gpu.py` |
| Apple Silicon | `tests/real/test_apple_silicon_mps.py` | Done (7 tests + macOS CI) |
| NVIDIA CUDA | `tests/real/test_cuda_gpu.py` | Done (7 tests; run via `scripts/run_tests_amdworkstation.sh`) |
| Gradio handlers | smoke only | `tests/real/test_gradio_*.py` per tab |
| Data CLI | e2e partial | `test_prepare_training_data_cli.py` |

---

*End of implementation guide. Update this file as the single source of truth for AI-driven test completion.*
