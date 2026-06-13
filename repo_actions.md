# SLM Domain Foundry: Repository Action Plan

**Project**: `slm-domain-foundry`  
**Purpose**: Prepare a domain-adaptive SLM training pipeline for public release on GitHub  
**Target Audience**: Medical AI team in South Korea + open-source community  
**Current Status**: Phase 1 complete (2026-06-13). Domain-neutral medical default, MPS documented, tests green. Pre-public GitHub push and Phase 2 enhancements remain.

---

## Current status (2026-06-13)

| Area | State |
|------|--------|
| **Phase 1 cleanup** | Complete — medical default, no Teradata/SQL coupling |
| **Apple Silicon (MPS)** | Documented (`requirements-mps.txt`, `run_local.sh`, CI macOS job) |
| **Tests** | ~1,207 passed, 3 skipped; 75% CI coverage gate on `app/`, `data/`, `train/` |
| **Release tag** | `v0.1.0-beta` (pre–full domain cleanup); post-cleanup commit on `main`: `87912f5` |
| **GitHub public** | Not yet — still private on Gitea |
| **Phase 2** | Not started (synthetic data, ORPO, DAPT, Korean, etc.) |

---

## Work completed

### Phase 1 (publication readiness)

- **Domain decoupling** — `domain_config.yaml`, `data/domain_config.py`, `--domain-config` CLI; medical keywords and clinical section labels; no SQL/Teradata regex or aliases in production code.
- **Teradata/SQL removal (final pass, commit `87912f5`)** — Deleted `data/sql_vocabulary.yaml`, `examples/domain_config_sql.yaml`, and non-medical pattern samples; generalized chunking (`chunk_text_structured_aware`), extractors, YAML loader (`pattern_alias`, template `content`), knowledge capture (`worked_example` / `example_summary`); all tests use medical fixtures.
- **Alternate domain example** — `examples/domain_config_financial.yaml` + `data/financial_vocabulary.yaml` (reference only; not default).
- **Config consolidation** — `config.yaml`, `train/config.py`, UI copy externalized to `config.yaml` → `ui:` section.
- **Medical samples** — `sample_data/medical_qa.csv`, `hypertension.yaml`, `aspirin_dosing.yaml`, `data/medical_vocabulary.yaml`.
- **Docs & legal** — README, CONTRIBUTING, MIT LICENSE; architecture diagram; domain adaptation table (medical / financial / custom).
- **Dependencies** — `requirements-core.txt`, `requirements-train.txt`, `requirements-inference.txt`, `requirements.txt`, **`requirements-mps.txt`**, `pyproject.toml` extras (`train`, `inference`, `mps`, `dev`).
- **Apple Silicon** — `train/finetune_cpu.py` MPS path, `run_local.sh`, `tests/real/test_apple_silicon_mps.py`, macOS CI job; README + CONTRIBUTING install instructions.
- **Tests & CI** — Domain-neutral fixtures in `tests/conftest.py`; Python 3.10–3.12 Linux matrix; 75% coverage gate; security scan script (`scripts/security_scan.sh`).
- **Pre-release** — Security scan run; `v0.1.0-beta` tag; documentation review (2026-06-12).

### Runtime paths (documented)

| Hardware | Training | Inference |
|----------|----------|-----------|
| NVIDIA CUDA | `train/finetune_unsloth.py` (Unsloth + QLoRA) | Unsloth or transformers |
| Apple Silicon (MPS) | `train/finetune_cpu.py` (HF Trainer + LoRA) | `app/model_loader.py` on MPS |
| CPU | `train/finetune_cpu.py` | transformers / optional ONNX |

---

## Work outstanding

### Pre-public release

- [ ] **Push to public GitHub** — `github.com/agkhan/slm-domain-foundry`; enable Issues/Discussions; add repo topics.
- [ ] **Optional: new release tag** — e.g. `v0.1.1-beta` after domain cleanup (`87912f5`) with updated release notes (current tag points at earlier beta).
- [ ] **Pin dependencies for production** — Replace open `>=` ranges with locked versions after CVE review (`pip-audit` flagged torch CVE-2025-3000 at scan time).
- [ ] **Align macOS CI install with `requirements-mps.txt`** — CI still filters `bitsandbytes` from `requirements.txt`; functionally equivalent but could use `requirements-mps.txt` for clarity.

### Quality & coverage (Phase 1 stretch goals)

- [ ] **Raise coverage toward 100%** on `app/`, `data/`, `train/` — largest gaps: `app/gradio_ui.py` (~46%), `train/finetune_unsloth.py` (CUDA/Unsloth body rarely executed in CI). Roadmap: `tests/AI_COVERAGE_IMPLEMENTATION.md`.
- [ ] **Dedicated CUDA CI job** — Optional `@pytest.mark.gpu` Unsloth smoke on NVIDIA runner.

### Phase 2 (post-release enhancements)

All Phase 2 items below remain **not started**. See [Phase 2: Cutting-Edge Enhancements](#phase-2-cutting-edge-enhancements) for detail.

| ID | Feature | Priority |
|----|---------|----------|
| A | Teacher-student synthetic data generation | Highest (medical data scarcity) |
| B | DoRA adapter support | Easy win |
| C | ORPO preference alignment | Critical (medical safety) |
| D | Domain-adaptive continued pre-training (DAPT) | Essential for Korean medical SLM |
| E | RAG-augmented fine-tuning (RAF) | High (unique pipeline advantage) |
| F | Medical evaluation suite | Production readiness |
| G | Korean / multilingual support | Critical for Seoul team |

---

## Work bypassed or deferred (with rationale)

| Item | Decision | Explanation |
|------|----------|-------------|
| **Teradata/SQL backward compatibility** | Bypassed | Explicit product decision: no legacy SQL field names (`teradata_function`, `sql_vocabulary`, `has_sql_content` aliases). Clean domain-adaptive API only. |
| **100% test coverage gate in CI** | Deferred | 75% gate enforced today; 100% blocked by large Gradio UI surface and CUDA-only Unsloth paths. Documented in `AI_COVERAGE_IMPLEMENTATION.md`; not required for beta. |
| **`unittest.mock` in core tests** | Partially bypassed | Policy prefers real I/O; some unit tests still mock I/O boundaries (e.g. chat/Ollama). Full mock removal is a separate hardening pass. |
| **Unsloth on Apple Silicon** | Bypassed | Unsloth requires CUDA; Mac uses `finetune_cpu.py` + MPS instead. |
| **GPU training in Docker on Mac** | Bypassed | Docker Desktop runs Linux VM — no Metal/MPS passthrough. Native `run_local.sh` is the supported Mac path. |
| **Dependency pinning in `requirements*.txt`** | Deferred | Open ranges kept for developer flexibility; security scan notes CVEs in ranges. Pin before production/medical deployment. |
| **GitHub public release** | Deferred | Awaiting explicit go-live; Gitea remains source of truth. |
| **Phase 2 features** | Deferred | Intentionally post–Phase 1; incremental PRs after public release. |
| **Legal/financial sample domains in README walkthrough** | Deferred | README uses medical quick start only; financial config exists as reference under `examples/`. Full legal/scientific walkthroughs not written. |
| **sentence-transformers in `requirements-core.txt`** | Deferred | Semantic chunking is optional (`chunking.py` imports lazily); not added to core requirements to keep data-only install light. |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Matters](#why-this-matters)
3. [Current Status](#current-status-2026-06-13)
4. [Work Completed](#work-completed)
5. [Work Outstanding](#work-outstanding)
6. [Work Bypassed or Deferred](#work-bypassed-or-deferred-with-rationale)
7. [Two-Phase Approach](#two-phase-approach)
8. [Phase 1: Cleanup Checklist](#phase-1-cleanup-checklist)
9. [Phase 2: Cutting-Edge Enhancements](#phase-2-cutting-edge-enhancements)
10. [Execution Priority](#execution-priority)
11. [Change Log](#change-log)

---

## Project Overview

### What We Have

The `ai_slm_training` codebase is a solid end-to-end pipeline for training small language models:

- **Data ingestion**: PDF (with manual-mode extraction), CSV, YAML pattern files → Alpaca/ShareGPT JSONL
- **Training**: Unsloth + SFTTrainer with QLoRA, 4-bit quantized base models
- **Inference**: Gradio UI + CLI chat + `SwarmManager` for multi-model parallel queries
- **RAG-lite**: Live YAML-based context injection via `KnowledgeRetriever`
- **Conversation memory**: Logging and approval workflows
- **CI/CD**: Gitea Actions for tests and coverage

### What Needs to Change

~~The codebase was tightly coupled to Teradata SQL domain knowledge.~~ **Resolved (2026-06-13).**

Remaining before broad public adoption:

1. **Public GitHub release** and optional post-cleanup version tag
2. **Phase 2 enhancements** (synthetic data, alignment, DAPT, Korean, eval suite)
3. **Production hardening** — dependency pinning, higher coverage, optional GPU CI

Once on GitHub, the repo is ready for the Korean medical AI team to fork and customize via YAML config.

---

## Why This Matters

### For the Korean Medical AI Team

- **Custom SLM on limited data**: Medical domain has high-quality but small labeled datasets. This pipeline's teacher-student synthetic data generation (Phase 2) is ideal for that scenario.
- **Korean language support**: Easy to add Korean-language base models and question templates.
- **RAG-first design**: Medical AI needs grounding in trusted knowledge bases (clinical guidelines, drug databases). The existing `KnowledgeRetriever` + training-time RAG (Phase 2) is purpose-built for this.
- **Privacy-first**: Runs entirely on-premise or in private cloud. No data leaves the deployment environment.

### For the Open-Source Community

- **Reproducible**: Docker + `requirements.txt` means anyone can run it.
- **Pluggable**: YAML-based domain config lets users adapt to legal, financial, scientific, or any other vertical.
- **Modern stack**: Unsloth (fastest LoRA/QLoRA), Gradio (best UI for demos), TRL (production SFT/DPO).
- **Tested**: CI with pytest + coverage roadmap ensures reliability.

---

## Two-Phase Approach

### Phase 1: Cleanup for Public Release

**Goal**: Remove all domain-specific coupling and make the repo publication-ready.

**Timeline**: ~3-5 days of focused work.

### Phase 2: Cutting-Edge Enhancements

**Goal**: Add 2025-2026 state-of-the-art techniques that are directly relevant to medical SLMs.

**Timeline**: ~2-3 weeks, can be done incrementally after public release.

---

## Phase 1: Cleanup Checklist

### 1. Remove Teradata-Specific Domain Coupling

**Status**: ✅ Complete (including final pass 2026-06-13, commit `87912f5`)

**Files affected** (all updated):
- `data/manual_extractor.py`, `data/chunking.py`, `data/prepare_training_data.py`, `data/yaml_pattern_loader.py`
- `data/knowledge_capture.py`, `data/question_templates.yaml`, `data/template_expander.py`
- `app/gradio_ui.py`, `domain_config.yaml`, `tests/conftest.py` + full test suite
- Removed: `data/sql_vocabulary.yaml`, `examples/domain_config_sql.yaml`

**Actions**:

- [x] **Extract hardcoded regex to `domain_config.yaml`**
  - Structured-content detection, optional `function_pattern`, example section labels
  - `--domain-config` CLI on `prepare_training_data.py`

- [x] **Replace domain-specific system prompt in `gradio_ui.py`**
  - Load from `config.yaml`; override via `SLM_SYSTEM_PROMPT`

- [x] **Medical sample data only**
  - Removed TD17 PDF and analytics/SQL pattern trees
  - Kept: `medical_qa.csv`, `hypertension.yaml`, `aspirin_dosing.yaml`, `medical_vocabulary.yaml`

- [x] **Remove SQL/Teradata backward compatibility**
  - No `has_sql_content`, `teradata_function`, or SQL vocabulary files
  - Alternate reference: `examples/domain_config_financial.yaml`

**Validation**: `pytest tests/` — ~1,207 passed (2026-06-13).

---

### 2. Configuration Consolidation

**Current state**: Config scattered across CLI args in:
- `train/finetune_unsloth.py`
- `data/prepare_training_data.py`
- `app/gradio_ui.py`

**Target**: Single `config.yaml` or `config.toml` at repo root.

**Actions**:

- [x] **Create `config.yaml` template** with sections:
  ```yaml
  domain:
    name: "medical"
    system_prompt: "You are a medical AI assistant..."
    domain_keywords: ["diagnosis", "treatment", "ICD", "medication", ...]
    section_labels: ["symptoms", "treatment", "dosage", "contraindications", ...]
    function_patterns: []  # optional; set in domain_config.yaml per vertical
  
  model:
    base_model: "unsloth/Llama-3.2-1B-Instruct"
    lora_r: 16
    lora_alpha: 32
    max_seq_length: 2048
    epochs: 3
    learning_rate: 2e-4
    batch_size: 2
    grad_accum: 4
  
  paths:
    training_data: "training_data"
    output_model: "output_model"
    knowledge_library: "knowledge_library"
  ```

- [x] **Add config loader** in `train/config.py`
  - Use `pyyaml` or `toml` stdlib
  - Merge config file + CLI args (CLI overrides config)

- [x] **Update all scripts** to load from config:
  - `finetune_unsloth.py`
  - `prepare_training_data.py`
  - `gradio_ui.py`

**Validation**: `python -m train.finetune_unsloth --config config.yaml` should work without additional args.

---

### 3. README Overhaul

**Current state**: README uses medical quick start; financial alternate documented; SQL examples removed.

**Actions**:

- [x] **Rewrite intro** with "bring your own domain" framing
- [x] **Architecture diagram** (Mermaid)
- [x] **Document config.yaml and domain adaptation** (medical, financial, custom)
- [x] **Prerequisites** — Python 3.10+, CUDA optional, Apple Silicon + `requirements-mps.txt`
- [x] **Quick Start** with medical example
- [x] **Apple Silicon section** — three runtime paths, `run_local.sh`, manual MPS install

**Validation**: Fresh clone + follow README should work without prior knowledge.

---

### 4. Add LICENSE and CONTRIBUTING.md

**Actions**:

- [x] **Add LICENSE file**
  - Recommendation: **MIT** or **Apache 2.0**
  - MIT: simpler, more permissive
  - Apache 2.0: includes patent grant, better for corporate users
  - Decision: **MIT** (medical AI community prefers permissive)

- [x] **Add CONTRIBUTING.md**
  - Code of conduct reference
  - How to report issues
  - How to submit PRs
  - Development setup (venv, pre-commit hooks)
  - Testing requirements (`pytest`, coverage ≥80%)

**Validation**: Check GitHub renders LICENSE badge correctly.

---

### 5. Dependency Pinning & Split Requirements

**Current state**: Single `requirements.txt` with `>=` everywhere.

**Actions**:

- [x] **Create `requirements-core.txt`** (data prep only, no torch/unsloth)
  ```
  pdfplumber>=0.10.0
  PyPDF2>=3.0.0
  pyyaml>=6.0
  numpy>=1.24.0
  pandas>=2.0.0
  sentence-transformers  # for semantic chunking
  ```

- [x] **Create `requirements-train.txt`** (full training stack)
  ```
  -r requirements-core.txt
  torch>=2.1.0
  transformers>=4.36.0
  datasets>=2.14.0
  peft>=0.7.0
  trl>=0.7.4
  unsloth  # Install separately: pip install unsloth
  ```

- [x] **Create `requirements-inference.txt`** (inference + Gradio)

- [x] **Create `requirements-mps.txt`** (Apple Silicon native: train + Gradio, no `bitsandbytes`)

- [x] **Add `pyproject.toml`** with optional extras: `train`, `inference`, `mps`, `dev`

**Validation**: `pip install -e .[train]` should install training deps.

---

### 6. Test Coverage & CI

**Current state**: Domain-neutral fixtures; medical default domain in autouse conftest.

**Actions**:

- [x] **Audit `tests/` for domain-specific fixtures** — medical/generic content only
- [x] **Config loader tests** — `tests/unit/test_config.py`
- [x] **CI workflow** — `.gitea/workflows/tests.yml`: Linux 3.10–3.12 @ 75% coverage; macOS MPS job
- [x] **MPS integration tests** — `tests/real/test_apple_silicon_mps.py`

**Validation**: `pytest tests/` passes; CI enforces `--cov-fail-under=75`.

---

### 7. Apple Silicon (MPS) Documentation & Dependencies

**Actions**:

- [x] **`requirements-mps.txt`** — documented Mac-native stack (no CUDA-only packages)
- [x] **`run_local.sh`** — installs `requirements-mps.txt`, detects MPS, launches Gradio
- [x] **`pyproject.toml` `[mps]` extra**
- [x] **README + CONTRIBUTING** — install paths and limitations (no Docker MPS)

**Validation**: `./run_local.sh` on Mac; `pytest tests/real/test_apple_silicon_mps.py -m mps`.

---

### 8. Final Pre-Release Checklist

- [x] **Security scan**
  - `safety scan` + `pip-audit` run 2026-06-12 (see changelog)
  - Git history scanned for credential patterns — **none found**
  - Re-run: `./scripts/security_scan.sh`
  - **Note**: Unpinned `>=` specifiers hide many CVEs in range scans; pin deps before production deploy. `pip-audit` flagged `torch` CVE-2025-3000 (no fix version published at scan time).

- [x] **Documentation review**
  - Spellcheck README, CONTRIBUTING, docstrings — done 2026-06-12
  - Ensure all links work — external HF/Karpathy links OK; GitHub repo/issues 404 until public (documented in README)
  - Updated stale TD17/Teradata references in test docs and CLI demo copy

- [x] **Sample data audit**
  - No personal data
  - No proprietary content
  - Properly attributed if using public datasets

- [x] **Version tagging**
  - Tag `v0.1.0-beta` before public push
  - Write release notes (in annotated tag + changelog below)

- [ ] **Make repo public on GitHub**
  - Push from Gitea → `github.com/agkhan/slm-domain-foundry`
  - Enable Issues, Discussions
  - Add topics: `slm`, `llm`, `fine-tuning`, `medical-ai`, `rag`, `unsloth`

---

## Phase 2: Cutting-Edge Enhancements

**Note**: These can be added incrementally after public release. Each is a standalone feature.

### A. Teacher-Student Synthetic Data Generation

**Priority**: ⭐⭐⭐ **Highest impact for medical AI**

**Why**: Medical SLMs have limited labeled data. Synthetic generation from a teacher model (GPT-4o, Claude, or local Llama 70B) can 10x the training corpus.

**Implementation**:

- [ ] **Add `data/synthetic_generator.py`**
  - Takes raw clinical text or knowledge base entries
  - Uses teacher model API (OpenAI, Anthropic, or Ollama local)
  - Generates Q&A pairs with chain-of-thought reasoning

- [ ] **Add critic/filter agent**
  - Second LLM call scores generated pairs for:
    - Factual accuracy
    - Clinical appropriateness
    - Groundedness (does it cite the source text?)
  - Rejects pairs below threshold

- [ ] **Integrate with existing pipeline**
  - `prepare_training_data.py --synthetic-from raw_medical_corpus.txt`
  - Outputs to same Alpaca/ShareGPT JSONL format

**Research basis**: 
- "A Modular Approach for Clinical SLMs" (ACL 2025)
- "Training SLMs on Synthetic Data: The Distillation Pipeline" (AAIA 2026)

**Validation**: Generate 1000 synthetic pairs, manually review 50, train model, compare to baseline.

---

### B. DoRA Adapter Support

**Priority**: ⭐⭐ **Easy win, measurable improvement**

**Why**: DoRA (Weight-Decomposed Low-Rank Adaptation) separates magnitude and direction updates → better performance than LoRA on low-data domains.

**Implementation**:

- [ ] **Add `--use-dora` flag** to `finetune_unsloth.py`
  ```python
  model = FastLanguageModel.get_peft_model(
      model,
      use_dora=args.use_dora,  # NEW
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      ...
  )
  ```

- [ ] **Update config.yaml** with `model.use_dora: true/false`

- [ ] **Benchmark**: Train same model with LoRA vs DoRA, compare eval loss

**Research basis**:
- "Exploring Efficient Learning of Small BERT Networks with LoRA and DoRA" (arXiv 2025)
- "Comparing Fine-Tuning Optimization Techniques" (Encora 2024)

**Validation**: DoRA should match or beat LoRA on medical Q&A eval set.

---

### C. ORPO Preference Alignment

**Priority**: ⭐⭐⭐ **Critical for medical safety**

**Why**: Medical AI must refuse harmful advice. ORPO combines SFT + alignment in one pass, 56% faster than DPO, no reference model needed.

**Implementation**:

- [ ] **Add `train/align_orpo.py`**
  - Takes preference pairs: `(question, safe_answer, unsafe_answer)`
  - Uses TRL's `ORPOTrainer`
  - Example:
    ```python
    from trl import ORPOTrainer, ORPOConfig
    
    config = ORPOConfig(
        learning_rate=1e-5,
        beta=0.1,  # ORPO-specific
        ...
    )
    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=preference_pairs,
        ...
    )
    ```

- [ ] **Create medical safety preference dataset**
  - Positive: "What is the dose of aspirin for MI?" → "75-325mg daily..."
  - Negative: Same question → "Take as much as you want" (rejected)
  - Source from clinical guidelines + GPT-4 generation

- [ ] **Two-stage training**: SFT → ORPO

**Research basis**:
- "ORPO: Monolithic Odds Ratio Preference Optimization" (EMNLP 2024)
- "DPO Isn't Enough: The Modern Post-Training Stack" (Medium 2025)

**Validation**: Test model on adversarial prompts ("How to overdose on X?") — should refuse.

---

### D. Domain-Adaptive Continued Pre-Training (DAPT)

**Priority**: ⭐⭐ **Essential for Korean medical SLM**

**Why**: Base models (Llama, Mistral) are trained on general web text. DAPT on Korean clinical notes + medical literature adapts the model's knowledge before fine-tuning.

**Implementation**:

- [ ] **Add `train/dapt.py`**
  - Next-token prediction on unlabeled medical corpus
  - Uses same Unsloth setup but simpler objective
  - Example:
    ```bash
    python -m train.dapt \
      --corpus korean_medical_corpus.txt \
      --base-model unsloth/Llama-3.2-1B \
      --output-dir dapt_model \
      --tokens 400M
    ```

- [ ] **Mix-CPT approach** (optional enhancement)
  - Interleave raw text + QA pairs in pretraining
  - Prevents catastrophic forgetting

- [ ] **Three-stage pipeline**: Base → DAPT → SFT → (optional ORPO)

**Research basis**:
- "Domain-Adaptive Continued Pre-Training" (arXiv 2025)
- "Mix-CPT: A Domain Adaptation Framework" (ICLR 2025)

**Validation**: MMLU score should increase on medical subset after DAPT.

---

### E. RAG-Augmented Fine-Tuning (RAF)

**Priority**: ⭐⭐⭐ **Unique advantage over other SLM pipelines**

**Why**: Current `KnowledgeRetriever` only works at inference. RAF teaches the model **how to use** retrieved context during training → better RAG performance.

**Implementation**:

- [ ] **Modify SFT data format** to include retrieved context
  - During `prepare_training_data.py`, query knowledge library
  - Prepend context to each training example:
    ```json
    {
      "messages": [
        {"role": "system", "content": "You are a medical assistant. Use the following context: [RETRIEVED GUIDELINES]"},
        {"role": "user", "content": "What is the treatment for hypertension?"},
        {"role": "assistant", "content": "Based on the guidelines, ..."}
      ]
    }
    ```

- [ ] **Replace TF-IDF with dense retrieval**
  - Use `sentence-transformers` + FAISS index
  - Or use **turbovec** (connects to your vector index work!)
  - Build index from `knowledge_library/*.yaml`

- [ ] **Add `--rag-training` flag** to `finetune_unsloth.py`

**Research basis**:
- "RAG-Enhanced Open SLMs for Hypertension Management" (PMC 2025)
- "Enhancing Medical AI with RAG" (PMC 2025)

**Validation**: Model with RAF should outperform non-RAF on questions requiring knowledge base lookup.

---

### F. Evaluation Suite for Medical AI

**Priority**: ⭐⭐ **Production readiness requirement**

**Why**: Can't deploy medical AI without metrics. Need automated eval on safety + accuracy.

**Implementation**:

- [ ] **Create `eval/` directory** with:
  - `factuality_scorer.py` — Compare answers to ground truth KB
  - `hallucination_detector.py` — Flag citations to non-existent sources
  - `safety_classifier.py` — Binary classifier for harmful outputs
  - `benchmarks.py` — MedQA, PubMedQA (if English), custom Korean eval set

- [ ] **Add `--eval-only` mode** to `finetune_unsloth.py`
  - Load checkpoint, run eval suite, output JSON report

- [ ] **CI integration**: Run eval on every PR

**Validation**: Eval suite should catch regression if model quality drops.

---

### G. Multi-Lingual / Korean Language Support

**Priority**: ⭐⭐⭐ **Critical for Korean medical AI team**

**Implementation**:

- [ ] **Add Korean base model support**
  - Update `config.yaml` with:
    ```yaml
    model:
      base_model: "EXAONE-3.5-2.4B-Instruct"  # or HCX-003, or Llama-3.1 + Korean DAPT
      language: "ko"
    ```

- [ ] **Korean question templates**
  - Add `data/question_templates_ko.yaml`
  - Mirror structure of English templates

- [ ] **Korean tokenizer handling**
  - Ensure proper chat template for Korean models
  - May need custom tokenizer settings

- [ ] **Bilingual mode** (optional)
  - Support English + Korean in same training run
  - Useful for code-switching medical scenarios

**Validation**: Train on Korean medical Q&A, verify fluent Korean responses.

---

## Execution Priority

### Done (Phase 1)

1. Domain decoupling + full SQL/Teradata removal  
2. Config consolidation + README + LICENSE + CONTRIBUTING  
3. Split requirements + `requirements-mps.txt` + MPS tests/CI  
4. Security scan + `v0.1.0-beta` tag + documentation review  

### Next (pre-public)

5. **GitHub public push** + release notes for post-cleanup `main`  
6. **Optional**: `v0.1.1-beta` tag; pin dependencies for production  

### Then (Phase 2, incremental)

7. **Phase 2A** — Synthetic data generator  
8. **Phase 2C** — ORPO alignment (medical safety)  
9. **Phase 2D** — DAPT (Korean medical corpus)  
10. **Phase 2E** — RAG-augmented fine-tuning  
11. **Phase 2B / F / G** — DoRA, eval suite, Korean templates  

### Stretch (quality)

12. Coverage toward 100% (`AI_COVERAGE_IMPLEMENTATION.md`)  
13. Dedicated CUDA CI job for Unsloth smoke tests  

---

## Change Log

### 2026-06-12

- **Repo created**: Forked from `ai_slm_training` to `slm-domain-foundry`
- **Status**: Private on Gitea
- **Initial commit**: 27 commits, 33 MiB, full history preserved
- **This document created**: `repo_actions.md` initialized with Phase 1 + Phase 2 plan

### 2026-06-12 (Phase 1 complete — initial)

- **Domain decoupling**: `domain_config.yaml`, `data/domain_config.py`, `examples/domain_config_sql.yaml` *(later removed)*
- **Config consolidation**: `config.yaml`, `train/config.py`, `--config` on prepare/train/gradio
- **Medical samples**: `sample_data/medical_qa.csv`, clinical YAML patterns, `data/medical_vocabulary.yaml`
- **Docs & legal**: README overhaul, MIT LICENSE, CONTRIBUTING.md
- **Dependencies**: Split requirements + `pyproject.toml`
- **Tests/CI**: `tests/unit/test_config.py`, Python 3.10–3.12 matrix, coverage target 75%

### 2026-06-12 (Pre-release: security scan + v0.1.0-beta)

- **Security scan** (`safety scan`, `pip-audit`, git history):
  - 0 vulnerabilities reported on resolved packages (safety)
  - 271 CVEs in unpinned dependency ranges ignored by default (pin before production)
  - `pip-audit`: torch CVE-2025-3000 on currently installed 2.12.0 (no fix version listed)
  - Git history: no API keys, passwords, or private keys detected
  - Repeatable script: `scripts/security_scan.sh`
- **Release tag**: `v0.1.0-beta` — first public-ready beta for domain-adaptive SLM training

#### v0.1.0-beta release notes

**Highlights**
- End-to-end SLM pipeline: data prep → Unsloth fine-tuning → Gradio/CLI inference
- Medical AI default profile (`config.yaml`, `domain_config.yaml`, sample clinical data)
- Domain-adaptive YAML config (medical default; financial reference under `examples/`)
- MIT licensed, split requirements, CI on Python 3.10–3.12

**Quick start**
```bash
pip install -r requirements.txt
python -m data.prepare_training_data --csv sample_data/medical_qa.csv --yaml-dir sample_data/patternexamples --output-dir training_data
python -m train.finetune_unsloth --config config.yaml
python -m app.gradio_ui --model-dir output_model
```

**Known limitations**
- Phase 2 features (synthetic data, ORPO, DAPT, Korean templates) not yet included
- Dependencies use `>=` ranges; pin for production deployments
- GPU training requires separate `unsloth` install with CUDA

### 2026-06-12 (Documentation review)

- README: Gitea clone URL, GitHub marked as planned; repository/issues table added
- CONTRIBUTING: full install steps, Gitea issue tracker wording, security scan mention
- `train/README_FROM_SCRATCH.md`: documents existing `scripts/export_for_from_scratch.py`
- `tests/TESTING.md` + `AI_COVERAGE_IMPLEMENTATION.md`: aligned with 75% CI gate; removed TD17 references
- `app/chat.py`: medical demo questions (removed Teradata sample prompts)
- Link check: nanoGPT, minGPT, Hugging Face run_clm — OK; GitHub repo/issues — 404 (expected pre-public)

### 2026-06-12 (Apple Silicon MPS — commit `83da807`)

- MPS training path documented in README; `finetune_cpu.py` + `run_local.sh`
- `tests/real/test_apple_silicon_mps.py` (7 tests); macOS CI job in `.gitea/workflows/tests.yml`
- Gradio auto-selects `finetune_cpu` when CUDA/Unsloth unavailable

### 2026-06-13 (Domain cleanup final + MPS requirements — commit `87912f5`)

- **Removed all Teradata/SQL assets and code paths** — no backward-compat aliases
- **Added** `requirements-mps.txt`, `examples/domain_config_financial.yaml`
- **Generalized** chunking, extractors, YAML patterns, question templates, knowledge capture UI fields
- **Tests** rewritten with medical fixtures; ~1,207 passed
- **Pushed** to Gitea `main`

#### Post-cleanup quick start (Mac MPS)

```bash
pip install -r requirements-mps.txt
./run_local.sh
```

#### Post-cleanup quick start (Linux CUDA)

```bash
pip install -r requirements.txt
pip install unsloth   # CUDA only
python -m data.prepare_training_data \
  --csv sample_data/medical_qa.csv \
  --yaml-dir sample_data/patternexamples \
  --output-dir training_data
python -m train.finetune_unsloth --config config.yaml
python -m app.gradio_ui --model-dir output_model
```

### [Future entries go here]

---

## Notes for Collaborators

- **Markdown format**: This file is version-controlled. Update checklist items by changing `[ ]` to `[x]` when complete.
- **Git workflow**: Create feature branches for each Phase 1 item (e.g., `cleanup/domain-config`, `cleanup/readme-overhaul`).
- **Issue tracking**: Link GitHub issues to checklist items for transparency.
- **Phase 2 PRs**: Each enhancement should be a separate PR with benchmarks in the description.

---

**Last Updated**: 2026-06-13 (Phase 1 complete; domain-neutral; MPS documented)  
**Maintained By**: AG Khan  
**Contact**: Gitea issues on the project host (GitHub issues after public release)