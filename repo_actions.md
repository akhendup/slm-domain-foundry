# SLM Domain Foundry: Repository Action Plan

**Project**: `slm-domain-foundry`  
**Purpose**: Prepare a domain-adaptive SLM training pipeline for public release on GitHub  
**Target Audience**: Medical AI team in South Korea + open-source community  
**Current Status**: Private Gitea repo, forked from `ai_slm_training` on 2026-06-12  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why This Matters](#why-this-matters)
3. [Two-Phase Approach](#two-phase-approach)
4. [Phase 1: Cleanup Checklist](#phase-1-cleanup-checklist)
5. [Phase 2: Cutting-Edge Enhancements](#phase-2-cutting-edge-enhancements)
6. [Execution Priority](#execution-priority)
7. [Change Log](#change-log)

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

The current codebase is **tightly coupled to Teradata SQL domain knowledge**. To make it generic and shareable:

1. **Decouple domain-specific logic** (SQL keywords, Teradata function patterns, TD17 sample data)
2. **Abstract configuration** into YAML/TOML files
3. **Generalize documentation** with medical AI examples
4. **Add LICENSE and CONTRIBUTING.md**
5. **Split dependencies** for data-only, training, and inference use cases

Once clean, the repo will be pushed to **public GitHub** and shared with the Korean medical AI team.

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

**Files affected**:
- `data/manual_extractor.py`
- `data/chunking.py`
- `app/gradio_ui.py`
- `sample_data/`

**Actions**:

- [ ] **Extract hardcoded SQL/Teradata regex to `domain_config.yaml`**
  - Move `_SQL_KW_RE`, `_TD_FUNC_RE`, `_NON_FUNC_SUFFIX_RE` from `manual_extractor.py` to a config file
  - Same for duplicated SQL regex in `chunking.py`
  - Add a `--domain-config` CLI arg to `prepare_training_data.py`

- [ ] **Replace Teradata system prompt in `gradio_ui.py`**
  - Current: hardcoded Teradata-specific system message
  - Target: Load from `config.yaml` or `--system-prompt` CLI arg
  - Example: `You are a medical AI assistant specialized in clinical decision support...`

- [ ] **Swap TD17 sample data for generic medical example**
  - Remove: `sample_data/TD17_Analytic_Functions.pdf`
  - Remove: `sample_data/patternexamples/` YAML files (if Teradata-specific)
  - Add: Public domain medical Q&A CSV or a synthetic clinical vocabulary example
  - Ensure no proprietary/personal data in `sample_data/`

**Validation**: Run `pytest` after changes to ensure no broken imports.

---

### 2. Configuration Consolidation

**Current state**: Config scattered across CLI args in:
- `train/finetune_unsloth.py`
- `data/prepare_training_data.py`
- `app/gradio_ui.py`

**Target**: Single `config.yaml` or `config.toml` at repo root.

**Actions**:

- [ ] **Create `config.yaml` template** with sections:
  ```yaml
  domain:
    name: "medical"
    system_prompt: "You are a medical AI assistant..."
    domain_keywords: ["diagnosis", "treatment", "ICD", "medication", ...]
    section_labels: ["symptoms", "treatment", "dosage", "contraindications", ...]
    function_patterns: []  # e.g., SQL functions for analytics domains
  
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

- [ ] **Add config loader** in `train/config.py`
  - Use `pyyaml` or `toml` stdlib
  - Merge config file + CLI args (CLI overrides config)

- [ ] **Update all scripts** to load from config:
  - `finetune_unsloth.py`
  - `prepare_training_data.py`
  - `gradio_ui.py`

**Validation**: `python -m train.finetune_unsloth --config config.yaml` should work without additional args.

---

### 3. README Overhaul

**Current state**: README assumes Teradata/SQL context.

**Actions**:

- [ ] **Rewrite intro** with "bring your own domain" framing
  - Remove SQL/TD17-specific examples
  - Add medical Q&A walkthrough as primary example

- [ ] **Add architecture diagram**
  - Data → Training → Inference → RAG flow
  - Use Mermaid or ASCII diagram

- [ ] **Document config.yaml usage**
  - Show how to adapt to new domains
  - Include 3 examples: medical, legal, financial

- [ ] **Add Prerequisites section**
  - Python 3.10+
  - CUDA 11.8+ for GPU training (optional)
  - Docker (optional)

- [ ] **Add Quick Start with medical example**
  ```bash
  # 1. Prepare data
  python -m data.prepare_training_data \
    --csv sample_data/medical_qa.csv \
    --output-dir training_data
  
  # 2. Fine-tune
  python -m train.finetune_unsloth \
    --config config.yaml \
    --train-file training_data/train_sharegpt.jsonl
  
  # 3. Run inference
  python -m app.gradio_ui --model-dir output_model
  ```

**Validation**: Fresh clone + follow README should work without prior knowledge.

---

### 4. Add LICENSE and CONTRIBUTING.md

**Actions**:

- [ ] **Add LICENSE file**
  - Recommendation: **MIT** or **Apache 2.0**
  - MIT: simpler, more permissive
  - Apache 2.0: includes patent grant, better for corporate users
  - Decision: **MIT** (medical AI community prefers permissive)

- [ ] **Add CONTRIBUTING.md**
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

- [ ] **Create `requirements-core.txt`** (data prep only, no torch/unsloth)
  ```
  pdfplumber>=0.10.0
  PyPDF2>=3.0.0
  pyyaml>=6.0
  numpy>=1.24.0
  pandas>=2.0.0
  sentence-transformers  # for semantic chunking
  ```

- [ ] **Create `requirements-train.txt`** (full training stack)
  ```
  -r requirements-core.txt
  torch>=2.1.0
  transformers>=4.36.0
  datasets>=2.14.0
  peft>=0.7.0
  trl>=0.7.4
  unsloth  # Install separately: pip install unsloth
  ```

- [ ] **Create `requirements-inference.txt`** (CPU inference only)
  ```
  -r requirements-core.txt
  torch>=2.1.0  # CPU-only
  transformers>=4.36.0
  gradio>=4.0.0
  ```

- [ ] **Add `pyproject.toml`** for proper package metadata
  ```toml
  [project]
  name = "slm-domain-foundry"
  version = "0.1.0"
  description = "Domain-adaptive SLM training pipeline"
  authors = [{name = "AG Khan", email = "your@email.com"}]
  license = {text = "MIT"}
  requires-python = ">=3.10"
  dependencies = ["pyyaml", "numpy", "pandas"]
  
  [project.optional-dependencies]
  train = ["torch", "transformers", "unsloth", ...]
  inference = ["gradio", ...]
  ```

**Validation**: `pip install -e .[train]` should install training deps.

---

### 6. Test Coverage & CI

**Current state**: Tests exist but may reference Teradata fixtures.

**Actions**:

- [ ] **Audit `tests/` for domain-specific fixtures**
  - Replace TD17/SQL test data with generic fixtures
  - Ensure tests use `config.yaml` if needed

- [ ] **Add test for config loader**
  - `tests/test_config.py`
  - Validate YAML parsing, CLI override, missing keys

- [ ] **Update CI workflow** (`.gitea/workflows/tests.yml`)
  - Run on: `push`, `pull_request`
  - Test matrix: Python 3.10, 3.11, 3.12
  - Coverage report (target ≥75%)

**Validation**: `pytest --cov=. --cov-report=html` should pass.

---

### 7. Final Pre-Release Checklist

- [ ] **Security scan**
  - Run `safety check` on requirements
  - Check for exposed secrets in commit history (`git log -p | grep -i password`)

- [ ] **Documentation review**
  - Spellcheck README, CONTRIBUTING, docstrings
  - Ensure all links work

- [ ] **Sample data audit**
  - No personal data
  - No proprietary content
  - Properly attributed if using public datasets

- [ ] **Version tagging**
  - Tag `v0.1.0-beta` before public push
  - Write release notes

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

### Immediate (This Week)

1. **Phase 1 Items 1-3**: Domain decoupling, config consolidation, README
2. **Phase 1 Item 4**: LICENSE + CONTRIBUTING.md
3. **Phase 1 Item 5**: Split requirements

### Next Week

4. **Phase 1 Items 6-7**: Test cleanup, security scan, public push
5. **Phase 2A**: Synthetic data generator (start implementation)

### Following 2 Weeks

6. **Phase 2C**: ORPO alignment (highest medical safety impact)
7. **Phase 2D**: DAPT stage (essential for Korean model)
8. **Phase 2E**: RAG-augmented fine-tuning

### Month 2

9. **Phase 2B**: DoRA support (easy, measurable win)
10. **Phase 2F**: Eval suite
11. **Phase 2G**: Korean language integration

---

## Change Log

### 2026-06-12

- **Repo created**: Forked from `ai_slm_training` to `slm-domain-foundry`
- **Status**: Private on Gitea
- **Initial commit**: 27 commits, 33 MiB, full history preserved
- **This document created**: `repo_actions.md` initialized with Phase 1 + Phase 2 plan

### [Future entries go here]

---

## Notes for Collaborators

- **Markdown format**: This file is version-controlled. Update checklist items by changing `[ ]` to `[x]` when complete.
- **Git workflow**: Create feature branches for each Phase 1 item (e.g., `cleanup/domain-config`, `cleanup/readme-overhaul`).
- **Issue tracking**: Link GitHub issues to checklist items for transparency.
- **Phase 2 PRs**: Each enhancement should be a separate PR with benchmarks in the description.

---

**Last Updated**: 2026-06-12 11:00 PDT  
**Maintained By**: AG Khan  
**Contact**: [GitHub Issues](https://github.com/agkhan/slm-domain-foundry/issues)