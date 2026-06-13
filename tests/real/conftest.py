"""
Fixtures for integration tests: real sample_data, real HuggingFace tiny models, real JSONL.
No unittest.mock — only tmp_path I/O and optional live services (Ollama).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA = PROJECT_ROOT / "sample_data"
TINY_HF_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_data_dir() -> Path:
    assert SAMPLE_DATA.is_dir(), f"Missing sample_data at {SAMPLE_DATA}"
    return SAMPLE_DATA


@pytest.fixture(scope="session")
def tiny_lm_dir(tmp_path_factory) -> Path:
    """Real tiny causal LM saved to disk (loadable by app.model_loader.load_model)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = tmp_path_factory.mktemp("tiny_lm")
    if (out / "config.json").exists():
        return out
    tok = AutoTokenizer.from_pretrained(TINY_HF_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_HF_MODEL)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def real_sharegpt_jsonl(tmp_path_factory, sample_data_dir) -> dict[str, Path]:
    """Build real train/val ShareGPT JSONL from sample_data via prepare_training_data."""
    out = tmp_path_factory.mktemp("training_data")
    csv_path = sample_data_dir / "sample_qa.csv"
    yaml_dir = sample_data_dir / "patternexamples"
    cmd = [
        sys.executable,
        "-m",
        "data.prepare_training_data",
        "--csv",
        str(csv_path),
        "--yaml-dir",
        str(yaml_dir),
        "--output-dir",
        str(out),
        "--format",
        "sharegpt",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    train = out / "train_sharegpt.jsonl"
    val = out / "val_sharegpt.jsonl"
    assert train.exists() and val.exists()
    lines = train.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    return {"dir": out, "train": train, "val": val}


@pytest.fixture
def real_memory_dir(tmp_path) -> Path:
    """Conversation memory directory with one real logged interaction."""
    from data.conversation_memory import log_interaction

    mem = tmp_path / "conversation_memory"
    log_interaction(
        mem,
        question="What is hypertension?",
        answer="Hypertension is chronic elevation of blood pressure.",
        model_name="tiny-gpt2",
    )
    assert (mem / "interactions.jsonl").exists()
    return mem


@pytest.fixture
def first_ollama_model(ollama_available) -> str:
    if not ollama_available:
        pytest.skip("Ollama not running at localhost:11434")
    from app.ollama_client import list_ollama_models

    names = list_ollama_models("http://localhost:11434")
    if not names:
        pytest.skip("Ollama has no models installed")
    return names[0]
