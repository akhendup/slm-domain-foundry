"""Apple Silicon / MPS integration tests (skip when MPS unavailable)."""

import sys

import pytest
import torch

pytestmark = [pytest.mark.real, pytest.mark.mps]

pytest.importorskip("peft")


@pytest.fixture(scope="module")
def require_mps():
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")


def test_get_device_selects_mps(require_mps):
    from app.model_loader import _get_device

    assert _get_device().type == "mps"


def test_dtype_for_mps_is_float16(require_mps):
    from app.model_loader import _dtype_for_device

    assert _dtype_for_device(torch.device("mps")) == torch.float16


def test_finetune_cpu_get_device_selects_mps(require_mps):
    from train.finetune_cpu import _get_device

    assert _get_device().type == "mps"


def test_gradio_training_device_type_mps(require_mps):
    from app.gradio_ui import _training_device_type

    assert _training_device_type() == "mps"


def test_gradio_selects_finetune_cpu_on_mps(require_mps, monkeypatch):
    """Gradio must use finetune_cpu (HF Trainer + LoRA), not Unsloth, on Apple Silicon."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from app.gradio_ui import _training_device_type, _unsloth_available

    assert _training_device_type() == "mps"
    assert _unsloth_available() is False
    script = "train.finetune_unsloth" if _unsloth_available() else "train.finetune_cpu"
    assert script == "train.finetune_cpu"


@pytest.mark.slow
def test_finetune_cpu_one_step_on_mps(require_mps, real_sharegpt_jsonl, tmp_path):
    """Run one real finetune_cpu step on MPS and load the checkpoint back on MPS."""
    from train.finetune_cpu import _get_device, main

    assert _get_device().type == "mps"

    out_dir = tmp_path / "output_model"
    argv = [
        "finetune_cpu",
        "--train-file",
        str(real_sharegpt_jsonl["train"]),
        "--val-file",
        str(real_sharegpt_jsonl["val"]),
        "--output-dir",
        str(out_dir),
        "--model-name",
        "sshleifer/tiny-gpt2",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--save-steps",
        "1000",
        "--no-eval",
        "--max-seq-length",
        "128",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = old

    assert (out_dir / "config.json").exists() or (out_dir / "adapter_config.json").exists()

    from app.model_loader import generate_response, load_model

    model, tok = load_model(out_dir)
    assert next(model.parameters()).device.type == "mps"
    text = generate_response(
        model,
        tok,
        [{"role": "user", "content": "Say hello briefly."}],
        max_new_tokens=12,
        temperature=0.0,
    )
    assert isinstance(text, str)
    assert text.strip()


def test_generate_response_on_mps(require_mps, tiny_lm_dir):
    from app.model_loader import generate_response, load_model

    model, tokenizer = load_model(tiny_lm_dir)
    assert next(model.parameters()).device.type == "mps"
    reply = generate_response(
        model,
        tokenizer,
        [{"role": "user", "content": "Say hello in one word."}],
        max_new_tokens=8,
    )
    assert isinstance(reply, str)
    assert reply.strip()
