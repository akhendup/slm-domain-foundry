"""NVIDIA CUDA integration tests (skip when CUDA unavailable)."""

import sys

import pytest
import torch

pytestmark = [pytest.mark.real, pytest.mark.gpu]

pytest.importorskip("peft")


@pytest.fixture(scope="module")
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")


def test_get_device_selects_cuda(require_cuda):
    from app.model_loader import _get_device

    assert _get_device().type == "cuda"


def test_dtype_for_cuda_is_bfloat16(require_cuda):
    from app.model_loader import _dtype_for_device

    assert _dtype_for_device(torch.device("cuda")) == torch.bfloat16


def test_finetune_cpu_get_device_selects_cuda(require_cuda):
    from train.finetune_cpu import _get_device

    assert _get_device().type == "cuda"


def test_gradio_training_device_type_cuda(require_cuda):
    from app.gradio_ui import _training_device_type

    assert _training_device_type() == "cuda"


def test_gradio_prefers_unsloth_when_cuda_and_installed(require_cuda):
    """On CUDA, Gradio uses finetune_unsloth only when Unsloth is importable."""
    from app.gradio_ui import _training_device_type, _unsloth_available

    assert _training_device_type() == "cuda"
    script = "train.finetune_unsloth" if _unsloth_available() else "train.finetune_cpu"
    assert script in ("train.finetune_unsloth", "train.finetune_cpu")


@pytest.mark.slow
def test_finetune_cpu_one_step_on_cuda(require_cuda, real_sharegpt_jsonl, tmp_path):
    """Run one real finetune_cpu step on CUDA and load the checkpoint back on GPU."""
    from train.finetune_cpu import _get_device, main

    assert _get_device().type == "cuda"

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
    assert next(model.parameters()).device.type == "cuda"
    text = generate_response(
        model,
        tok,
        [{"role": "user", "content": "Say hello briefly."}],
        max_new_tokens=12,
        temperature=0.0,
    )
    assert isinstance(text, str)
    assert text.strip()


def test_generate_response_on_cuda(require_cuda, tiny_lm_dir):
    from app.model_loader import generate_response, load_model

    model, tokenizer = load_model(tiny_lm_dir)
    assert next(model.parameters()).device.type == "cuda"
    reply = generate_response(
        model,
        tokenizer,
        [{"role": "user", "content": "Say hello in one word."}],
        max_new_tokens=8,
    )
    assert isinstance(reply, str)
    assert reply.strip()
