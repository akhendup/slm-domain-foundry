"""Apple Silicon / MPS integration tests (skip when MPS unavailable)."""

import pytest
import torch

pytestmark = [pytest.mark.real, pytest.mark.unit]


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
