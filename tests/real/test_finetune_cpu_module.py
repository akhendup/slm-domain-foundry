"""Exercise train.finetune_cpu helpers with real tokenizers (no mocks)."""
from types import SimpleNamespace

import pytest
import torch

peft = pytest.importorskip("peft")
import train.finetune_cpu as fc

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_lora_target_modules_gpt2(tiny_lm_dir):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(tiny_lm_dir))
    assert fc._lora_target_modules(model) == ["c_attn"]


def test_format_sharegpt_example_real_tokenizer(tiny_lm_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_lm_dir))
    row = {
        "conversations": [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant", "content": "Hypertension is chronic elevation of blood pressure."},
        ]
    }
    out = fc._format_sharegpt_example(row, tok)
    assert "text" in out
    assert len(out["text"]) > 0


def test_print_progress_callback_logs(capsys):
    cb = fc._PrintProgressCallback()
    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(epoch=0, global_step=1, max_steps=10)
    cb.on_epoch_begin(args, state, None)
    cb.on_log(args, state, None, logs={"loss": 1.5, "learning_rate": 1e-4})
    cb.on_step_end(args, state, None)
    cb.on_epoch_end(args, state, None)
    if torch.cuda.is_available():
        cb.on_evaluate(args, state, None)
    assert "step" in capsys.readouterr().out
