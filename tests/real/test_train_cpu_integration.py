"""Run finetune_cpu on real JSONL from sample_data with a tiny base model (in-process for coverage)."""
import sys

import pytest

pytestmark = [pytest.mark.real, pytest.mark.slow]


def test_finetune_cpu_one_step(real_sharegpt_jsonl, tmp_path):
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
        from train.finetune_cpu import main

        assert main() == 0
    finally:
        sys.argv = old

    assert (out_dir / "config.json").exists()

    from app.model_loader import generate_response, load_model

    model, tok = load_model(out_dir)
    text = generate_response(
        model,
        tok,
        [{"role": "user", "content": "What is CSUM?"}],
        max_new_tokens=12,
        temperature=0.0,
    )
    assert isinstance(text, str)
