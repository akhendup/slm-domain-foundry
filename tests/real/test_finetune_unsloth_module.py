"""Test Unsloth training module surface without requiring a GPU run."""
import pytest

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_make_formatting_func(tiny_lm_dir):
    from transformers import AutoTokenizer

    import train.finetune_unsloth as fu

    tok = AutoTokenizer.from_pretrained(str(tiny_lm_dir))
    fn = fu._make_formatting_func(tok)
    texts = fn(
        {
            "conversations": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "There"},
            ]
        }
    )
    assert len(texts) == 1


def test_main_missing_train_file(tmp_path):
    import sys

    import train.finetune_unsloth as fu

    argv = [
        "finetune_unsloth",
        "--train-file",
        str(tmp_path / "missing_train.jsonl"),
        "--val-file",
        str(tmp_path / "missing_val.jsonl"),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as exc:
            fu.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old
