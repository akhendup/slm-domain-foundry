"""Unsloth module callbacks and CLI guards without loading weights."""
import sys
from types import SimpleNamespace

import pytest

import train.finetune_unsloth as fu

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_print_progress_callback(capsys):
    cb = fu._PrintProgressCallback()
    args = SimpleNamespace(num_train_epochs=1)
    state = SimpleNamespace(epoch=0, global_step=2, max_steps=5)
    cb.on_epoch_begin(args, state, None)
    cb.on_log(args, state, None, logs={"loss": 0.5})
    cb.on_epoch_end(args, state, None)
    assert "Epoch" in capsys.readouterr().out


def test_main_missing_val_file(tmp_path, real_sharegpt_jsonl):
    argv = [
        "finetune_unsloth",
        "--train-file",
        str(real_sharegpt_jsonl["train"]),
        "--val-file",
        str(tmp_path / "no_val.jsonl"),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit) as exc:
            fu.main()
        assert exc.value.code == 1
    finally:
        sys.argv = old
