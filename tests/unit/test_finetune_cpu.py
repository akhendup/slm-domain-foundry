"""Unit tests for train/finetune_cpu.py — device detection, callback, formatter.

No model weights or GPU required; all heavy I/O is mocked.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import train.finetune_cpu as fc

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _get_device
# ---------------------------------------------------------------------------


class TestGetDevice:
    def test_returns_torch_device(self):
        assert isinstance(fc._get_device(), torch.device)

    def test_device_type_valid(self):
        assert fc._get_device().type in ("cpu", "cuda", "mps")

    def test_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert fc._get_device().type == "cuda"

    def test_cpu_when_no_cuda_no_mps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert fc._get_device().type == "cpu"

    def test_mps_when_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        monkeypatch.setattr(torch.backends, "mps", mock_mps, raising=False)
        assert fc._get_device().type == "mps"


# ---------------------------------------------------------------------------
# _format_sharegpt_example
# ---------------------------------------------------------------------------


class TestFormatShareGptExample:
    def _tok(self, text="formatted text"):
        tok = MagicMock()
        tok.apply_chat_template.return_value = text
        return tok

    def test_basic_format_returns_text(self):
        result = fc._format_sharegpt_example(
            {"conversations": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]},
            self._tok("hi hello"),
        )
        assert result == {"text": "hi hello"}

    def test_empty_conversations_returns_empty_text(self):
        assert fc._format_sharegpt_example({"conversations": []}, self._tok()) == {"text": ""}

    def test_missing_key_returns_empty_text(self):
        assert fc._format_sharegpt_example({}, self._tok()) == {"text": ""}

    def test_invalid_items_skipped(self):
        # Non-dict items and dicts missing keys are silently dropped
        tok = self._tok()
        result = fc._format_sharegpt_example(
            {"conversations": ["bad", {"role": "user"}, {"no_role": True}]},
            tok,
        )
        assert result == {"text": ""}
        tok.apply_chat_template.assert_not_called()

    def test_user_role_preserved(self):
        tok = self._tok("ok")
        fc._format_sharegpt_example(
            {"conversations": [{"role": "user", "content": "q"}]},
            tok,
        )
        messages = tok.apply_chat_template.call_args[0][0]
        assert messages[0]["role"] == "user"

    def test_non_user_role_becomes_assistant(self):
        tok = self._tok("ok")
        fc._format_sharegpt_example(
            {"conversations": [{"role": "gpt", "content": "a"}]},
            tok,
        )
        messages = tok.apply_chat_template.call_args[0][0]
        assert messages[0]["role"] == "assistant"

    def test_tokenizer_called_with_chat_template_kwargs(self):
        tok = self._tok("result")
        fc._format_sharegpt_example(
            {"conversations": [{"role": "user", "content": "x"}]},
            tok,
        )
        _, kwargs = tok.apply_chat_template.call_args
        assert kwargs.get("tokenize") is False
        assert kwargs.get("add_generation_prompt") is False


# ---------------------------------------------------------------------------
# _PrintProgressCallback
# ---------------------------------------------------------------------------


def _state(epoch=0, global_step=1, max_steps=10):
    return SimpleNamespace(epoch=epoch, global_step=global_step, max_steps=max_steps)


def _args(num_train_epochs=3):
    return SimpleNamespace(num_train_epochs=num_train_epochs)


class TestPrintProgressCallback:
    def test_on_epoch_begin_contains_epoch_number(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_epoch_begin(_args(), _state(epoch=0), None)
        assert "Epoch" in capsys.readouterr().out

    def test_on_log_shows_loss(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_log(_args(), _state(), None, logs={"loss": 0.4321})
        assert "loss=0.4321" in capsys.readouterr().out

    def test_on_log_shows_eval_loss(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_log(_args(), _state(), None, logs={"eval_loss": 0.1111})
        assert "eval_loss=0.1111" in capsys.readouterr().out

    def test_on_log_shows_learning_rate(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_log(_args(), _state(), None, logs={"learning_rate": 2e-4})
        out = capsys.readouterr().out
        assert "lr=" in out

    def test_on_log_empty_logs_prints_nothing(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_log(_args(), _state(), None, logs={})
        assert capsys.readouterr().out == ""

    def test_on_log_none_logs_prints_nothing(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_log(_args(), _state(), None, logs=None)
        assert capsys.readouterr().out == ""

    def test_on_step_end_shows_step_fraction(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_step_end(_args(), _state(global_step=5, max_steps=20), None)
        assert "5/20" in capsys.readouterr().out

    def test_on_step_end_zero_max_steps_shows_question_mark(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_step_end(_args(), _state(global_step=1, max_steps=0), None)
        assert "1/?" in capsys.readouterr().out

    def test_on_epoch_end_prints_complete(self, capsys):
        cb = fc._PrintProgressCallback()
        cb.on_epoch_end(_args(), _state(epoch=2), None)
        out = capsys.readouterr().out
        assert "complete" in out.lower() or "Epoch" in out

    def test_on_evaluate_clears_cuda_cache_when_available(self):
        cb = fc._PrintProgressCallback()
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_clear:
            cb.on_evaluate(_args(), _state(), None)
        mock_clear.assert_called_once()

    def test_on_evaluate_skips_cache_clear_when_no_cuda(self):
        cb = fc._PrintProgressCallback()
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.empty_cache") as mock_clear:
            cb.on_evaluate(_args(), _state(), None)
        mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# main() — argument validation (no model load required)
# ---------------------------------------------------------------------------


class TestMainArgValidation:
    def test_exits_1_when_train_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["finetune_cpu",
             "--train-file", str(tmp_path / "missing.jsonl"),
             "--val-file", str(tmp_path / "val.jsonl")],
        )
        with pytest.raises(SystemExit) as exc:
            fc.main()
        assert exc.value.code == 1

    def test_exits_1_when_val_file_missing(self, tmp_path, monkeypatch):
        train_file = tmp_path / "train.jsonl"
        train_file.write_text('{"conversations": []}\n', encoding="utf-8")
        monkeypatch.setattr(
            "sys.argv",
            ["finetune_cpu",
             "--train-file", str(train_file),
             "--val-file", str(tmp_path / "missing.jsonl")],
        )
        with pytest.raises(SystemExit) as exc:
            fc.main()
        assert exc.value.code == 1
