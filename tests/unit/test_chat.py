"""Unit tests for demo/chat.py"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# run_demo — model dir missing
# ---------------------------------------------------------------------------

class TestRunDemoMissingDir:
    def test_exits_when_dir_missing(self, tmp_path):
        from demo.chat import run_demo
        nonexistent = tmp_path / "no_model"
        with pytest.raises(SystemExit):
            run_demo(nonexistent)


# ---------------------------------------------------------------------------
# run_demo — non-interactive mode
# ---------------------------------------------------------------------------

class TestRunDemoNonInteractive:
    def _setup(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        return mock_model, mock_tokenizer

    def test_calls_generate_three_times(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        gen_calls = []

        def fake_gen(m, t, msgs):
            gen_calls.append(msgs)
            return "Test response"

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", side_effect=fake_gen):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=False)

        assert len(gen_calls) == 3

    def test_prints_responses(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="My answer"):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=False)

        captured = capsys.readouterr()
        assert "My answer" in captured.out

    def test_prints_questions(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=False)

        captured = capsys.readouterr()
        assert "Q:" in captured.out

    def test_prints_loading_message(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=False)

        captured = capsys.readouterr()
        assert "Loading model" in captured.out or "Model loaded" in captured.out


# ---------------------------------------------------------------------------
# run_demo — interactive mode
# ---------------------------------------------------------------------------

class TestRunDemoInteractive:
    def _setup(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        return MagicMock(), MagicMock()

    def test_exits_on_quit(self, tmp_path):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"), \
             patch("builtins.input", side_effect=["What is CSUM?", "quit"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

    def test_exits_on_exit(self, tmp_path):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"), \
             patch("builtins.input", side_effect=["exit"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

    def test_exits_on_q(self, tmp_path):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"), \
             patch("builtins.input", side_effect=["q"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

    def test_skips_empty_input(self, tmp_path):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        gen_calls = []

        def fake_gen(m, t, msgs):
            gen_calls.append(msgs)
            return "Response"

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", side_effect=fake_gen), \
             patch("builtins.input", side_effect=["", "   ", "actual question", "quit"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

        # Only "actual question" triggers generate_response
        assert len(gen_calls) == 1

    def test_keyboard_interrupt_exits_cleanly(self, tmp_path):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"), \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            from demo.chat import run_demo
            # Should not raise
            run_demo(tmp_path, interactive=True)

    def test_prints_goodbye_on_exit(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="Response"), \
             patch("builtins.input", side_effect=["quit"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out or "goodbye" in captured.out.lower()

    def test_prints_model_response(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", return_value="The answer is 42"), \
             patch("builtins.input", side_effect=["What is the answer?", "quit"]):
            from demo.chat import run_demo
            run_demo(tmp_path, interactive=True)

        captured = capsys.readouterr()
        assert "The answer is 42" in captured.out


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_exits_when_no_model(self, tmp_path, monkeypatch):
        """main() with a nonexistent model-dir should call sys.exit(1)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "demo.chat", "--model-dir", str(tmp_path / "no_model")],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0
