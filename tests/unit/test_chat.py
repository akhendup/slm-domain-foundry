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

    def test_calls_generate_for_each_demo_question(self, tmp_path, capsys):
        mock_model, mock_tokenizer = self._setup(tmp_path)
        gen_calls = []

        def fake_gen(m, t, msgs):
            gen_calls.append(msgs)
            return "Test response"

        with patch("demo.chat.load_model", return_value=(mock_model, mock_tokenizer)), \
             patch("demo.chat.generate_response", side_effect=fake_gen):
            from demo.chat import run_demo, _DEMO_QUESTIONS
            run_demo(tmp_path, interactive=False)

        assert len(gen_calls) == len(_DEMO_QUESTIONS)

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


# ---------------------------------------------------------------------------
# run_demo_ollama
# ---------------------------------------------------------------------------

class TestRunDemoOllama:
    def _fake_chat(self, *args, **kwargs):
        return "Ollama response"

    def test_calls_chat_for_each_demo_question(self, capsys):
        from unittest.mock import patch
        from demo.chat import run_demo_ollama, _DEMO_QUESTIONS
        with patch("demo.chat._chat_with_ollama", return_value="Ollama response") as mock_fn:
            run_demo_ollama("http://localhost:11434", "llama3", interactive=False)
        assert mock_fn.call_count == len(_DEMO_QUESTIONS)

    def test_prints_responses(self, capsys):
        from unittest.mock import patch
        from demo.chat import run_demo_ollama
        with patch("demo.chat._chat_with_ollama", return_value="The CSUM answer"):
            run_demo_ollama("http://localhost:11434", "llama3", interactive=False)
        captured = capsys.readouterr()
        assert "The CSUM answer" in captured.out

    def test_interactive_exits_on_quit(self, capsys):
        from unittest.mock import patch
        from demo.chat import run_demo_ollama
        with patch("demo.chat._chat_with_ollama", return_value="Response"), \
             patch("builtins.input", side_effect=["What is CSUM?", "quit"]):
            run_demo_ollama("http://localhost:11434", "llama3", interactive=True)

    def test_interactive_exits_on_keyboard_interrupt(self, capsys):
        from unittest.mock import patch
        from demo.chat import run_demo_ollama
        with patch("demo.chat._chat_with_ollama", return_value="Response"), \
             patch("builtins.input", side_effect=KeyboardInterrupt):
            run_demo_ollama("http://localhost:11434", "llama3", interactive=True)


# ---------------------------------------------------------------------------
# _chat_with_ollama — unit tests
# ---------------------------------------------------------------------------

class TestChatWithOllama:
    def test_returns_reply_on_success(self):
        from unittest.mock import MagicMock, patch
        from demo.chat import _chat_with_ollama

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "CSUM computes a cumulative sum."}}]
        }
        with patch("requests.post", return_value=mock_resp):
            result = _chat_with_ollama("What is CSUM?", [], "http://localhost:11434", "llama3")
        assert result == "CSUM computes a cumulative sum."

    def test_returns_error_message_on_connection_error(self):
        from unittest.mock import patch
        import requests as req
        from demo.chat import _chat_with_ollama

        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            result = _chat_with_ollama("Q?", [], "http://localhost:11434", "llama3")
        assert "ERROR" in result or "Cannot connect" in result

    def test_includes_history_in_messages(self):
        from unittest.mock import MagicMock, patch, call
        from demo.chat import _chat_with_ollama

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
        history = [("previous question", "previous answer")]
        with patch("requests.post", return_value=mock_resp) as mock_post:
            _chat_with_ollama("follow-up", history, "http://localhost:11434", "llama3")
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
        roles = [m["role"] for m in payload["messages"]]
        assert "user" in roles
        assert "assistant" in roles
