"""Unit tests for app/ollama_client.py."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.ollama_client import (
    _model_installed,
    default_ollama_model,
    format_local_llm_error,
    list_ollama_models,
)


class TestModelInstalled:
    def test_exact_match(self):
        assert _model_installed("qwen3:8b", ["qwen3:8b", "qwen3:14b"])

    def test_prefix_tag_match(self):
        assert _model_installed("qwen3", ["qwen3:14b"])

    def test_missing(self):
        assert not _model_installed("llama3", ["qwen3:8b"])


class TestListOllamaModels:
    def test_parses_tags(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3:8b"}, {"name": "qwen3:14b"}],
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            assert list_ollama_models("http://localhost:11434") == ["qwen3:8b", "qwen3:14b"]

    def test_returns_empty_on_failure(self):
        with patch("requests.get", side_effect=requests.ConnectionError):
            assert list_ollama_models("http://localhost:11434") == []


class TestDefaultOllamaModel:
    def test_first_model(self):
        with patch("app.ollama_client.list_ollama_models", return_value=["qwen3:14b", "qwen3:8b"]):
            assert default_ollama_model() == "qwen3:14b"

    def test_empty_when_none(self):
        with patch("app.ollama_client.list_ollama_models", return_value=[]):
            assert default_ollama_model() == ""


class TestFormatLocalLlmError:
    def test_model_not_found_404(self):
        resp = MagicMock()
        resp.status_code = 404
        resp.json.return_value = {
            "error": {"message": "model 'llama3' not found", "type": "not_found_error"},
        }
        err = requests.HTTPError(response=resp)
        with patch("app.ollama_client.list_ollama_models", return_value=["qwen3:8b"]):
            msg = format_local_llm_error(
                err, host="http://localhost:11434", model="llama3", url="http://x/v1/chat/completions"
            )
        assert "llama3" in msg
        assert "qwen3:8b" in msg
        assert "not installed" in msg.lower() or "not installed" in msg

    def test_connection_error(self):
        msg = format_local_llm_error(
            requests.ConnectionError(),
            host="http://localhost:11434",
            model="x",
            url="http://x/v1/chat/completions",
        )
        assert "Cannot connect" in msg
