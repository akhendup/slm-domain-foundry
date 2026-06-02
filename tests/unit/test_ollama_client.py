"""Unit tests for app/ollama_client.py using live Ollama when available."""
import pytest
import requests

from app.ollama_client import (
    _model_installed,
    default_ollama_model,
    format_local_llm_error,
    list_ollama_models,
    local_llm_chat,
)

pytestmark = pytest.mark.unit


class TestModelInstalled:
    def test_exact_match(self):
        assert _model_installed("qwen3:8b", ["qwen3:8b", "qwen3:14b"])

    def test_prefix_tag_match(self):
        assert _model_installed("qwen3", ["qwen3:14b"])

    def test_missing(self):
        assert not _model_installed("llama3", ["qwen3:8b"])


class TestListOllamaModels:
    def test_live_tags(self, ollama_available):
        if not ollama_available:
            pytest.skip("Ollama not running")
        names = list_ollama_models("http://localhost:11434")
        assert isinstance(names, list)


class TestDefaultOllamaModel:
    def test_live_default(self, ollama_available):
        if not ollama_available:
            pytest.skip("Ollama not running")
        name = default_ollama_model()
        if list_ollama_models("http://localhost:11434"):
            assert name


class TestFormatLocalLlmError:
    def test_connection_error(self):
        msg = format_local_llm_error(
            requests.ConnectionError(),
            host="http://localhost:11434",
            model="x",
            url="http://localhost:11434/v1/chat/completions",
        )
        assert "Cannot connect" in msg

    def test_live_missing_model(self, ollama_available):
        if not ollama_available:
            pytest.skip("Ollama not running")
        try:
            local_llm_chat(
                "http://localhost:11434",
                "this-model-does-not-exist-xyz",
                [{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
        except ValueError as exc:
            assert "not installed" in str(exc)
        except requests.HTTPError as exc:
            msg = format_local_llm_error(
                exc,
                host="http://localhost:11434",
                model="this-model-does-not-exist-xyz",
                url="http://localhost:11434/v1/chat/completions",
            )
            assert "not installed" in msg.lower() or "model" in msg.lower()
