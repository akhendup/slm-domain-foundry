"""Live Ollama tests when a server is running (no mocks)."""
import pytest

from app.ollama_client import default_ollama_model, list_ollama_models, local_llm_chat

pytestmark = pytest.mark.real


def test_list_models_live(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama not running")
    names = list_ollama_models("http://localhost:11434")
    assert isinstance(names, list)


def test_chat_live(first_ollama_model):
    reply = local_llm_chat(
        "http://localhost:11434",
        first_ollama_model,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
        temperature=0.0,
    )
    assert isinstance(reply, str)
    assert len(reply) > 0


def test_default_model_live(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama not running")
    name = default_ollama_model()
    assert name
