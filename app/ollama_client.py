"""
Helpers for Ollama and other local HTTP chat-completions servers (optional; self-hosted only).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def list_ollama_models(host: str, timeout: float = 5) -> List[str]:
    """Return model names from Ollama ``/api/tags``, or an empty list on failure."""
    import requests

    host = (host or "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def default_ollama_model(host: str = "http://localhost:11434") -> str:
    """First installed Ollama model, or empty string if none / server unreachable."""
    models = list_ollama_models(host)
    return models[0] if models else ""


def _model_installed(requested: str, available: List[str]) -> bool:
    if not requested:
        return False
    req = requested.strip()
    return any(req == n or req in n or n.startswith(req + ":") for n in available)


def format_local_llm_error(
    exc: Exception,
    *,
    host: str,
    model: str,
    url: str,
) -> str:
    """Turn HTTP errors into actionable messages (especially missing Ollama models)."""
    import requests

    host = (host or "http://localhost:11434").rstrip("/")
    model = (model or "").strip()

    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Cannot connect to local LLM at `{host}`. "
            "Start Ollama (`ollama serve`) or check the Server URL."
        )

    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        resp = exc.response
        body: Dict[str, Any] = {}
        try:
            body = resp.json()
        except Exception:
            pass

        err_msg = ""
        if isinstance(body.get("error"), dict):
            err_msg = str(body["error"].get("message", ""))
        elif isinstance(body.get("error"), str):
            err_msg = body["error"]
        err_lower = err_msg.lower()

        if resp.status_code == 404 and ("model" in err_lower and "not found" in err_lower):
            available = list_ollama_models(host)
            lines = [
                f"Model `{model or '(empty)'}` is not installed on Ollama at {host}.",
            ]
            if available:
                lines.append(f"Installed models: {', '.join(available)}")
                lines.append(f"Set **Model name** to one of these (e.g. `{available[0]}`).")
            else:
                lines.append("No models found. Install one, e.g. `ollama pull qwen3:8b`.")
            if model and model != "llama3":
                lines.append(f"Or pull this model: `ollama pull {model}`")
            elif model == "llama3":
                lines.append("The UI default `llama3` is only an example — use a model you have pulled.")
            return "\n".join(lines)

    return f"Error talking to local LLM at {url}: {exc}"


def local_llm_chat(
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: float = 60,
) -> str:
    """
    Chat via local HTTP ``/v1/chat/completions`` (common JSON shape used by Ollama, llama.cpp, LM Studio, vLLM).

    Raises ``requests.HTTPError`` / ``ConnectionError``; callers can use
    ``format_local_llm_error`` for user-facing text.
    """
    import requests

    host = (host or "http://localhost:11434").rstrip("/")
    model = (model or "").strip()
    if not model:
        available = list_ollama_models(host)
        hint = f" Installed: {', '.join(available)}" if available else ""
        raise ValueError(f"Model name is required.{hint}")

    available = list_ollama_models(host)
    if available and not _model_installed(model, available):
        raise ValueError(
            f"Model `{model}` is not installed. Installed models: {', '.join(available)}"
        )

    url = f"{host}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
