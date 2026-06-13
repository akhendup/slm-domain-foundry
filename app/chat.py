#!/usr/bin/env python3
"""
Simple Q&A demo: load your trained model and answer questions in the terminal.
Works on NVIDIA GPU (Unsloth), Mac (MPS), and CPU (Hugging Face transformers).

Also supports chatting with a local LLM (Ollama / llama.cpp / LM Studio) via
the OpenAI-compatible API — no trained model needed.

Usage:
  # Fine-tuned SLM from output_model/
  python -m app.chat --model-dir output_model
  python -m app.chat --model-dir output_model --interactive

  # Local LLM via Ollama (no model training needed)
  python -m app.chat --ollama --ollama-model llama3
  python -m app.chat --ollama --ollama-host http://localhost:11434 --ollama-model mistral

  # llama.cpp server
  python -m app.chat --ollama --ollama-host http://localhost:8080 --ollama-model local

  # LM Studio
  python -m app.chat --ollama --ollama-host http://localhost:1234 --ollama-model local
"""

import argparse
import sys
from pathlib import Path

from app.model_loader import load_model, generate_response


def _load_demo_questions() -> list[str]:
    from train.config import get_ui_list

    return get_ui_list(key="demo_questions")


def _chat_with_ollama(
    message: str,
    history: list,
    host: str,
    model: str,
    system_prompt: str = "You are a concise, helpful assistant. Answer questions directly.",
) -> str:
    """Send a message to an Ollama/llama.cpp/LM Studio server and return the reply."""
    try:
        import requests  # noqa: F401 — checked by local_llm_chat
    except ImportError:
        return "ERROR: 'requests' package required. Run: pip install requests"

    from app.ollama_client import format_local_llm_error, local_llm_chat

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    host = host.rstrip("/")
    url = f"{host}/v1/chat/completions"
    try:
        return local_llm_chat(host, model, messages)
    except ValueError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:
        return f"ERROR: {format_local_llm_error(exc, host=host, model=model, url=url)}"


def run_demo_ollama(host: str, model: str, interactive: bool = False):
    """Run demo or interactive chat against a local Ollama/llama.cpp/LM Studio server."""
    print(f"Using local LLM: {model} @ {host}\n")
    history = []

    if interactive:
        print("Ask questions (type 'quit' or 'exit' to stop).\n")
        while True:
            try:
                user = input("You: ").strip()
                if user.lower() in ("quit", "exit", "q"):
                    break
                if not user:
                    continue
                reply = _chat_with_ollama(user, history, host, model)
                print("Model:", reply, "\n")
                history.append((user, reply))
            except KeyboardInterrupt:
                break
        print("Goodbye.")
    else:
        print("Demo: asking sample questions.\n")
        for q in _load_demo_questions():
            reply = _chat_with_ollama(q, history, host, model)
            print("Q:", q)
            print("A:", reply, "\n")
            history.append((q, reply))


def run_demo(model_dir: Path, interactive: bool = False):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Loading model from {model_dir}...")
    model, tokenizer = load_model(model_dir)
    print("Model loaded.\n")

    if interactive:
        print("Ask questions (type 'quit' or 'exit' to stop).\n")
        while True:
            try:
                user = input("You: ").strip()
                if user.lower() in ("quit", "exit", "q"):
                    break
                if not user:
                    continue
                messages = [{"role": "user", "content": user}]
                reply = generate_response(model, tokenizer, messages)
                print("Model:", reply, "\n")
            except KeyboardInterrupt:
                break
        print("Goodbye.")
    else:
        print("Demo: asking sample questions.\n")
        for q in _load_demo_questions():
            messages = [{"role": "user", "content": q}]
            reply = generate_response(model, tokenizer, messages)
            print("Q:", q)
            print("A:", reply, "\n")


def main():
    p = argparse.ArgumentParser(description="Q&A demo with trained SLM or local LLM")
    p.add_argument("--model-dir", type=Path, default=Path("output_model"),
                   help="Directory containing the trained model (default: output_model)")
    p.add_argument("--interactive", action="store_true", help="Chat in the terminal")

    # Local LLM (Ollama / llama.cpp / LM Studio) options
    p.add_argument("--ollama", action="store_true",
                   help="Use a local OpenAI-compatible LLM server instead of a trained model")
    p.add_argument("--ollama-host", default="http://localhost:11434",
                   help="Local LLM server URL (default: http://localhost:11434)")
    p.add_argument("--ollama-model", default=None,
                   help="Model name on the local server (default: first model from ollama list)")
    args = p.parse_args()

    if args.ollama:
        from app.ollama_client import default_ollama_model
        ollama_model = args.ollama_model or default_ollama_model(args.ollama_host)
        if not ollama_model:
            print("ERROR: No --ollama-model given and no models found. Run: ollama pull <model>")
            sys.exit(1)
        run_demo_ollama(args.ollama_host, ollama_model, args.interactive)
    else:
        run_demo(args.model_dir, args.interactive)


if __name__ == "__main__":
    main()
