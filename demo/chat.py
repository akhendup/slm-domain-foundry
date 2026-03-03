#!/usr/bin/env python3
"""
Simple Q&A demo: load your trained model and answer questions in the terminal.
Works on NVIDIA GPU (Unsloth), Mac (MPS), and CPU (Hugging Face transformers).

Usage:
  python -m demo.chat --model-dir output_model
  python -m demo.chat --model-dir output_model --interactive
"""

import argparse
import sys
from pathlib import Path

from demo.model_loader import load_model, generate_response


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
        demo_questions = [
            "What is this document about?",
            "Summarize the main points.",
            "How do I get started?",
        ]
        print("Demo: asking a few sample questions.\n")
        for q in demo_questions:
            messages = [{"role": "user", "content": q}]
            reply = generate_response(model, tokenizer, messages)
            print("Q:", q)
            print("A:", reply, "\n")


def main():
    p = argparse.ArgumentParser(description="Q&A demo with trained SLM")
    p.add_argument("--model-dir", type=Path, default=Path("output_model"))
    p.add_argument("--interactive", action="store_true", help="Chat in the terminal")
    args = p.parse_args()
    run_demo(args.model_dir, args.interactive)


if __name__ == "__main__":
    main()
