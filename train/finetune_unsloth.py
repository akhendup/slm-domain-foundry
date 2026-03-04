#!/usr/bin/env python3
"""
Fine-tune a small language model (e.g. TinyLlama, Llama-3.2-1B) on your data with Unsloth.
Uses ShareGPT-format JSONL by default. Run after preparing data with data.prepare_training_data.

Usage:
  python -m train.finetune_unsloth --train-file training_data/train_sharegpt.jsonl --val-file training_data/val_sharegpt.jsonl
  python -m train.finetune_unsloth --train-file training_data/train_sharegpt.jsonl --val-file training_data/val_sharegpt.jsonl --model-name unsloth/TinyLlama-1.1b-Chat-v1.0
"""

import inspect
import json
import os
from pathlib import Path

try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainerCallback, TrainingArguments
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install unsloth torch datasets transformers trl")
    raise SystemExit(1)

# Detect TrainingArguments API
_sig = inspect.signature(TrainingArguments.__init__)
USE_EVAL_STRATEGY = "eval_strategy" in _sig.parameters


class _PrintProgressCallback(TrainerCallback):
    """Print step/epoch progress to stdout so it streams through the UI."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) + 1
        print(f"\n--- Epoch {epoch}/{int(args.num_train_epochs)} ---", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        total = state.max_steps or "?"
        pct = f"{100 * state.global_step / state.max_steps:.0f}%" if state.max_steps else ""
        print(f"  step {state.global_step}/{total} {pct}  {' | '.join(parts)}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"  Epoch {int(state.epoch)} complete.", flush=True)


def formatting_func_sharegpt(examples):
    """Format ShareGPT conversations for the trainer."""
    conversations = examples.get("conversations", [])
    if not conversations:
        return []
    if isinstance(conversations[0], dict) and "role" in conversations[0]:
        conversations = [conversations]
    texts = []
    for convo in conversations:
        if not convo:
            continue
        messages = []
        for msg in convo:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = "user" if msg["role"] == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
        if messages:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
    return texts


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune SLM with Unsloth")
    p.add_argument("--train-file", type=Path, default=Path("training_data/train_sharegpt.jsonl"))
    p.add_argument("--val-file", type=Path, default=Path("training_data/val_sharegpt.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("output_model"))
    p.add_argument("--model-name", type=str, default="unsloth/TinyLlama-1.1b-Chat-v1.0")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=50)
    args = p.parse_args()

    if not args.train_file.exists():
        print(f"Train file not found: {args.train_file}")
        print("Run: python -m data.prepare_training_data --pdf-dir <dir> or --csv <file>")
        raise SystemExit(1)
    if not args.val_file.exists():
        print(f"Val file not found: {args.val_file}")
        raise SystemExit(1)

    global tokenizer
    print("Loading model:", args.model_name, flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml" if "TinyLlama" in args.model_name else "llama-3.1")

    print("Model loaded.", flush=True)
    print("Adding LoRA adapters...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading dataset...", flush=True)
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
    )

    train_args_dict = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": 10,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 42,
        "output_dir": str(args.output_dir),
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "eval_steps": 50,
    }
    if USE_EVAL_STRATEGY:
        train_args_dict["eval_strategy"] = "steps"
    else:
        train_args_dict["evaluation_strategy"] = "steps"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_func_sharegpt,
        args=TrainingArguments(**train_args_dict),
        callbacks=[_PrintProgressCallback()],
    )

    n_train = len(dataset["train"])
    steps_per_epoch = max(1, n_train // (args.batch_size * args.grad_accum))
    print(f"Starting training: {n_train} examples, ~{steps_per_epoch} steps/epoch, "
          f"{args.epochs} epoch(s)", flush=True)
    trainer.train()
    print("Saving merged model to", args.output_dir, "...", flush=True)
    model.save_pretrained_merged(str(args.output_dir), tokenizer, save_method="merged_16bit")
    print("Saved to", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
