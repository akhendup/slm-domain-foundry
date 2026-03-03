#!/usr/bin/env python3
"""
Fine-tune a small language model using standard HuggingFace Trainer + PEFT LoRA.
Works on CPU, MPS (Mac Apple Silicon), and CUDA without requiring Unsloth.
Merges LoRA weights into the base model before saving so the output is a
standalone model loadable by demo/model_loader.py.

Usage:
  python -m train.finetune_cpu --train-file training_data/train_sharegpt.jsonl \
      --val-file training_data/val_sharegpt.jsonl
"""

import inspect
import json
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        TrainingArguments,
    )
    from trl import SFTTrainer
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install torch transformers datasets peft trl accelerate")
    raise SystemExit(1)

_sig = inspect.signature(TrainingArguments.__init__)
USE_EVAL_STRATEGY = "eval_strategy" in _sig.parameters


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_sharegpt(examples, tokenizer):
    """Convert ShareGPT conversations to text strings."""
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

    p = argparse.ArgumentParser(description="Fine-tune SLM with HF Trainer + PEFT (CPU/GPU)")
    p.add_argument("--train-file", type=Path, default=Path("training_data/train_sharegpt.jsonl"))
    p.add_argument("--val-file", type=Path, default=Path("training_data/val_sharegpt.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("output_model"))
    p.add_argument("--model-name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=8)
    args = p.parse_args()

    if not args.train_file.exists():
        print(f"Train file not found: {args.train_file}")
        raise SystemExit(1)
    if not args.val_file.exists():
        print(f"Val file not found: {args.val_file}")
        raise SystemExit(1)

    device = _get_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU is very slow. Use a GPU for practical fine-tuning.")

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # MPS supports float16 but not bfloat16; CUDA gets bfloat16; CPU gets float32
    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type in ("cpu", "mps"):
        model = model.to(device)

    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
    )

    def formatting_func(examples):
        return _format_sharegpt(examples, tokenizer)

    train_args_dict = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": 1,
        "eval_steps": args.save_steps,
        "load_best_model_at_end": True,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "fp16": device.type == "mps",
        "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
        "report_to": "none",
        # no_cuda=True lets the Trainer use MPS (if available) or CPU instead of CUDA
        "no_cuda": device.type != "cuda",
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
        formatting_func=formatting_func,
        args=TrainingArguments(**train_args_dict),
    )

    print("Training...")
    trainer.train()

    print("Merging LoRA weights and saving to", args.output_dir, "...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Saved to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
