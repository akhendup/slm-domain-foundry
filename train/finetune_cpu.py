#!/usr/bin/env python3
"""
Fine-tune a small language model using standard HuggingFace Trainer + PEFT LoRA.
Works on CPU, MPS (Mac Apple Silicon), and CUDA without requiring Unsloth.
Merges LoRA weights into the base model before saving so the output is a
standalone model loadable by app/model_loader.py.

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
        TrainerCallback,
        TrainingArguments,
    )
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
        _HAS_SFT_CONFIG = True
    except ImportError:
        _HAS_SFT_CONFIG = False
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: pip install torch transformers datasets peft trl accelerate")
    raise SystemExit(1)

_sig = inspect.signature(TrainingArguments.__init__)
USE_EVAL_STRATEGY = "eval_strategy" in _sig.parameters
_sft_sig = inspect.signature(SFTTrainer.__init__)
_SFT_USES_PROCESSING_CLASS = "processing_class" in _sft_sig.parameters
# TRL >=0.14 prefers dataset_text_field over formatting_func to avoid batching issues.
_SFT_USES_DATASET_TEXT_FIELD = "dataset_text_field" in _sft_sig.parameters


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

    def on_step_end(self, args, state, control, **kwargs):
        # Print step position on every step so the UI progress bar updates continuously.
        # Loss values come separately via on_log (every logging_steps).
        total = state.max_steps or "?"
        pct = f"{100 * state.global_step / state.max_steps:.0f}%" if state.max_steps else ""
        print(f"  step {state.global_step}/{total} {pct}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"  Epoch {int(state.epoch)} complete.", flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        """Free CUDA cache after every eval to avoid OOM on the next backward pass."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lora_target_modules(model) -> list[str]:
    """Pick LoRA target module names for the loaded architecture."""
    names = {n for n, _ in model.named_modules()}
    llama_like = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if llama_like.issubset(names):
        return sorted(llama_like)
    if any(n.endswith("c_attn") for n in names):
        return ["c_attn"]
    if any(n.endswith("q_proj") for n in names):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    raise ValueError(
        "Could not infer LoRA target modules for this model. "
        "Use a Llama/Mistral-style or GPT-2-style base checkpoint."
    )


def _format_sharegpt_example(example, tokenizer):
    """Convert a single ShareGPT example to a dict with a 'text' string."""
    from train.sharegpt_format import format_sharegpt_messages

    text = format_sharegpt_messages(example.get("conversations", []), tokenizer)
    return {"text": text}


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
    p.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT_DIR",
        help="Path to a checkpoint-N directory to resume training from.",
    )
    p.add_argument(
        "--no-eval", action="store_true",
        help="Disable evaluation during training to save memory (useful on CPU with limited RAM).",
    )
    p.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing to reduce memory usage (slower but uses much less RAM/VRAM).",
    )
    args = p.parse_args()

    if not args.train_file.exists():
        print(f"Train file not found: {args.train_file}")
        raise SystemExit(1)
    if not args.val_file.exists():
        print(f"Val file not found: {args.val_file}")
        raise SystemExit(1)

    device = _get_device()
    print(f"Device: {device}", flush=True)
    if device.type == "cpu":
        print("WARNING: Training on CPU is very slow. Use a GPU for practical fine-tuning.", flush=True)

    print(f"Loading base model: {args.model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # MPS supports float16 but not bfloat16; CUDA gets bfloat16; CPU gets float32
    if device.type == "cuda":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type in ("cpu", "mps"):
        model = model.to(device)

    # Align model config with tokenizer to suppress pad_token_id mismatch warnings.
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("Model loaded.", flush=True)
    print("Adding LoRA adapters...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=_lora_target_modules(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("Loading dataset...", flush=True)
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
    )

    # Pre-process dataset into a "text" column (one string per example).
    # This avoids TRL's formatting_func batching inconsistencies in newer versions
    # where add_eos expects example["text"] to be a string, not a list.
    def _fmt(example):
        return _format_sharegpt_example(example, tokenizer)

    dataset = dataset.map(_fmt, remove_columns=["conversations"])
    # Drop empty examples that failed formatting
    dataset = dataset.filter(lambda ex: len(ex["text"]) > 0)

    train_args_dict = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_steps": 5,
        "logging_steps": 10,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "eval_steps": args.save_steps,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "fp16": device.type == "mps",
        "bf16": device.type == "cuda" and torch.cuda.is_bf16_supported(),
        "report_to": "none",
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    # no_cuda was removed in Transformers >=4.46; use_cpu replaced it for CPU-only mode.
    # For MPS the Trainer auto-detects it in newer versions without any flag.
    if "no_cuda" in _sig.parameters:
        train_args_dict["no_cuda"] = device.type != "cuda"
    elif "use_cpu" in _sig.parameters and device.type == "cpu":
        train_args_dict["use_cpu"] = True
    if args.no_eval:
        if USE_EVAL_STRATEGY:
            train_args_dict["eval_strategy"] = "no"
        else:
            train_args_dict["evaluation_strategy"] = "no"
    else:
        if USE_EVAL_STRATEGY:
            train_args_dict["eval_strategy"] = "steps"
        else:
            train_args_dict["evaluation_strategy"] = "steps"

    # Build SFTTrainer kwargs compatible with both old TRL (<0.10) and new TRL (>=0.10).
    # In TRL >=0.10 max_seq_length moved into SFTConfig; tokenizer was renamed processing_class.
    # We never use formatting_func — the dataset already has a pre-processed "text" column.
    # dataset_text_field tells TRL which column to use; older TRL defaults to "text" anyway.
    _trainer_kwargs: dict = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": None if args.no_eval else dataset["validation"],
        "callbacks": [_PrintProgressCallback()],
    }
    if _SFT_USES_DATASET_TEXT_FIELD:
        # SFTTrainer accepts it directly (older TRL without SFTConfig)
        _trainer_kwargs["dataset_text_field"] = "text"
    if _SFT_USES_PROCESSING_CLASS:
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        _trainer_kwargs["tokenizer"] = tokenizer

    if _HAS_SFT_CONFIG:
        # TRL >=0.10: max_seq_length moved into SFTConfig.
        # TRL >=0.14 renamed it to max_length — detect at runtime.
        # SFTConfig overrides __init__ so filter train_args_dict against its own
        # signature to drop any params (e.g. no_cuda) it no longer accepts.
        _sft_config_sig = inspect.signature(SFTConfig.__init__)
        _sft_params = set(_sft_config_sig.parameters.keys())
        filtered_train_args = {k: v for k, v in train_args_dict.items() if k in _sft_params}
        if "max_seq_length" in _sft_config_sig.parameters:
            seq_len_kwarg = {"max_seq_length": args.max_seq_length}
        elif "max_length" in _sft_config_sig.parameters:
            seq_len_kwarg = {"max_length": args.max_seq_length}
        else:
            seq_len_kwarg = {}
        # If SFTConfig accepts dataset_text_field, set it there (move from trainer_kwargs if present)
        if "dataset_text_field" in _sft_config_sig.parameters:
            _trainer_kwargs.pop("dataset_text_field", None)
            seq_len_kwarg["dataset_text_field"] = "text"
        _trainer_kwargs["args"] = SFTConfig(**seq_len_kwarg, **filtered_train_args)
    else:
        _trainer_kwargs["max_seq_length"] = args.max_seq_length
        _trainer_kwargs["args"] = TrainingArguments(**train_args_dict)

    trainer = SFTTrainer(**_trainer_kwargs)

    n_train = len(dataset["train"])
    steps_per_epoch = max(1, n_train // (args.batch_size * args.grad_accum))
    print(f"Starting training: {n_train} examples, ~{steps_per_epoch} steps/epoch, "
          f"{args.epochs} epoch(s)", flush=True)

    resume_from = args.resume if args.resume and Path(args.resume).exists() else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)

    print("Merging LoRA weights and saving ...", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("Saved to", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
