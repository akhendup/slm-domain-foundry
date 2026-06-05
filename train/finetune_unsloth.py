#!/usr/bin/env python3
"""
Fine-tune a small language model (e.g. TinyLlama, Llama-3.2-1B) on your data with Unsloth.
Uses ShareGPT-format JSONL by default. Run after preparing data with data.prepare_training_data.

Usage:
  python -m train.finetune_unsloth --train-file training_data/train_sharegpt.jsonl --val-file training_data/val_sharegpt.jsonl
  python -m train.finetune_unsloth --train-file training_data/train_sharegpt.jsonl --val-file training_data/val_sharegpt.jsonl --model-name unsloth/tinyllama-chat-bnb-4bit
"""

import inspect
import os
from pathlib import Path

from train.sharegpt_format import make_sharegpt_formatting_func


def _apply_stable_unsloth_env() -> None:
    """Conservative Unsloth settings for 12GB GPUs (avoids segfaults from fast paths)."""
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")


def _ensure_unsloth_compile_location() -> None:
    """Use a user cache dir, not repo unsloth_compiled_cache (avoids TRL pickle errors)."""
    if os.environ.get("UNSLOTH_COMPILE_LOCATION"):
        return
    repo_cache = Path(__file__).resolve().parents[1] / "unsloth_compiled_cache"
    if repo_cache.is_dir():
        os.environ["UNSLOTH_COMPILE_LOCATION"] = str(Path.home() / ".cache" / "unsloth_compiled")


def _use_eval_strategy_key() -> bool:
    from transformers import TrainingArguments

    return "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters


def _make_progress_callback(TrainerCallback):
    """Build a TrainerCallback that prints step/epoch progress to stdout."""

    class _PrintProgressCallback(TrainerCallback):
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

    return _PrintProgressCallback()


def _make_formatting_func(tok):
    """Return a ShareGPT formatting function bound to *tok* (avoids globals)."""
    return make_sharegpt_formatting_func(tok)


def main():
    import argparse

    # Parse --stable early so env vars are set before importing unsloth.
    _early = argparse.ArgumentParser(add_help=False)
    _early.add_argument("--stable", action="store_true")
    _early_args, _ = _early.parse_known_args()
    if _early_args.stable:
        _apply_stable_unsloth_env()

    _ensure_unsloth_compile_location()
    try:
        import unsloth  # noqa: F401 — must import before transformers/trl
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import get_chat_template
        from datasets import load_dataset
        from transformers import TrainerCallback
        from trl import SFTTrainer
        try:
            from trl import SFTConfig
        except ImportError:
            SFTConfig = None  # type: ignore[misc, assignment]
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install unsloth torch datasets transformers trl")
        raise SystemExit(1)

    progress_cb = _make_progress_callback(TrainerCallback)
    p = argparse.ArgumentParser(description="Fine-tune SLM with Unsloth")
    p.add_argument(
        "--stable",
        action="store_true",
        help="12GB-safe mode: disable padding-free/double-buffer, standard grad checkpointing",
    )
    p.add_argument("--train-file", type=Path, default=Path("training_data/train_sharegpt.jsonl"))
    p.add_argument("--val-file", type=Path, default=Path("training_data/val_sharegpt.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("output_model"))
    p.add_argument(
        "--model-name",
        type=str,
        default="unsloth/tinyllama-chat-bnb-4bit",
        help="Hugging Face model id (e.g. unsloth/tinyllama-chat-bnb-4bit or TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 avoids CUDA fork segfaults under Slurm)",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Checkpoint every N steps (0 = no intermediate checkpoints; final merged model still saved)",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip validation during training (recommended on 12GB GPUs — eval can segfault after long val runs)",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Run validation every N steps (ignored when --no-eval)",
    )
    args = p.parse_args()

    if args.stable:
        _apply_stable_unsloth_env()
        if args.max_seq_length > 512:
            print(f"Stable mode: capping max_seq_length {args.max_seq_length} -> 512", flush=True)
            args.max_seq_length = 512
        if args.batch_size > 1:
            print(f"Stable mode: capping batch_size {args.batch_size} -> 1", flush=True)
            args.batch_size = 1
        print("Stable mode enabled (12GB GPU profile).", flush=True)

    if not args.train_file.exists():
        print(f"Train file not found: {args.train_file}")
        print("Run: python -m data.prepare_training_data --pdf-dir <dir> or --csv <file>")
        raise SystemExit(1)
    if not args.no_eval and not args.val_file.exists():
        print(f"Val file not found: {args.val_file}")
        raise SystemExit(1)

    print("Loading model:", args.model_name, flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml" if "tinyllama" in args.model_name.lower() else "llama-3.1",
    )

    print("Model loaded.", flush=True)
    print("Adding LoRA adapters...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True if args.stable else "unsloth",
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
        "save_total_limit": 2,
        "dataloader_num_workers": args.dataloader_workers,
        "dataloader_pin_memory": False,
    }
    if args.save_steps > 0:
        train_args_dict["save_steps"] = args.save_steps
        train_args_dict["save_strategy"] = "steps"
    else:
        train_args_dict["save_strategy"] = "no"

    if args.no_eval:
        if _use_eval_strategy_key():
            train_args_dict["eval_strategy"] = "no"
        else:
            train_args_dict["evaluation_strategy"] = "no"
        print("Evaluation disabled (--no-eval).", flush=True)
    else:
        train_args_dict["eval_steps"] = args.eval_steps
        train_args_dict["per_device_eval_batch_size"] = 1
        if _use_eval_strategy_key():
            train_args_dict["eval_strategy"] = "steps"
        else:
            train_args_dict["evaluation_strategy"] = "steps"

    _sft_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "formatting_func": _make_formatting_func(tokenizer),
        "callbacks": [progress_cb],
    }
    if not args.no_eval:
        trainer_kwargs["eval_dataset"] = dataset["validation"]
    if "processing_class" in _sft_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    if SFTConfig is not None:
        sft_sig = inspect.signature(SFTConfig.__init__)
        sft_params = set(sft_sig.parameters.keys())
        filtered = {k: v for k, v in train_args_dict.items() if k in sft_params}
        if "max_seq_length" in sft_params:
            filtered["max_seq_length"] = args.max_seq_length
        elif "max_length" in sft_params:
            filtered["max_length"] = args.max_seq_length
        if "padding_free" in sft_params:
            filtered["padding_free"] = False
        trainer_kwargs["args"] = SFTConfig(**filtered)
    else:
        from transformers import TrainingArguments

        trainer_kwargs["max_seq_length"] = args.max_seq_length
        trainer_kwargs["args"] = TrainingArguments(**train_args_dict)

    trainer = SFTTrainer(**trainer_kwargs)

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
