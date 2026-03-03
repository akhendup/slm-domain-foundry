"""
End-to-end Gradio UI: Prepare Data → Train → Chat.
All three stages run inside the same container; training streams live logs.

Usage:
  python run_gradio_ui.py --host 0.0.0.0          # Docker (all defaults)
  python run_gradio_ui.py --model-dir output_model  # local override
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import gradio as gr
import torch

from demo.model_loader import generate_response, load_model

# ---------------------------------------------------------------------------
# Default paths (Docker layout; overridable via CLI)
# ---------------------------------------------------------------------------
_DATA_DIR = Path("/app/data")
_TRAINING_DATA_DIR = Path("/app/training_data")
_OUTPUT_MODEL_DIR = Path("/app/output_model")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None


def _get_device_label() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA — {name}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    return "CPU"


def _unsloth_available() -> bool:
    try:
        if torch.cuda.is_available():
            import unsloth  # noqa: F401
            return True
    except Exception:
        pass
    return False


def _model_ready() -> bool:
    return (_OUTPUT_MODEL_DIR / "config.json").exists()


# ---------------------------------------------------------------------------
# Tab 1 — Data Preparation
# ---------------------------------------------------------------------------

def _run_data_prep(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
    val_ratio: float,
    fmt: str,
) -> Generator[str, None, None]:
    yield "Starting data preparation...\n"

    if not uploaded_files:
        yield "ERROR: No files uploaded. Please upload at least one PDF or CSV.\n"
        return

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pdfs, csvs = [], []
    for f in uploaded_files:
        src = Path(f)
        dest = _DATA_DIR / src.name
        shutil.copy2(src, dest)
        if src.suffix.lower() == ".csv":
            csvs.append(str(dest))
        else:
            pdfs.append(str(dest))
    yield f"Copied {len(pdfs)} PDF(s) and {len(csvs)} CSV file(s) to {_DATA_DIR}\n"

    cmd = [
        sys.executable, "-m", "data.prepare_training_data",
        "--output-dir", str(_TRAINING_DATA_DIR),
        "--format", fmt,
        "--chunk-size", str(chunk_size),
        "--chunk-overlap", str(chunk_overlap),
        "--val-ratio", str(val_ratio),
    ]
    if pdfs:
        cmd += ["--pdf-dir", str(_DATA_DIR)]
    if csvs:
        cmd += ["--csv", csvs[0]]

    yield f"Running: {' '.join(cmd)}\n\n"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    if proc.returncode == 0:
        yield "\nData preparation complete.\n"
        files = sorted(_TRAINING_DATA_DIR.glob("*.jsonl"))
        if files:
            yield "Output files:\n" + "\n".join(f"  {f.name}" for f in files) + "\n"
    else:
        yield f"\nData preparation FAILED (exit code {proc.returncode}).\n"


# ---------------------------------------------------------------------------
# Tab 2 — Training
# ---------------------------------------------------------------------------

def _run_training(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_len: int,
) -> Generator[str, None, None]:
    device_label = _get_device_label()
    use_unsloth = _unsloth_available()
    script = "train.finetune_unsloth" if use_unsloth else "train.finetune_cpu"

    yield f"Device: {device_label}\n"
    yield f"Backend: {'Unsloth (GPU fast path)' if use_unsloth else 'HuggingFace Trainer (CPU/MPS)'}\n"
    if "CPU" in device_label and not use_unsloth:
        yield (
            "WARNING: Training on CPU is very slow. "
            "For a 1.1B model expect several minutes per step on a standard machine.\n"
        )
    yield "\n"

    train_file = _TRAINING_DATA_DIR / "train_sharegpt.jsonl"
    val_file = _TRAINING_DATA_DIR / "val_sharegpt.jsonl"
    if not train_file.exists():
        yield f"ERROR: {train_file} not found. Complete the 'Prepare Data' step first.\n"
        return
    if not val_file.exists():
        yield f"ERROR: {val_file} not found. Complete the 'Prepare Data' step first.\n"
        return

    _OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", script,
        "--train-file", str(train_file),
        "--val-file", str(val_file),
        "--output-dir", str(_OUTPUT_MODEL_DIR),
        "--model-name", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--max-seq-length", str(max_seq_len),
    ]
    yield f"Running: {' '.join(cmd)}\n\n"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    if proc.returncode == 0:
        yield "\nTraining complete. Switch to the Chat tab and click 'Load Model'.\n"
    else:
        yield f"\nTraining FAILED (exit code {proc.returncode}).\n"


# ---------------------------------------------------------------------------
# Tab 3 — Chat
# ---------------------------------------------------------------------------

def _load_model_ui() -> Tuple[str, gr.update]:
    global _model, _tokenizer
    if not _model_ready():
        return (
            "No trained model found at output_model/. Complete the Train step first.",
            gr.update(interactive=True),
        )
    try:
        _model, _tokenizer = load_model(_OUTPUT_MODEL_DIR)
        return "Model loaded. Start chatting!", gr.update(interactive=False, value="Model loaded")
    except Exception as exc:
        return f"Error loading model: {exc}", gr.update(interactive=True)


def _chat(message: str, history: List) -> str:
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return "Model not loaded. Click 'Load Model' first."
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    return generate_response(_model, _tokenizer, messages)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    device_label = _get_device_label()

    with gr.Blocks(title="SLM Training Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            f"# SLM Training Demo\n"
            f"Train a small language model on your own data and chat with it — "
            f"all in the browser.\n\n"
            f"**Runtime device:** `{device_label}`"
        )

        with gr.Tabs():

            # ── Tab 1 ──────────────────────────────────────────────────────
            with gr.Tab("1 · Prepare Data"):
                gr.Markdown(
                    "Upload your source files (PDFs and/or a CSV with `question,answer` columns). "
                    "The tool chunks the content and writes ShareGPT-format JSONL files for training."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Upload files (PDF and/or CSV)",
                            file_count="multiple",
                            file_types=[".pdf", ".csv"],
                        )
                    with gr.Column(scale=1):
                        chunk_size = gr.Slider(
                            200, 2000, value=800, step=50, label="Chunk size (chars)"
                        )
                        chunk_overlap = gr.Slider(
                            0, 400, value=150, step=25, label="Chunk overlap"
                        )
                        val_ratio = gr.Slider(
                            0.05, 0.4, value=0.15, step=0.05, label="Validation split ratio"
                        )
                        fmt_choice = gr.Radio(
                            ["sharegpt", "alpaca", "both"],
                            value="sharegpt",
                            label="Output format",
                        )

                prep_btn = gr.Button("Prepare Data", variant="primary")
                prep_log = gr.Textbox(
                    label="Log",
                    lines=15,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True,
                )
                prep_btn.click(
                    fn=_run_data_prep,
                    inputs=[file_upload, chunk_size, chunk_overlap, val_ratio, fmt_choice],
                    outputs=prep_log,
                )

            # ── Tab 2 ──────────────────────────────────────────────────────
            with gr.Tab("2 · Train"):
                gr.Markdown(
                    "Configure parameters and click **Start Training**. "
                    "Logs stream live. Training is done when you see *'Saved to output_model'*."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        model_name = gr.Dropdown(
                            choices=[
                                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                "unsloth/TinyLlama-1.1b-Chat-v1.0",
                                "unsloth/Llama-3.2-1B-Instruct",
                                "meta-llama/Llama-3.2-1B-Instruct",
                            ],
                            value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            label="Base model",
                            allow_custom_value=True,
                        )
                        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                        batch_size = gr.Slider(
                            1, 8, value=2, step=1, label="Batch size per device"
                        )
                        lr = gr.Number(value=2e-4, label="Learning rate", precision=6)
                        max_seq_len = gr.Slider(
                            256, 2048, value=512, step=128, label="Max sequence length"
                        )

                train_btn = gr.Button("Start Training", variant="primary")
                train_log = gr.Textbox(
                    label="Training log",
                    lines=25,
                    max_lines=60,
                    interactive=False,
                    show_copy_button=True,
                )
                train_btn.click(
                    fn=_run_training,
                    inputs=[model_name, epochs, batch_size, lr, max_seq_len],
                    outputs=train_log,
                )

            # ── Tab 3 ──────────────────────────────────────────────────────
            with gr.Tab("3 · Chat"):
                gr.Markdown(
                    "Once training is done, load the model and start asking questions."
                )
                load_btn = gr.Button(
                    "Load Model" if not _model_ready() else "Load Model (ready)",
                    variant="primary",
                )
                load_status = gr.Textbox(
                    label="Status",
                    value=(
                        "Model found — click Load Model to begin."
                        if _model_ready()
                        else "No trained model yet. Complete the Train step first."
                    ),
                    interactive=False,
                    lines=1,
                )
                chat_interface = gr.ChatInterface(  # noqa: F841
                    fn=_chat,
                    examples=[
                        "What is this document about?",
                        "Summarize the main points.",
                        "How do I get started?",
                    ],
                )
                load_btn.click(
                    fn=_load_model_ui,
                    inputs=[],
                    outputs=[load_status, load_btn],
                )

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse
    import os

    p = argparse.ArgumentParser(description="End-to-end SLM training and demo UI")
    p.add_argument(
        "--model-dir", type=Path, default=None,
        help="Trained model directory (default: /app/output_model)",
    )
    p.add_argument(
        "--training-data-dir", type=Path, default=None,
        help="Training data directory (default: /app/training_data)",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Raw uploaded data directory (default: /app/data)",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Bind address (use 0.0.0.0 in Docker)")
    p.add_argument("--share", action="store_true", default=False)
    args = p.parse_args()

    global _DATA_DIR, _TRAINING_DATA_DIR, _OUTPUT_MODEL_DIR
    if args.data_dir:
        _DATA_DIR = args.data_dir
    if args.training_data_dir:
        _TRAINING_DATA_DIR = args.training_data_dir

    # Support old --model-dir flag and MODEL_DIR env var for backwards compat
    model_dir = args.model_dir or Path(os.environ.get("MODEL_DIR", "")) or None
    if model_dir and str(model_dir) not in ("", "."):
        _OUTPUT_MODEL_DIR = model_dir

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
