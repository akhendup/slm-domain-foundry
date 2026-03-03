# AI Small Language Model Training

This project shows **how to train a small language model** end-to-end: from your own data (PDF or CSV) to a model you can ask questions and get answers from.

## What it does

1. **Data** – Load content from PDFs or CSV (e.g. question/answer columns).
2. **Training data** – Chunk and convert that into instruction/response pairs (Alpaca and ShareGPT formats).
3. **Training** – Either:
   - **(a) Train from scratch** – Build a tiny transformer and train it on your text (see `train/README_FROM_SCRATCH.md` and `scripts/export_for_from_scratch.py`).
   - **(b) Fine-tune** – Fine-tune a small model like **TinyLlama** (or Llama-3.2-1B) on your data with Unsloth.
4. **Demo** – Run a simple Q&A in the terminal: you ask questions, the model answers, so you can see how it behaves after training.

## Quick start

### Option 1: Docker (runs on any platform with Docker)

All dependencies are in `requirements.txt`; the image works on Linux, macOS, and Windows (Docker Desktop).

```bash
# Build the image
docker build -t ai_slm_training .

# Prepare training data (mount a folder with your CSV or PDFs)
docker run --rm -v "$(pwd)/my_data:/data" ai_slm_training \
  python -m data.prepare_training_data --csv /data/qa.csv --output-dir /data/training_data

# Or open a shell and run commands
docker run -it --rm -v "$(pwd)/my_data:/data" ai_slm_training bash
# Inside: python -m data.prepare_training_data --csv /data/qa.csv --output-dir /data/training_data
#         python -m train.finetune_unsloth --train-file /data/training_data/train_sharegpt.jsonl ...
#         python -m demo.chat --model-dir /data/output_model --interactive
```

On **Windows (PowerShell)** use `${PWD}` or full path for the volume, e.g. `-v "C:\path\to\my_data:/data"`.

### Option 2: Local install

```bash
cd ai_slm_training
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare training data

From a directory of PDFs:

```bash
python -m data.prepare_training_data --pdf-dir path/to/pdfs --output-dir training_data
```

From a CSV (with `question` / `answer` columns, or a single `text` column):

```bash
python -m data.prepare_training_data --csv path/to/qa.csv --output-dir training_data
```

You can use both:

```bash
python -m data.prepare_training_data --pdf-dir ./pdfs --csv ./qa.csv --output-dir training_data
```

Outputs in `training_data/`:

- `train_sharegpt.jsonl` / `val_sharegpt.jsonl` – for Unsloth fine-tuning.
- `train_alpaca.jsonl` / `val_alpaca.jsonl` – Alpaca instruction format.

### 3. Train the model

**Option B – Fine-tune TinyLlama (recommended for a first run):**

```bash
python -m train.finetune_unsloth \
  --train-file training_data/train_sharegpt.jsonl \
  --val-file training_data/val_sharegpt.jsonl \
  --output-dir output_model
```

Use `--model-name unsloth/TinyLlama-1.1b-Chat-v1.0` (default) or e.g. `unsloth/Llama-3.2-1B-Instruct` for a slightly larger model.

**Option A – Train from scratch:**  
See `train/README_FROM_SCRATCH.md`. Use `scripts/export_for_from_scratch.py` to export `train_sharegpt.jsonl` to a single text file, then use NanoGPT/minGPT or a minimal Hugging Face script.

### 4. Run the Q&A demo

**Web UI (recommended)** – Open a browser and chat with the model:

```bash
# Local
python -m demo.gradio_ui --model-dir output_model
# Then open http://127.0.0.1:7860 in your browser.

# Docker (mount your trained model, map port 7860)
docker run -p 7860:7860 -v "$(pwd)/output_model:/app/model:ro" ai_slm_training \
  python run_gradio_ui.py --model-dir /app/model --host 0.0.0.0
# Then open http://localhost:7860
```

**CLI** – Terminal-only:

```bash
python -m demo.chat --model-dir output_model              # sample questions
python -m demo.chat --model-dir output_model --interactive  # type questions in terminal
```

The demo shows how the model responds after training: your data → training → answers in the UI or CLI.

## Project layout

```
ai_slm_training/
├── data/
│   ├── pdf_extractor.py      # PDF → text (with optional OCR)
│   ├── csv_loader.py          # CSV → text or Q&A pairs
│   ├── chunking.py           # Text chunking
│   └── prepare_training_data.py  # PDF/CSV → Alpaca/ShareGPT JSONL
├── train/
│   ├── finetune_unsloth.py   # Fine-tune TinyLlama (or other) with Unsloth
│   └── README_FROM_SCRATCH.md
├── demo/
│   ├── chat.py               # CLI Q&A with trained model
│   └── gradio_ui.py          # Web UI for Q&A (Gradio)
├── scripts/
│   └── export_for_from_scratch.py  # ShareGPT JSONL → single .txt
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- **Single file:** `requirements.txt` – PDF (pdfplumber, PyPDF2), CSV (stdlib), PyTorch, Unsloth, transformers, datasets, trl, etc. Used for both local install and Docker. The default Docker image uses CPU-only PyTorch so it runs on any host; for GPU training use a CUDA base image and install PyTorch with CUDA.

## Summary

- **Data:** PDF or CSV → `data/prepare_training_data.py` → `training_data/*.jsonl`.
- **Train:** Either (a) from scratch (see `train/README_FROM_SCRATCH.md`) or (b) `train/finetune_unsloth.py` (TinyLlama).
- **Demo:** `demo/gradio_ui.py` (web UI) or `demo/chat.py` (CLI) – ask questions and see answers so anyone can see how the model is trained and used.
