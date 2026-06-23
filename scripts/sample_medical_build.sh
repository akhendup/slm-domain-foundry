#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sample_medical_build.sh — End-to-end medical sample: data → train → chat
#
# Prepares a realistic reference dataset (CSV + YAML patterns + vocabulary
# expansion) and fine-tunes a chat-capable instruct model on CPU, MPS, or CUDA.
#
# Usage:
#   chmod +x scripts/sample_medical_build.sh
#   ./scripts/sample_medical_build.sh
#
# Options (environment variables):
#   SAMPLE_MODEL=unsloth/Llama-3.2-1B-Instruct   # default (matches config.yaml)
#   SAMPLE_EPOCHS=1
#   SAMPLE_MAX_STEPS=500   # default cap for a practical demo run; set to 0 for a full epoch
#   SKIP_VOCAB=1           # skip medical_vocabulary.yaml (smaller dataset, not recommended)
#   SKIP_TRAIN=1           # data prep + sample preview only
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PYTHON="$VENV/bin/python"
SAMPLE_MODEL="${SAMPLE_MODEL:-unsloth/Llama-3.2-1B-Instruct}"
SAMPLE_EPOCHS="${SAMPLE_EPOCHS:-1}"
# 500 steps ≈ reasonable demo on MPS/CPU; training_data/ still holds the full reference set.
SAMPLE_MAX_STEPS="${SAMPLE_MAX_STEPS:-500}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating virtual environment…"
  python3 -m venv "$VENV"
fi

echo "Installing dependencies…"
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel
if [[ "$(uname)" == "Darwin" ]]; then
  "$VENV/bin/pip" install --quiet --prefer-binary -r "$REPO_ROOT/requirements-mps.txt"
else
  "$VENV/bin/pip" install --quiet --prefer-binary -r "$REPO_ROOT/requirements.txt"
fi

echo ""
"$PYTHON" - <<'PY'
import torch, platform, sys
print(f"Python:  {platform.python_version()}  ({sys.platform})")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device:  CUDA ({torch.cuda.get_device_name(0)})")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("Device:  MPS (Apple Silicon)")
else:
    print("Device:  CPU")
PY
echo ""

mkdir -p training_data output_model

PREP_ARGS=(
  --config config.yaml
  --csv sample_data/medical_qa.csv
  --yaml-dir sample_data/patternexamples
  --output-dir training_data
)
if [[ "${SKIP_VOCAB:-0}" != "1" ]]; then
  PREP_ARGS+=(--vocab-dir data/medical_vocabulary.yaml)
  echo "Including medical_vocabulary.yaml expansion (CSV + patterns + medical vocab combinatorics)."
else
  echo "SKIP_VOCAB=1 — CSV and YAML patterns only (no vocabulary expansion)."
fi

echo ""
echo "=== Step 1: Prepare medical training data ==="
PYTHONPATH="$REPO_ROOT" "$PYTHON" -m data.prepare_training_data "${PREP_ARGS[@]}"

if [[ ! -f training_data/train_sharegpt.jsonl ]]; then
  echo "Error: training_data/train_sharegpt.jsonl was not created." >&2
  exit 1
fi

TRAIN_LINES=$(wc -l < training_data/train_sharegpt.jsonl | tr -d ' ')
echo ""
echo "  → ${TRAIN_LINES} ShareGPT training examples in training_data/train_sharegpt.jsonl"
echo "  → Alpaca + ShareGPT JSONL under training_data/ is the full reference dataset."

echo ""
echo "=== Sample training examples (reference) ==="
PYTHONPATH="$REPO_ROOT" "$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("training_data/train_sharegpt.jsonl")
shown = 0
with path.open() as f:
    for line in f:
        if shown >= 3:
            break
        obj = json.loads(line)
        conv = obj.get("conversations") or []
        user = next((c.get("content", "") for c in conv if c.get("role") == "user"), "")
        assistant = next((c.get("content", "") for c in conv if c.get("role") == "assistant"), "")
        if not user:
            continue
        print(f"\n--- Example {shown + 1} ---")
        print(f"Q: {user[:200]}{'…' if len(user) > 200 else ''}")
        print(f"A: {assistant[:280]}{'…' if len(assistant) > 280 else ''}")
        shown += 1
if shown == 0:
    print("(no examples found)")
PY

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo ""
  echo "SKIP_TRAIN=1 — skipping fine-tune. Inspect training_data/ for the reference dataset."
  exit 0
fi

TRAIN_ARGS=(
  --train-file training_data/train_sharegpt.jsonl
  --val-file training_data/val_sharegpt.jsonl
  --model-name "$SAMPLE_MODEL"
  --output-dir output_model
  --epochs "$SAMPLE_EPOCHS"
  --batch-size 1
  --grad-accum 4
  --save-steps 1000
  --no-eval
  --max-seq-length 512
  --gradient-checkpointing
)
if [[ "$SAMPLE_MAX_STEPS" != "0" ]]; then
  TRAIN_ARGS+=(--max-steps "$SAMPLE_MAX_STEPS")
  echo ""
  echo "=== Step 2: Fine-tune (${SAMPLE_MODEL}, max ${SAMPLE_MAX_STEPS} steps) ==="
  echo "  Set SAMPLE_MAX_STEPS=0 for a full epoch (~${TRAIN_LINES} optimizer steps)."
else
  echo ""
  echo "=== Step 2: Fine-tune (${SAMPLE_MODEL}, ${SAMPLE_EPOCHS} full epoch, ~${TRAIN_LINES} steps) ==="
fi

PYTHONPATH="$REPO_ROOT" "$PYTHON" -m train.finetune_cpu "${TRAIN_ARGS[@]}"

echo ""
echo "=== Step 3: Smoke inference ==="
PYTHONPATH="$REPO_ROOT" "$PYTHON" - <<'PY'
from app.model_loader import generate_response, load_model

model, tok = load_model("output_model")
system = (
    "You are a medical AI assistant specialized in clinical decision support. "
    "Provide accurate, evidence-based information in clear language."
)
questions = [
    "What is hypertension?",
    "What aspirin dose is used for secondary prevention?",
    "What lifestyle changes help manage elevated blood pressure?",
]
for q in questions:
    ans = generate_response(
        model, tok,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        max_new_tokens=128,
        temperature=0.0,
    )
    print(f"Q: {q}")
    print(f"A: {ans.strip()[:400]}")
    print()
PY

echo "Done. Model saved to output_model/"
echo "Launch UI: PYTHONPATH=. $PYTHON run_gradio_ui.py --model-dir output_model"
