#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_local.sh — Launch the Gradio UI natively on Mac (MPS / Apple Silicon)
#
# Why native instead of Docker?
#   Docker on Mac runs in a Linux VM — Metal/MPS never passes through.
#   Running here directly gives PyTorch access to the Apple Silicon GPU,
#   which is ~5-10x faster than Docker CPU for fine-tuning.
#
# First run (sets up venv + installs deps):
#   chmod +x run_local.sh && ./run_local.sh
#
# Subsequent runs:
#   ./run_local.sh
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"

# ── 1. Create venv if it doesn't exist ──────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
    echo "Creating virtual environment…"
    python3 -m venv "$VENV"
fi

# ── 2. Install / update dependencies ────────────────────────────────────────
echo "Checking dependencies…"
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel

# bitsandbytes has no macOS build; skip it there (not needed for standard LoRA).
# On Linux it installs fine and works with CUDA if available.
if [[ "$(uname)" == "Darwin" ]]; then
    "$VENV/bin/pip" install --quiet --prefer-binary -r "$SCRIPT_DIR/requirements-mps.txt"
else
    "$VENV/bin/pip" install --quiet --prefer-binary -r "$SCRIPT_DIR/requirements.txt"
fi

# ── 3. Report device ─────────────────────────────────────────────────────────
echo ""
"$PYTHON" - <<'PY'
import torch, platform, sys
print(f"Python:  {platform.python_version()}  ({sys.platform})")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device:  CUDA ({torch.cuda.get_device_name(0)}) ✓  — training will use the GPU")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("Device:  MPS (Apple Silicon GPU) ✓  — training will use the GPU")
else:
    print("Device:  CPU  (no GPU found — training will be slow)")
PY
echo ""

# ── 4. Local data directories (relative to project root) ────────────────────
mkdir -p "$SCRIPT_DIR/data"
mkdir -p "$SCRIPT_DIR/training_data"
mkdir -p "$SCRIPT_DIR/output_model"
mkdir -p "$SCRIPT_DIR/saved_models"

# ── 5. Launch ────────────────────────────────────────────────────────────────
# GRADIO_HOST controls the bind address (not the access URL).
#   0.0.0.0  = all interfaces — local + remote (default)
#   127.0.0.1 = localhost only
# Access the UI at http://<this-machine-hostname>:${GRADIO_PORT} from any machine.
GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

echo "Starting Gradio UI — bound to ${GRADIO_HOST}:${GRADIO_PORT}"
echo "Access at: http://$(hostname):${GRADIO_PORT}  (or http://localhost:${GRADIO_PORT} locally)"
echo "(Ctrl+C to stop)"
echo ""

PYTHONPATH="$SCRIPT_DIR" "$PYTHON" run_gradio_ui.py \
    --data-dir        "$SCRIPT_DIR/data" \
    --training-data-dir "$SCRIPT_DIR/training_data" \
    --model-dir       "$SCRIPT_DIR/output_model" \
    --host "$GRADIO_HOST" \
    --port "$GRADIO_PORT"
