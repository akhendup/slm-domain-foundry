#!/usr/bin/env bash
# Sync this repo to amdworkstation and run CUDA-marked tests there.
#
# Prerequisites:
#   - SSH host `amdworkstation` configured in ~/.ssh/config
#   - NVIDIA driver + CUDA visible to PyTorch on the remote host
#
# Usage:
#   ./scripts/run_tests_amdworkstation.sh              # GPU tests only
#   ./scripts/run_tests_amdworkstation.sh --full       # full suite (excludes MPS-only file)
#   ./scripts/run_tests_amdworkstation.sh --sync-only  # rsync, no pytest
#
set -euo pipefail

REMOTE="${AMDWORKSTATION_HOST:-amdworkstation}"
REMOTE_DIR="${AMDWORKSTATION_DIR:-~/slm-domain-foundry}"
MODE="${1:-gpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Syncing ${REPO_ROOT} → ${REMOTE}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  --exclude 'training_data/' \
  --exclude 'output_model/' \
  --exclude 'saved_models/' \
  --exclude '.coverage' \
  --exclude '.coverage.*' \
  --exclude 'coverage.json' \
  --exclude 'unsloth_compiled_cache/' \
  "$REPO_ROOT/" "${REMOTE}:${REMOTE_DIR}/"

if [[ "$MODE" == "--sync-only" ]]; then
  echo "Sync complete."
  exit 0
fi

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
cd "$REMOTE_DIR"
PYTHON="${REMOTE_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv .venv
  PYTHON="${REMOTE_DIR}/.venv/bin/python"
fi
"$PYTHON" -m pip install --quiet --upgrade pip
"$PYTHON" -m pip install --quiet -r requirements.txt pytest pytest-cov
"$PYTHON" - <<'PY'
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
EOF
)

ssh "$REMOTE" "REMOTE_DIR=${REMOTE_DIR} bash -s" <<< "$REMOTE_CMD"

if [[ "$MODE" == "--full" ]]; then
  echo "Running full test suite on ${REMOTE} (excluding MPS-only tests)..."
  ssh "$REMOTE" "cd ${REMOTE_DIR} && .venv/bin/pytest tests/ --tb=short \
    --ignore=tests/real/test_apple_silicon_mps.py \
    --cov=app --cov=data --cov=train --cov-report=term-missing --cov-fail-under=75"
elif [[ "$MODE" == "gpu" || "$MODE" == "--gpu" ]]; then
  echo "Running CUDA GPU tests on ${REMOTE}..."
  ssh "$REMOTE" "cd ${REMOTE_DIR} && .venv/bin/pytest tests/real/test_cuda_gpu.py -v --tb=short -m gpu"
else
  echo "Unknown mode: $MODE (use --gpu, --full, or --sync-only)" >&2
  exit 1
fi

echo "Done."
