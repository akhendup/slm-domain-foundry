#!/bin/bash
#SBATCH --job-name=ai_slm_train
#SBATCH --output=slurm_output_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# 1. Force Python to recognize your root directory for modular imports
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH}"
# Avoid repo unsloth_compiled_cache shadowing TRL (pickle errors when saving checkpoints)
export UNSLOTH_COMPILE_LOCATION="${UNSLOTH_COMPILE_LOCATION:-${HOME}/.cache/unsloth_compiled}"
cd "${SLURM_SUBMIT_DIR}"

# 2. Fine-tune with Unsloth (see README.md § "Train the model")
# Slurm does not inherit your interactive conda env — pick a Python that has unsloth.
# Override: SLURM_PYTHON=/path/to/python sbatch submit_unsloth.slurm
_select_python() {
  local candidate
  for candidate in \
    "${SLURM_PYTHON:-}" \
    "${HOME}/.unsloth/studio/unsloth_studio/bin/python" \
    "${SLURM_SUBMIT_DIR}/.venv/bin/python" \
    "${SLURM_SUBMIT_DIR}/venv/bin/python" \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
  do
    [ -n "${candidate}" ] || continue
    [ -x "${candidate}" ] || continue
    if "${candidate}" -c "import unsloth, torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON="$(_select_python)" || {
  echo "ERROR: No Python with unsloth + CUDA found." >&2
  echo "Activate your env and verify: python -c \"import unsloth, torch; print(torch.cuda.is_available())\"" >&2
  echo "Then either install unsloth in .venv or set SLURM_PYTHON to that interpreter." >&2
  exit 1
}
echo "Using Python: ${PYTHON}" >&2

# 3. Free GPU VRAM held by Ollama (only models currently loaded — no hardcoded name)
if command -v curl >/dev/null 2>&1; then
  "${PYTHON}" - <<'PY' 2>/dev/null || true
import json, subprocess, urllib.request

try:
    with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as resp:
        data = json.load(resp)
except Exception:
    raise SystemExit(0)

for m in data.get("models") or []:
    name = m.get("name")
    if not name:
        continue
    payload = json.dumps({"model": name, "keep_alive": 0}).encode()
    subprocess.run(
        ["curl", "-sf", "-X", "POST", "http://localhost:11434/api/generate", "-d", payload.decode()],
        check=False,
    )
    print(f"Ollama: unloaded {name} from VRAM", flush=True)
PY
fi

export PYTHONUNBUFFERED=1
# RTX 3060 / 12GB: disable Unsloth fast paths that can segfault under memory pressure
export UNSLOTH_DISABLE_AUTO_PADDING_FREE=1
export UNSLOTH_DISABLE_DOUBLE_BUFFER=1
export UNSLOTH_COMPILE_DISABLE=1
# expandable_segments can itself trigger Unsloth buffer issues on some setups — leave unset
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

# RTX 3060 (12GB): --stable caps seq/batch; --no-eval skips validation during training
"${PYTHON}" -m train.finetune_unsloth \
  --train-file training_data/train_sharegpt.jsonl \
  --val-file training_data/val_sharegpt.jsonl \
  --output-dir output_model \
  --model-name "${SLURM_MODEL_NAME:-unsloth/tinyllama-chat-bnb-4bit}" \
  --no-eval \
  --stable \
  --grad-accum 8 \
  --dataloader-workers 0

# Gradio UI (interactive; not typical under Slurm):
# "${PYTHON}" run_gradio_ui.py