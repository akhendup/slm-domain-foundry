# run_local.ps1 — Launch the Gradio UI natively on Windows (CUDA or CPU)
#
# Why native instead of Docker?
#   Running natively gives PyTorch direct access to the NVIDIA GPU via CUDA,
#   which is ~10-50x faster than CPU for fine-tuning.
#   (Docker + GPU requires nvidia-container-toolkit and WSL2 — more complex to set up.)
#
# Prerequisites:
#   - Python 3.10+ installed and on PATH  (https://www.python.org/downloads/)
#   - For GPU: NVIDIA driver + CUDA toolkit installed
#
# First run (sets up venv + installs deps):
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # (one-time, if blocked)
#   .\run_local.ps1
#
# Subsequent runs:
#   .\run_local.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$Venv   = "$ScriptDir\.venv"
$Python = "$Venv\Scripts\python.exe"
$Pip    = "$Venv\Scripts\pip.exe"

# ── 1. Create venv if it doesn't exist ──────────────────────────────────────
if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    python -m venv $Venv
}

# ── 2. Install / update dependencies ────────────────────────────────────────
Write-Host "Checking dependencies..."
& $Pip install --quiet --upgrade pip setuptools wheel
& $Pip install --quiet --prefer-binary -r "$ScriptDir\requirements.txt"

# ── 3. Report device ─────────────────────────────────────────────────────────
Write-Host ""
& $Python - << 'PY'
import torch, platform, sys
print(f"Python:  {platform.python_version()}  ({sys.platform})")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print(f"Device:  CUDA ({torch.cuda.get_device_name(0)}) ✓  — training will use the GPU")
else:
    print("Device:  CPU  (no GPU found — training will be slow)")
    print("         Tip: install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/")
PY

# ── 4. Local data directories ────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "$ScriptDir\data"          | Out-Null
New-Item -ItemType Directory -Force -Path "$ScriptDir\training_data" | Out-Null
New-Item -ItemType Directory -Force -Path "$ScriptDir\output_model"  | Out-Null
New-Item -ItemType Directory -Force -Path "$ScriptDir\saved_models"  | Out-Null

# ── 5. Launch ────────────────────────────────────────────────────────────────
Write-Host "Starting Gradio UI at http://localhost:7860"
Write-Host "(Ctrl+C to stop)"
Write-Host ""

$env:PYTHONPATH = $ScriptDir
& $Python run_gradio_ui.py `
    --data-dir          "$ScriptDir\data" `
    --training-data-dir "$ScriptDir\training_data" `
    --model-dir         "$ScriptDir\output_model" `
    --host 0.0.0.0 `
    --port 7860
