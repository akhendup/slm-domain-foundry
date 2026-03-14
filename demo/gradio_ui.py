"""
End-to-end Gradio UI: Upload → Extract Training Data → Train → Manage Models → Chat.
Runs in Docker or natively; training streams live logs.

Runtime modes (auto-detected):
  Docker   — paths default to /app/{data,training_data,output_model,...}
  Native   — paths default to <project_root>/{data,training_data,output_model,...}

Device/backend (auto-detected):
  CUDA     — Unsloth fast path for training; transformers for inference
  MPS      — HuggingFace Trainer + transformers (Apple Silicon)
  CPU      — HuggingFace Trainer + ONNX Runtime (if optimum installed) for inference

Usage:
  python -m demo.gradio_ui                          # auto-detect mode
  python -m demo.gradio_ui --host 0.0.0.0          # bind to all interfaces
  python -m demo.gradio_ui --model-dir output_model  # override model path
  DOCKER_CONTAINER=1 python -m demo.gradio_ui       # force Docker paths
"""

import html as _html
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import gradio as gr
import pandas as pd
import torch

from demo.model_loader import generate_response, load_model
from data.knowledge_retriever import KnowledgeRetriever
from data.conversation_memory import (
    log_interaction,
    load_interactions,
    set_approval,
    memory_stats,
    export_approved_to_jsonl,
    mine_frequent_questions,
)
from data.knowledge_capture import (
    FIELD_DEFS,
    form_to_pattern,
    save_to_library,
    load_library_entries,
    library_stats,
    delete_from_library,
    load_pattern_for_edit,
    preview_qa,
)

# ---------------------------------------------------------------------------
# Runtime mode detection: Docker vs native
# ---------------------------------------------------------------------------
def _is_docker() -> bool:
    """Return True when running inside a Docker container."""
    # Explicit env override always wins
    if os.environ.get("DOCKER_CONTAINER", "").lower() in ("1", "true", "yes"):
        return True
    # Standard Docker marker file
    if Path("/.dockerenv").exists():
        return True
    # cgroup-based detection (Linux containers)
    try:
        with open("/proc/self/cgroup") as f:
            if "docker" in f.read() or "kubepods" in f.read():
                return True
    except OSError:
        pass
    return False

_IN_DOCKER = _is_docker()

# ---------------------------------------------------------------------------
# Default paths — Docker uses /app/, native uses project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_APP_ROOT = Path("/app") if _IN_DOCKER else _PROJECT_ROOT
_DATA_DIR = _APP_ROOT / "data"
_TRAINING_DATA_DIR = _APP_ROOT / "training_data"
_OUTPUT_MODEL_DIR = _APP_ROOT / "output_model"
_SAVED_MODELS_DIR = _APP_ROOT / "saved_models"
_LIBRARY_DIR = _APP_ROOT / "knowledge_library"
_MEMORY_DIR = _APP_ROOT / "conversation_memory"

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None

# Knowledge Library retriever — loaded lazily on first chat query
_knowledge_retriever: Optional["KnowledgeRetriever"] = None

# Chat session ID — changes each time the model is (re)loaded
_chat_session_id: str = ""


def _get_retriever() -> "KnowledgeRetriever":
    global _knowledge_retriever
    if _knowledge_retriever is None:
        _knowledge_retriever = KnowledgeRetriever(_LIBRARY_DIR)
    return _knowledge_retriever


def _active_model_name() -> str:
    """Return the name of the currently loaded model, best-effort."""
    try:
        meta_path = _OUTPUT_MODEL_DIR / "_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return meta.get("name") or meta.get("active_source") or "output_model"
    except Exception:
        pass
    return "output_model"

# Pipeline completion tracking — one key per step
_pipeline_status: dict = {
    "upload":  "pending",
    "extract": "pending",
    "approve": "pending",
    "train":   "pending",
}

# Training state — persists across browser reconnections; updated live by _run_training
_training_state: dict = {
    "active": False,
    "done": False,
    "failed": False,
    "log": "",
    "phase": "loading",
    "current_msg": "",
    "warnings": [],
    "error": None,
    "current_epoch": 0,
    "total_epochs": 1,
    "global_step": 0,
    "total_steps": 0,
    "last_pct": 0.0,
    "initial_loss": None,
    "last_loss": None,
    "last_eval_loss": None,
    "n_examples": 0,
    "loss_history": [],
    "start_time": None,
    "elapsed": 0.0,
}


# ---------------------------------------------------------------------------
# Dataset snapshot helpers
# ---------------------------------------------------------------------------

def _save_dataset_snapshot() -> Tuple[Optional[Path], str]:
    """Zip all JSONL files in training_data/ (including subdirs) into a snapshot archive."""
    files = sorted(
        list(_TRAINING_DATA_DIR.glob("*.jsonl")) +
        list(_TRAINING_DATA_DIR.glob("*/*.jsonl"))
    )
    if not files:
        return None, "No training data found — run Extract first."
    snap_path = _TRAINING_DATA_DIR / "dataset_snapshot.zip"
    with zipfile.ZipFile(snap_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(_TRAINING_DATA_DIR))
    size_kb = snap_path.stat().st_size / 1024
    return snap_path, (
        f"Snapshot ready: {len(files)} JSONL files, {size_kb:.1f} KB — "
        "download it using the file widget below."
    )


def _load_dataset_snapshot(upload_path: str) -> str:
    """Extract an uploaded zip (or copy a single JSONL) into training_data/."""
    src = Path(upload_path)
    _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            names = zf.namelist()
            zf.extractall(_TRAINING_DATA_DIR)
        return (
            f"Loaded snapshot: extracted {len(names)} file(s) to {_TRAINING_DATA_DIR}. "
            "You can now start training without running Extract."
        )
    elif src.suffix.lower() == ".jsonl":
        dest = _TRAINING_DATA_DIR / src.name
        shutil.copy2(src, dest)
        return (
            f"Loaded {src.name} → {_TRAINING_DATA_DIR}. "
            "You can now start training without running Extract."
        )
    return f"Unsupported file type '{src.suffix}'. Upload a .zip snapshot or a .jsonl file."


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the path of the most recently modified checkpoint-N dir, or None."""
    ckpts = sorted(
        (d for d in output_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
    )
    return ckpts[-1] if ckpts else None


def _get_device_label() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA — {name}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "Apple Silicon (CPU)"
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
# Training data file helpers
# ---------------------------------------------------------------------------

def _find_train_jsonl() -> Optional[Path]:
    """Return the first available train_sharegpt.jsonl (root then subdirs)."""
    root = _TRAINING_DATA_DIR / "train_sharegpt.jsonl"
    if root.exists():
        return root
    files = sorted(_TRAINING_DATA_DIR.glob("*/train_sharegpt.jsonl"))
    return files[0] if files else None


def _merge_jsonl(sources: List[Path], dest: Path) -> None:
    """Concatenate multiple JSONL source files into dest."""
    with open(dest, "w", encoding="utf-8") as out:
        for src in sources:
            if src.exists():
                with open(src, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)


def _find_training_files() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (train_path, val_path) for training.
    If root-level files don't exist but per-manual subdirectory files do,
    merge them into root-level files first.
    """
    train = _TRAINING_DATA_DIR / "train_sharegpt.jsonl"
    val = _TRAINING_DATA_DIR / "val_sharegpt.jsonl"
    if train.exists() and val.exists():
        return train, val

    trains = sorted(_TRAINING_DATA_DIR.glob("*/train_sharegpt.jsonl"))
    vals = sorted(_TRAINING_DATA_DIR.glob("*/val_sharegpt.jsonl"))
    if not trains:
        return None, None

    _merge_jsonl(trains, train)
    if vals:
        _merge_jsonl(vals, val)
    elif train.exists():
        # Carve a val set from the merged train file
        import random as _random
        train_lines = [l for l in train.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_val = max(1, int(len(train_lines) * 0.15))
        _random.shuffle(train_lines)
        val.write_text("\n".join(train_lines[:n_val]) + "\n", encoding="utf-8")
        train.write_text("\n".join(train_lines[n_val:]) + "\n", encoding="utf-8")

    return (train if train.exists() else None, val if val.exists() else None)


# ---------------------------------------------------------------------------
# Progress parsing helpers
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(r"--- Epoch (\d+)/(\d+) ---")
_STEP_RE = re.compile(r"step (\d+)/(\d+) (\d+)%")
_LOSS_RE = re.compile(r"\bloss=(\d+\.\d+)")
_EVAL_LOSS_RE = re.compile(r"eval_loss=(\d+\.\d+)")
_EXAMPLES_RE = re.compile(r"Starting training: (\d+) examples")
_TQDM_WEIGHTS_RE = re.compile(r"Loading weights:\s+(\d+)%")
_TRAINABLE_RE = re.compile(
    r"trainable params:\s*([\d,]+).*?all params:\s*([\d,]+).*?(\d+\.\d+)%"
)


def _activity_text(log: str, n: int = 15) -> str:
    """Return the last *n* meaningful training-progress lines from raw log."""
    activity = []
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (
            _EPOCH_RE.search(line)
            or _STEP_RE.search(line)
            or _LOSS_RE.search(line)
            or _EVAL_LOSS_RE.search(line)
            or any(
                k in line
                for k in (
                    "Model loaded",
                    "Adding LoRA",
                    "trainable params",
                    "Loading dataset",
                    "Starting training",
                    "complete.",
                    "Merging LoRA",
                    "Saved to",
                )
            )
        ):
            activity.append(line)
    return "\n".join(activity[-n:])


def _data_activity_text(log: str) -> str:
    """Return meaningful data-prep / extraction progress lines."""
    keep = []
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (
            line.startswith("[")
            or line.startswith("Found ")
            or line.startswith("Loading CSV")
            or line.startswith("Total:")
            or line.startswith("Alpaca:")
            or line.startswith("ShareGPT:")
            or "complete" in line.lower()
            or "Q&A pairs" in line
            or "Extraction" in line
            or "sections=" in line
        ):
            keep.append(line)
    return "\n".join(keep[-15:])


def _rebuild_training_ui():
    """Reconstruct the training tab UI from _training_state on page load / reconnect."""
    s = _training_state
    if not s["active"] and not s["done"] and not s["failed"]:
        return tuple(gr.update() for _ in range(7))

    elapsed = (
        s.get("elapsed", 0.0)
        if (s["done"] or s["failed"])
        else time.time() - (s["start_time"] or time.time())
    )

    card = _make_train_status_html(
        s["phase"], s["current_msg"], s["warnings"], s["error"],
        s["done"], s["failed"],
    )

    if s["global_step"] > 0 or s["phase"] in ("training", "saving"):
        progress_html = _make_training_progress_html(
            max(s["current_epoch"], 1),
            s["total_epochs"],
            s["global_step"],
            s["total_steps"],
            s["last_pct"],
            elapsed,
            s["last_loss"],
            done=s["done"],
            failed=s["failed"],
        )
    else:
        progress_html = ""

    pipe_html = _make_pipeline_html()
    quality_html = (
        _make_quality_html(
            s["last_loss"], s["last_eval_loss"], s["initial_loss"], s["n_examples"]
        )
        if s["done"] else ""
    )
    chart_update = (
        gr.update(value=pd.DataFrame(s["loss_history"]), visible=True)
        if s["loss_history"] else gr.update(visible=False)
    )
    return (
        card,
        progress_html,
        pipe_html,
        quality_html,
        s["log"],
        chart_update,
        _activity_text(s["log"]),
    )


def _format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _make_elapsed_html(elapsed: float, done: bool = False, failed: bool = False) -> str:
    color = "#28a745" if done else ("#dc3545" if failed else "#4a90d9")
    label = "Complete" if done else ("Failed" if failed else "Running&hellip;")
    return (
        f'<div style="font-family:monospace;font-size:13px;padding:8px 16px;'
        f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;'
        f'display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
        f'<span style="color:{color};font-weight:bold;">{label}</span>'
        f'<span style="color:#555;">Elapsed&nbsp;{_format_time(elapsed)}</span></div>'
    )


def _make_training_progress_html(
    current_epoch: int,
    total_epochs: int,
    global_step: int,
    total_steps: int,
    pct: float,
    elapsed: float,
    loss: Optional[float] = None,
    done: bool = False,
    failed: bool = False,
) -> str:
    color = "#28a745" if done else ("#dc3545" if failed else "#4a90d9")
    label = "Complete" if done else ("Failed" if failed else "Training&hellip;")
    eta_str = ""
    if pct > 1 and not done and not failed:
        eta_secs = elapsed / (pct / 100) - elapsed
        eta_str = f"ETA&nbsp;{_format_time(eta_secs)}"
    details = "&nbsp;&middot;&nbsp;".join(
        filter(None, [
            f"Step&nbsp;{global_step}/{total_steps}" if total_steps else "",
            f"loss&nbsp;{loss:.4f}" if loss is not None else "",
            eta_str,
        ])
    )
    return (
        f'<div style="font-family:monospace;font-size:13px;padding:12px 16px;'
        f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
        f'<span style="color:{color};font-weight:bold;">{label}&nbsp;&mdash;&nbsp;'
        f'Epoch&nbsp;{current_epoch}/{total_epochs}</span>'
        f'<span style="color:#555;">Elapsed&nbsp;{_format_time(elapsed)}</span></div>'
        f'<div style="background:#dee2e6;border-radius:4px;height:18px;overflow:hidden;">'
        f'<div style="background:{color};width:{pct:.1f}%;height:100%;'
        f'border-radius:4px;transition:width 0.4s ease;"></div></div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;color:#666;">'
        f'<span>{details}</span><span>{pct:.1f}%&nbsp;complete</span></div></div>'
    )


_TRAIN_PHASES = [
    ("loading", "Load model"),
    ("lora",    "Setup LoRA"),
    ("dataset", "Load data"),
    ("training","Train"),
    ("saving",  "Save"),
]


def _make_train_status_html(
    phase: str,
    current_msg: str,
    warnings: List[str],
    error: Optional[str] = None,
    done: bool = False,
    failed: bool = False,
) -> str:
    phase_keys = [p for p, _ in _TRAIN_PHASES]
    phase_idx = phase_keys.index(phase) if phase in phase_keys else 0

    crumb_parts = []
    for i, (_, plabel) in enumerate(_TRAIN_PHASES):
        if done:
            dot_bg, dot_icon, txt_color, txt_weight = "#28a745", "✓", "#555", "normal"
        elif failed and i == phase_idx:
            dot_bg, dot_icon, txt_color, txt_weight = "#dc3545", "✗", "#333", "bold"
        elif i < phase_idx:
            dot_bg, dot_icon, txt_color, txt_weight = "#28a745", "✓", "#555", "normal"
        elif i == phase_idx:
            dot_bg, dot_icon, txt_color, txt_weight = "#4a90d9", "▶", "#222", "bold"
        else:
            dot_bg, dot_icon, txt_color, txt_weight = "#dee2e6", "–", "#adb5bd", "normal"
        crumb_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'<span style="width:18px;height:18px;border-radius:50%;background:{dot_bg};'
            f'color:white;display:inline-flex;align-items:center;justify-content:center;'
            f'font-size:9px;font-weight:bold;">{dot_icon}</span>'
            f'<span style="color:{txt_color};font-weight:{txt_weight};font-size:12px;">'
            f'{plabel}</span></span>'
        )
    sep = '&nbsp;<span style="color:#dee2e6;font-size:12px;">›</span>&nbsp;'
    breadcrumb = (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:4px;'
        f'padding:10px 14px;background:#f8f9fa;border-bottom:1px solid #dee2e6;">'
        + sep.join(crumb_parts) + '</div>'
    )

    if failed:
        bg, border, fg, icon = "#f8d7da", "#dc3545", "#721c24", "✗"
        body = _html.escape(error or current_msg)
    elif done:
        bg, border, fg, icon = "#d4edda", "#28a745", "#155724", "✓"
        body = _html.escape(current_msg)
    else:
        bg, border, fg, icon = "#e8f4fd", "#4a90d9", "#0c5460", "…"
        body = _html.escape(current_msg)
    msg_html = (
        f'<div style="display:flex;align-items:center;gap:10px;padding:12px 14px;">'
        f'<span style="width:22px;height:22px;border-radius:50%;background:{border};'
        f'color:white;display:inline-flex;align-items:center;justify-content:center;'
        f'font-size:11px;font-weight:bold;flex-shrink:0;">{icon}</span>'
        f'<span style="color:{fg};font-size:13px;">{body}</span></div>'
    )

    warn_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;'
        f'padding:8px 14px;border-top:1px solid #dee2e6;">'
        f'<span style="color:#856404;font-size:13px;flex-shrink:0;">⚠</span>'
        f'<span style="color:#856404;font-size:12px;">{_html.escape(w)}</span></div>'
        for w in warnings
    )

    return (
        f'<div style="font-family:monospace;border:1px solid #dee2e6;border-radius:8px;'
        f'overflow:hidden;margin-bottom:8px;">'
        + breadcrumb + msg_html + warn_html + '</div>'
    )


def _make_pipeline_html() -> str:
    """Render the 5-step pipeline breadcrumb from global _pipeline_status."""
    _CFG = {
        "pending":  ("○", "#adb5bd", "#6c757d"),
        "running":  ("↻", "#4a90d9", "#4a90d9"),
        "complete": ("✓", "#28a745", "#28a745"),
        "warning":  ("⚠", "#ffc107", "#856404"),
        "failed":   ("✗", "#dc3545", "#dc3545"),
    }
    chat_status = "complete" if _pipeline_status.get("train") == "complete" else "pending"
    steps = [
        ("1", "Upload",  _pipeline_status.get("upload",  "pending")),
        ("2", "Extract", _pipeline_status.get("extract", "pending")),
        ("3", "Approve", _pipeline_status.get("approve", "pending")),
        ("4", "Train",   _pipeline_status.get("train",   "pending")),
        ("5", "Chat",    chat_status),
    ]
    parts = []
    for num, name, status in steps:
        icon, bg, tc = _CFG.get(status, _CFG["pending"])
        parts.append(
            f'<div style="display:inline-flex;align-items:center;gap:5px;">'
            f'<span style="width:20px;height:20px;border-radius:50%;background:{bg};color:white;'
            f'display:inline-flex;align-items:center;justify-content:center;font-size:10px;'
            f'font-weight:bold;flex-shrink:0;">{icon}</span>'
            f'<span style="color:#333;font-size:12px;"><b>{num}</b>&nbsp;{name}</span>'
            f'&nbsp;<span style="color:{tc};font-weight:bold;font-size:11px;">[{status.capitalize()}]</span>'
            f'</div>'
        )
    sep = '&nbsp;<span style="color:#adb5bd;font-size:14px;">&#8594;</span>&nbsp;'
    return (
        f'<div style="display:flex;align-items:center;gap:2px;padding:8px 16px;'
        f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;'
        f'font-family:monospace;flex-wrap:wrap;margin-bottom:6px;">'
        + sep.join(parts) + '</div>'
    )


def _make_quality_html(
    final_loss: Optional[float],
    final_eval_loss: Optional[float],
    initial_loss: Optional[float],
    n_examples: int,
) -> str:
    if final_loss is None:
        return ""
    primary = final_eval_loss if final_eval_loss is not None else final_loss
    reduction_pct: Optional[float] = None
    if initial_loss and initial_loss > 0:
        reduction_pct = (initial_loss - final_loss) / initial_loss * 100

    if primary < 1.2 and (reduction_pct is None or reduction_pct > 40):
        rating, border, bg_light, tc = "Excellent", "#28a745", "#d4edda", "#155724"
        note = "Strong convergence and generalization. Responses should be highly relevant and coherent."
    elif primary < 1.8 and (reduction_pct is None or reduction_pct > 25):
        rating, border, bg_light, tc = "Good", "#34a853", "#d4edda", "#1b6b3a"
        note = "Solid learning with good generalization. Most responses should be relevant and useful."
    elif primary < 2.2 and (reduction_pct is None or reduction_pct > 10):
        rating, border, bg_light, tc = "Okay", "#ffc107", "#fff3cd", "#7d5a00"
        note = "Moderate learning. Responses may be inconsistent — consider more data or more epochs."
    elif primary < 2.8:
        rating, border, bg_light, tc = "Fair", "#fd7e14", "#ffe5d0", "#7d3200"
        note = "Limited learning. Try more training data, more epochs, or a lower learning rate."
    else:
        rating, border, bg_light, tc = "Poor", "#dc3545", "#f8d7da", "#721c24"
        note = "Little improvement detected. Check data quality, increase dataset size, or adjust hyperparameters."

    rows = []
    if initial_loss is not None:
        rows.append(("Initial train loss", f"{initial_loss:.4f}", "Baseline before fine-tuning"))
    if final_loss is not None:
        rows.append(("Final train loss", f"{final_loss:.4f}", "Lower = better fit to training data"))
    if final_eval_loss is not None:
        gap = "⚠ Possible overfitting" if final_eval_loss > (final_loss or 0) * 1.3 else "Good generalization"
        rows.append(("Final eval loss", f"{final_eval_loss:.4f}", f"Primary quality metric · {gap}"))
    if reduction_pct is not None:
        icon = "↓" if reduction_pct > 0 else "↑"
        q = "Strong" if reduction_pct > 30 else "Moderate" if reduction_pct > 10 else "Weak"
        rows.append(("Loss reduction", f"{reduction_pct:.1f}%", f"{icon} {q} improvement over baseline"))
    if n_examples > 0:
        size = ("Very small — quality limited" if n_examples < 50
                else "Small — more data recommended" if n_examples < 200
                else "Moderate" if n_examples < 1000 else "Good dataset size")
        rows.append(("Training examples", str(n_examples), size))

    rows_html = "".join(
        f'<tr style="border-bottom:1px solid #dee2e6;">'
        f'<td style="padding:5px 8px;color:#444;">{m}</td>'
        f'<td style="padding:5px 8px;font-weight:bold;">{v}</td>'
        f'<td style="padding:5px 8px;color:#666;font-size:12px;">{i}</td></tr>'
        for m, v, i in rows
    )
    warns = []
    if 0 < n_examples < 50:
        warns.append(f"⚠ Very small dataset ({n_examples} examples) — model will memorise rather than generalise")
    elif 0 < n_examples < 200:
        warns.append(f"⚠ Small dataset ({n_examples} examples) — adding more examples will improve quality")
    if final_eval_loss and final_loss and final_eval_loss > final_loss * 1.3:
        warns.append("⚠ Eval loss significantly higher than train loss — overfitting likely")
    warns_html = "".join(
        f'<div style="padding:4px 10px;margin-top:4px;font-size:12px;color:#856404;'
        f'background:#fff3cd;border:1px solid #ffc107;border-radius:4px;">{w}</div>'
        for w in warns
    )
    return (
        f'<div style="border:2px solid {border};border-radius:8px;padding:14px 16px;'
        f'font-family:monospace;font-size:13px;margin-top:8px;">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
        f'<span style="background:{border};color:white;padding:4px 16px;border-radius:6px;'
        f'font-size:15px;font-weight:bold;">{rating}</span>'
        f'<span style="color:#555;font-size:12px;">Model quality estimate based on training metrics</span></div>'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:#f8f9fa;font-size:12px;">'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Metric</th>'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Value</th>'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Interpretation</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
        f'{warns_html}'
        f'<div style="margin-top:8px;padding:8px 10px;background:{bg_light};border-radius:4px;'
        f'color:{tc};font-size:12px;"><b>What this means:</b>&nbsp;{note}</div></div>'
    )


def _make_data_sample_html(train_jsonl: Path) -> str:
    """Render a dataset stats bar + 3 Q&A sample cards from the middle of the file."""
    if not train_jsonl.exists():
        return ""
    try:
        lines = [l for l in train_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return ""
    n_total = len(lines)
    if n_total == 0:
        return ""

    size_bytes = train_jsonl.stat().st_size
    if size_bytes >= 1024 ** 3:
        size_str = f"{size_bytes / 1024 ** 3:.2f} GB"
    elif size_bytes >= 1024 ** 2:
        size_str = f"{size_bytes / 1024 ** 2:.2f} MB"
    else:
        size_str = f"{size_bytes / 1024:.1f} KB"

    mid = n_total // 2
    if n_total <= 3:
        indices = list(range(n_total))
    else:
        step = max(1, n_total // 5)
        indices = sorted({max(0, mid - step), mid, min(n_total - 1, mid + step)})[:3]

    samples = []
    for idx in indices:
        try:
            obj = json.loads(lines[idx])
            q, a = "", ""
            for msg in obj.get("conversations", []):
                if msg.get("role") == "user" and not q:
                    q = msg.get("content", "")
                elif msg.get("role") in ("assistant", "gpt") and not a:
                    a = msg.get("content", "")
            if q or a:
                samples.append((idx + 1, q, a))
        except Exception:
            continue

    if not samples:
        return ""

    stats_html = (
        f'<div style="display:flex;align-items:center;gap:14px;padding:10px 16px;'
        f'background:#e8f4fd;border:1px solid #bee5eb;border-radius:8px;'
        f'font-family:monospace;font-size:13px;margin-bottom:10px;">'
        f'<span style="font-size:20px;line-height:1;">&#128202;</span>'
        f'<div>'
        f'<span style="font-weight:bold;color:#0c5460;">Training Dataset</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#0c5460;">{n_total:,}&nbsp;Q&amp;A pairs&nbsp;used&nbsp;in&nbsp;training</span>'
        f'&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
        f'<span style="color:#0c5460;">{size_str}</span>'
        f'</div></div>'
    )

    cards = []
    for pair_num, q, a in samples:
        q_disp = (_html.escape(q[:280]) + "…") if len(q) > 280 else _html.escape(q)
        a_disp = (_html.escape(a[:280]) + "…") if len(a) > 280 else _html.escape(a)
        cards.append(
            f'<div style="border:1px solid #dee2e6;border-radius:6px;margin-bottom:8px;overflow:hidden;">'
            f'<div style="padding:3px 10px;background:#f8f9fa;border-bottom:1px solid #dee2e6;'
            f'font-family:monospace;font-size:11px;color:#888;">Example&nbsp;#{pair_num}&nbsp;of&nbsp;{n_total:,}</div>'
            f'<div style="padding:8px 12px;">'
            f'<div style="margin-bottom:6px;display:flex;gap:6px;">'
            f'<span style="background:#4a90d9;color:white;font-size:11px;padding:1px 7px;'
            f'border-radius:3px;font-weight:bold;flex-shrink:0;align-self:flex-start;">Q</span>'
            f'<span style="color:#333;font-family:monospace;font-size:12px;line-height:1.5;">{q_disp}</span></div>'
            f'<div style="display:flex;gap:6px;">'
            f'<span style="background:#28a745;color:white;font-size:11px;padding:1px 7px;'
            f'border-radius:3px;font-weight:bold;flex-shrink:0;align-self:flex-start;">A</span>'
            f'<span style="color:#555;font-family:monospace;font-size:12px;line-height:1.5;">{a_disp}</span></div>'
            f'</div></div>'
        )

    return (
        f'<div style="margin-top:10px;">'
        f'{stats_html}'
        f'<div style="font-family:monospace;font-size:12px;color:#666;margin-bottom:6px;">'
        f'Sample Q&amp;A pairs from your training data (drawn from the middle of the dataset):</div>'
        + "".join(cards)
        + '</div>'
    )


def _make_qa_review_html(n_samples: int = 15) -> str:
    """
    Build the Step-2 approval preview: stats + question-type breakdown +
    sample Q&A cards spread across all extracted training data.
    """
    all_files = (
        sorted(_TRAINING_DATA_DIR.glob("train_sharegpt.jsonl")) +
        sorted(_TRAINING_DATA_DIR.glob("*/train_sharegpt.jsonl"))
    )
    if not all_files:
        return (
            '<p style="color:#888;font-size:13px;">'
            'No training data yet. Click <b>Extract Training Data</b> above.</p>'
        )

    all_pairs: List[Tuple[str, str, str]] = []  # (source_label, question, answer)
    for jf in all_files:
        label = jf.parent.name if jf.parent != _TRAINING_DATA_DIR else "combined"
        try:
            for raw in jf.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                obj = json.loads(raw)
                q, a = "", ""
                for msg in obj.get("conversations", []):
                    if msg.get("role") == "user" and not q:
                        q = msg.get("content", "")
                    elif msg.get("role") == "assistant" and not a:
                        a = msg.get("content", "")
                if q or a:
                    all_pairs.append((label, q, a))
        except Exception:
            continue

    if not all_pairs:
        return '<p style="color:#888;font-size:13px;">No Q&A pairs found.</p>'

    n_total = len(all_pairs)

    # Question-type breakdown
    q_types: dict = {
        "What is / Describe": 0,
        "Syntax / SQL": 0,
        "Arguments / Params": 0,
        "Example / Demo": 0,
        "Usage notes": 0,
        "Other": 0,
    }
    for _, q, _ in all_pairs:
        ql = q.lower()
        if "syntax" in ql or "write a" in ql or "expression" in ql:
            q_types["Syntax / SQL"] += 1
        elif "argument" in ql or "parameter" in ql:
            q_types["Arguments / Params"] += 1
        elif "example" in ql or "demonstrate" in ql or "show me" in ql:
            q_types["Example / Demo"] += 1
        elif "usage notes" in ql:
            q_types["Usage notes"] += 1
        elif ql.startswith("what is") or ql.startswith("describe"):
            q_types["What is / Describe"] += 1
        else:
            q_types["Other"] += 1

    type_badges = "".join(
        f'<span style="display:inline-block;background:#e9ecef;border-radius:3px;'
        f'padding:2px 8px;font-size:11px;margin:2px;">{k}: <b>{v}</b></span>'
        for k, v in q_types.items() if v > 0
    )

    # Multi-turn count
    mt_files = (
        list(_TRAINING_DATA_DIR.glob("train_multiturn.jsonl")) +
        list(_TRAINING_DATA_DIR.glob("*/train_multiturn.jsonl"))
    )
    n_mt = sum(
        sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        for f in mt_files if f.exists()
    )
    mt_note = (
        f'&nbsp;&middot;&nbsp;<b>{n_mt:,}</b>&nbsp;multi-turn conversations'
        if n_mt > 0 else ""
    )

    # Sample cards spread evenly across the dataset
    step = max(1, n_total // n_samples)
    indices = list(range(0, min(n_total, n_samples * step), step))[:n_samples]
    cards = []
    for idx in indices:
        label, q, a = all_pairs[idx]
        q_disp = (_html.escape(q[:300]) + "…") if len(q) > 300 else _html.escape(q)
        a_disp = (_html.escape(a[:300]) + "…") if len(a) > 300 else _html.escape(a)
        src_badge = (
            f'<span style="font-size:10px;background:#6c757d;color:white;'
            f'padding:1px 5px;border-radius:3px;margin-left:6px;">{_html.escape(label)}</span>'
            if label != "combined" else ""
        )
        _sf = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif'
        cards.append(
            f'<div style="border:1px solid #dee2e6;border-radius:6px;margin-bottom:8px;overflow:hidden;">'
            f'<div style="padding:3px 10px;background:#f8f9fa;border-bottom:1px solid #dee2e6;'
            f'font-family:monospace;font-size:10px;color:#888;">'
            f'Pair #{idx + 1:,} of {n_total:,}{src_badge}</div>'
            f'<div style="padding:10px 14px;">'
            f'<div style="display:flex;gap:8px;margin-bottom:7px;">'
            f'<span style="background:#4a90d9;color:white;font-size:11px;padding:2px 7px;'
            f'border-radius:3px;flex-shrink:0;align-self:flex-start;font-weight:600;">Q</span>'
            f'<span style="color:#1a1a1a;font-size:14px;line-height:1.6;font-family:{_sf};">{q_disp}</span></div>'
            f'<div style="display:flex;gap:8px;">'
            f'<span style="background:#28a745;color:white;font-size:11px;padding:2px 7px;'
            f'border-radius:3px;flex-shrink:0;align-self:flex-start;font-weight:600;">A</span>'
            f'<span style="color:#444;font-size:13px;line-height:1.7;font-family:{_sf};">{a_disp}</span></div>'
            f'</div></div>'
        )

    _SF = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif'
    return (
        f'<div style="margin-top:10px;">'
        f'<div style="padding:10px 16px;background:#e8f4fd;border:1px solid #bee5eb;'
        f'border-radius:8px;font-family:{_SF};font-size:13px;margin-bottom:10px;">'
        f'<b style="color:#0c5460;">{n_total:,} Q&amp;A pairs ready for training</b>'
        f'{mt_note}'
        f'<div style="margin-top:6px;">{type_badges}</div>'
        f'</div>'
        f'<div style="font-family:{_SF};font-size:13px;color:#555;margin-bottom:8px;">'
        f'Showing {len(cards)} samples spread evenly across the dataset. '
        f'Review and click <b>✓ Approve Training Data</b> below to proceed to training.</div>'
        + "".join(cards)
        + '</div>'
    )


def _make_uploaded_files_html() -> str:
    """List files currently in _DATA_DIR."""
    if not _DATA_DIR.exists():
        return ""
    files = sorted(list(_DATA_DIR.glob("*.pdf")) + list(_DATA_DIR.glob("*.csv")))
    if not files:
        return ""
    items = "".join(
        f'<div style="padding:4px 10px;font-family:monospace;font-size:12px;'
        f'border-bottom:1px solid #dee2e6;">'
        f'<span style="background:{"#dc3545" if f.suffix == ".pdf" else "#28a745"};'
        f'color:white;font-size:10px;padding:1px 5px;border-radius:3px;">'
        f'{f.suffix.upper()[1:]}</span>'
        f'&nbsp;{_html.escape(f.name)}'
        f'&nbsp;<span style="color:#888;">({f.stat().st_size // 1024 + 1} KB)</span></div>'
        for f in files
    )
    return (
        f'<div style="border:1px solid #dee2e6;border-radius:6px;overflow:hidden;'
        f'font-family:monospace;margin-top:8px;">'
        f'<div style="padding:6px 10px;background:#f8f9fa;border-bottom:1px solid #dee2e6;'
        f'font-size:12px;font-weight:bold;color:#333;">'
        f'{len(files)} file(s) in upload folder</div>'
        + items + '</div>'
    )


# ---------------------------------------------------------------------------
# Conversation Memory helpers
# ---------------------------------------------------------------------------

def _make_memory_stats_html() -> str:
    """Render a stats bar for the conversation memory."""
    try:
        stats = memory_stats(_MEMORY_DIR)
    except Exception:
        stats = {"total": 0, "approved": 0, "rejected": 0, "pending": 0}
    total    = stats["total"]
    approved = stats["approved"]
    rejected = stats["rejected"]
    pending  = stats["pending"]
    if total == 0:
        return (
            '<div style="padding:10px 14px;background:#f8f9fa;border:1px solid #dee2e6;'
            'border-radius:6px;font-family:monospace;font-size:13px;color:#6c757d;">'
            'No conversations logged yet. Start chatting in Step 5 · Chat to build memory.</div>'
        )
    badges = "".join([
        f'<span style="background:#6c757d;color:white;font-size:11px;padding:2px 8px;'
        f'border-radius:10px;margin-right:6px;">{total} total</span>',
        f'<span style="background:#28a745;color:white;font-size:11px;padding:2px 8px;'
        f'border-radius:10px;margin-right:6px;">{approved} approved</span>',
        f'<span style="background:#ffc107;color:#333;font-size:11px;padding:2px 8px;'
        f'border-radius:10px;margin-right:6px;">{pending} pending</span>',
        f'<span style="background:#dc3545;color:white;font-size:11px;padding:2px 8px;'
        f'border-radius:10px;">{rejected} rejected</span>',
    ])
    return (
        f'<div style="padding:10px 14px;background:#f8f9fa;border:1px solid #dee2e6;'
        f'border-radius:6px;font-family:monospace;font-size:13px;">'
        f'<b>Conversation Memory</b>&nbsp;&nbsp;{badges}</div>'
    )


def _memory_interaction_choices(limit: int = 100) -> List[str]:
    """Return dropdown choices: 'id | Q: <truncated question>'."""
    try:
        records = load_interactions(_MEMORY_DIR, limit=limit)
    except Exception:
        return []
    choices = []
    for r in records:
        ts = r.get("timestamp", "")[:16].replace("T", " ")
        q = r.get("question", "")[:60].rstrip()
        status = {"approved": "✓", "rejected": "✗"}.get(
            {True: "approved", False: "rejected"}.get(r.get("approved"), "pending"), "·"
        )
        choices.append(f"{r['id']} | {status} {ts}  {q!r}")
    return choices


def _parse_interaction_id(choice: str) -> str:
    """Extract the record ID from a dropdown choice string."""
    if not choice:
        return ""
    return choice.split(" | ")[0].strip()


def _make_interaction_detail_html(record_id: str) -> str:
    """Render a single interaction Q+A for review."""
    if not record_id:
        return ""
    try:
        records = load_interactions(_MEMORY_DIR)
    except Exception:
        return ""
    rec = next((r for r in records if r.get("id") == record_id), None)
    if not rec:
        return ""
    approved_val = rec.get("approved")
    status_label = {True: "✓ Approved", False: "✗ Rejected", None: "· Pending review"}[approved_val]
    status_color = {True: "#28a745",     False: "#dc3545",    None: "#ffc107"}[approved_val]
    ts = rec.get("timestamp", "")[:19].replace("T", " ") + " UTC"
    model = rec.get("model_name", "")
    kb = " · KB context injected" if rec.get("kb_context_used") else ""
    return (
        f'<div style="border:1px solid #dee2e6;border-radius:8px;padding:14px 16px;'
        f'font-family:monospace;font-size:13px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;">'
        f'<span style="color:#555;font-size:11px;">{ts}{(" · " + model) if model else ""}{kb}</span>'
        f'<span style="background:{status_color};color:white;font-size:11px;padding:2px 8px;'
        f'border-radius:10px;">{status_label}</span></div>'
        f'<div style="background:#e8f4fd;border-radius:4px;padding:8px 12px;margin-bottom:8px;">'
        f'<b style="color:#0c5460;">Q:</b>&nbsp;{_html.escape(rec.get("question", ""))}</div>'
        f'<div style="background:#f8f9fa;border-radius:4px;padding:8px 12px;white-space:pre-wrap;">'
        f'<b style="color:#333;">A:</b>&nbsp;{_html.escape(rec.get("answer", ""))}</div></div>'
    )


def _make_frequent_questions_html(min_count: int = 2) -> str:
    """Render frequently-asked questions for pattern promotion."""
    try:
        frequent = mine_frequent_questions(_MEMORY_DIR, min_count=min_count)
    except Exception:
        return ""
    if not frequent:
        return (
            '<div style="padding:10px;color:#888;font-family:monospace;font-size:12px;">'
            f'No question asked {min_count}+ times yet.</div>'
        )
    rows = "".join(
        f'<tr style="border-bottom:1px solid #dee2e6;">'
        f'<td style="padding:5px 8px;font-weight:bold;color:#333;">{cnt}×</td>'
        f'<td style="padding:5px 8px;color:#0c5460;">{_html.escape(q[:80])}</td>'
        f'<td style="padding:5px 8px;font-size:11px;color:#555;">'
        f'{_html.escape(a[:60])}…</td></tr>'
        for q, cnt, a in frequent[:20]
    )
    return (
        f'<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">'
        f'<thead><tr style="background:#f8f9fa;">'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Count</th>'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Question</th>'
        f'<th style="padding:5px 8px;text-align:left;color:#666;">Latest answer (preview)</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )


# Memory action handlers

def _memory_refresh() -> Tuple[str, str, List[str]]:
    """Refresh stats, reset dropdown."""
    return (
        _make_memory_stats_html(),
        _make_frequent_questions_html(),
        _memory_interaction_choices(),
    )


def _memory_select(choice: str) -> Tuple[str, str]:
    """Load detail for selected interaction."""
    rid = _parse_interaction_id(choice)
    return rid, _make_interaction_detail_html(rid)


def _memory_approve(record_id: str) -> Tuple[str, str, str, List[str]]:
    if record_id:
        set_approval(_MEMORY_DIR, record_id, True)
    return (
        "Approved." if record_id else "No interaction selected.",
        _make_memory_stats_html(),
        _make_interaction_detail_html(record_id),
        _memory_interaction_choices(),
    )


def _memory_reject(record_id: str) -> Tuple[str, str, str, List[str]]:
    if record_id:
        set_approval(_MEMORY_DIR, record_id, False)
    return (
        "Rejected." if record_id else "No interaction selected.",
        _make_memory_stats_html(),
        _make_interaction_detail_html(record_id),
        _memory_interaction_choices(),
    )


def _memory_export() -> Tuple[str, object]:
    """Export approved interactions to training_data/memory_approved.jsonl."""
    out_path = _TRAINING_DATA_DIR / "memory_approved.jsonl"
    try:
        count = export_approved_to_jsonl(_MEMORY_DIR, out_path)
        if count == 0:
            return "No approved interactions to export. Approve some conversations first.", gr.update(visible=False)
        return (
            f"Exported {count} approved interactions → {out_path}. "
            f"They will be included in the next training run if 'Include conversation memory' is checked.",
            gr.update(value=str(out_path), visible=True),
        )
    except Exception as exc:
        return f"Export error: {exc}", gr.update(visible=False)


# ---------------------------------------------------------------------------
# Knowledge Library helpers
# ---------------------------------------------------------------------------

def _make_library_html() -> str:
    """Render the knowledge library as an HTML table."""
    stats = library_stats(_LIBRARY_DIR)
    entries = stats.get("entries", [])
    total_pat = stats.get("total_patterns", 0)
    total_qa = stats.get("total_qa_pairs", 0)
    by_cat = stats.get("by_category", {})

    cat_badges = " ".join(
        f'<span style="background:#6c757d;color:white;font-size:10px;'
        f'padding:2px 6px;border-radius:10px;margin:2px;">{c}: {n}</span>'
        for c, n in sorted(by_cat.items())
    )

    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 12px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px 6px 0 0;">'
        f'<span style="font-weight:bold;">{total_pat} patterns · {total_qa} Q&A pairs</span>'
        f'<span>{cat_badges}</span></div>'
    )

    if not entries:
        return (
            header +
            '<div style="padding:16px;border:1px solid #dee2e6;border-top:none;'
            'border-radius:0 0 6px 6px;color:#888;text-align:center;">'
            'No patterns yet. Use the form above to add your first one.</div>'
        )

    rows = []
    for e in entries:
        title = _html.escape(e.get("title", e.get("slug", "?")))
        cat = _html.escape(e.get("category") or "general")
        qa_n = e.get("qa_count", 0)
        created = (e.get("created") or "")[:10]
        slug = _html.escape(e.get("slug", ""))
        rows.append(
            f'<tr>'
            f'<td style="padding:6px 10px;font-weight:500;">{title}</td>'
            f'<td style="padding:6px 10px;color:#6c757d;">{cat}</td>'
            f'<td style="padding:6px 10px;text-align:center;">{qa_n}</td>'
            f'<td style="padding:6px 10px;color:#6c757d;font-size:12px;">{created}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;font-size:11px;color:#888;">{slug}</td>'
            f'</tr>'
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;border:1px solid #dee2e6;'
        'border-top:none;border-radius:0 0 6px 6px;overflow:hidden;">'
        '<thead><tr style="background:#e9ecef;">'
        '<th style="padding:6px 10px;text-align:left;">Name</th>'
        '<th style="padding:6px 10px;text-align:left;">Category</th>'
        '<th style="padding:6px 10px;text-align:center;">Q&A pairs</th>'
        '<th style="padding:6px 10px;text-align:left;">Added</th>'
        '<th style="padding:6px 10px;text-align:left;">Slug (for delete)</th>'
        '</tr></thead><tbody>'
        + "".join(rows) +
        '</tbody></table>'
    )
    return header + table


def _make_qa_preview_html(qa_pairs: list, max_show: int = 10) -> str:
    """Render a preview of Q&A pairs for the knowledge capture form."""
    if not qa_pairs:
        return '<div style="color:#888;padding:8px;">Fill in the fields above and click Preview.</div>'
    shown = qa_pairs[:max_show]
    items = []
    for q, a in shown:
        items.append(
            f'<div style="margin-bottom:10px;padding:8px 12px;'
            f'background:#f8f9fa;border-radius:6px;border-left:3px solid #0d6efd;">'
            f'<div style="font-weight:600;color:#0d6efd;font-size:13px;">Q: {_html.escape(q)}</div>'
            f'<div style="color:#333;font-size:12px;margin-top:4px;white-space:pre-wrap;">'
            f'{_html.escape(a[:300])}{"..." if len(a) > 300 else ""}</div>'
            f'</div>'
        )
    total = len(qa_pairs)
    footer = (
        f'<div style="color:#6c757d;font-size:12px;margin-top:4px;">'
        f'Showing {len(shown)} of {total} Q&A pairs that would be generated.</div>'
        if total > max_show else ""
    )
    return "".join(items) + footer


def _kb_preview(
    title: str, description: str, use_cases_text: str, parameters_text: str,
    sql_example: str, sql_description: str, example_output: str,
    common_errors_text: str, best_practices: str, category: str,
) -> str:
    form_data = {
        "title": title, "description": description, "use_cases_text": use_cases_text,
        "parameters_text": parameters_text, "sql_example": sql_example,
        "sql_description": sql_description, "example_output": example_output,
        "common_errors_text": common_errors_text, "best_practices": best_practices,
        "category": category,
    }
    if not title.strip() or not description.strip():
        return '<div style="color:#888;padding:8px;">Fill in at least Name and Description to see a preview.</div>'
    qa, _ = preview_qa(form_data)
    return _make_qa_preview_html(qa)


def _kb_save(
    title: str, description: str, use_cases_text: str, parameters_text: str,
    sql_example: str, sql_description: str, example_output: str,
    common_errors_text: str, best_practices: str, category: str,
) -> Tuple[str, str]:
    """Save the form as a library entry. Returns (status_msg, library_html)."""
    if not title.strip():
        return "Name is required.", _make_library_html()
    if not description.strip():
        return "Description is required.", _make_library_html()
    form_data = {
        "title": title, "description": description, "use_cases_text": use_cases_text,
        "parameters_text": parameters_text, "sql_example": sql_example,
        "sql_description": sql_description, "example_output": example_output,
        "common_errors_text": common_errors_text, "best_practices": best_practices,
        "category": category,
    }
    try:
        pattern = form_to_pattern(form_data)
        saved_path, qa_count = save_to_library(pattern, _LIBRARY_DIR)
        # Reload retriever so the new entry is immediately available at chat time
        _get_retriever().reload()
        return (
            f"Saved '{title}' to library ({qa_count} Q&A pairs). "
            f"It will be included automatically next time you extract training data "
            f"and is immediately available in Chat.",
            _make_library_html(),
        )
    except Exception as e:
        return f"Error saving: {e}", _make_library_html()


def _kb_delete(slug: str) -> Tuple[str, str]:
    """Delete a library entry by slug."""
    slug = slug.strip()
    if not slug:
        return "Enter a slug from the table above.", _make_library_html()
    deleted = delete_from_library(slug, _LIBRARY_DIR)
    if deleted:
        return f"Deleted '{slug}' from library.", _make_library_html()
    return f"Entry '{slug}' not found.", _make_library_html()


# ---------------------------------------------------------------------------
# Model artifact management
# ---------------------------------------------------------------------------

def _compute_quality_rating(
    final_loss: Optional[float],
    final_eval_loss: Optional[float],
    initial_loss: Optional[float],
) -> str:
    """Return quality label matching _make_quality_html logic."""
    if final_loss is None:
        return ""
    primary = final_eval_loss if final_eval_loss is not None else final_loss
    reduction_pct: Optional[float] = None
    if initial_loss and initial_loss > 0:
        reduction_pct = (initial_loss - final_loss) / initial_loss * 100
    if primary < 1.2 and (reduction_pct is None or reduction_pct > 40):
        return "Excellent"
    if primary < 1.8 and (reduction_pct is None or reduction_pct > 25):
        return "Good"
    if primary < 2.2 and (reduction_pct is None or reduction_pct > 10):
        return "Okay"
    if primary < 2.8:
        return "Fair"
    return "Poor"


def _dir_size_str(path: Path) -> str:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return "?"
    if total >= 1024 ** 3:
        return f"{total / 1024 ** 3:.2f} GB"
    if total >= 1024 ** 2:
        return f"{total / 1024 ** 2:.1f} MB"
    return f"{total / 1024:.0f} KB"


def _list_all_artifacts() -> list:
    """Return a list of dicts for output_model and all named saved_models."""
    artifacts = []
    active_source = ""
    if _OUTPUT_MODEL_DIR.exists() and (_OUTPUT_MODEL_DIR / "config.json").exists():
        base_model = ""
        try:
            cfg = json.loads((_OUTPUT_MODEL_DIR / "config.json").read_text())
            base_model = cfg.get("_name_or_path", "")
        except Exception:
            pass
        try:
            om_meta = json.loads((_OUTPUT_MODEL_DIR / "_meta.json").read_text())
            active_source = om_meta.get("active_source", "")
        except Exception:
            pass
        artifacts.append({
            "name": "output_model",
            "path": _OUTPUT_MODEL_DIR,
            "size": _dir_size_str(_OUTPUT_MODEL_DIR),
            "saved_at": datetime.fromtimestamp(_OUTPUT_MODEL_DIR.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "base_model": base_model,
            "is_current": True,
            "active_source": active_source,
        })
    if _SAVED_MODELS_DIR.exists():
        for d in sorted(_SAVED_MODELS_DIR.iterdir()):
            if not d.is_dir() or not (d / "config.json").exists():
                continue
            meta: dict = {}
            meta_file = d / "_meta.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except Exception:
                    pass
            saved_at = meta.get("saved_at") or datetime.fromtimestamp(
                d.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M")
            artifacts.append({
                "name": d.name,
                "path": d,
                "size": _dir_size_str(d),
                "saved_at": saved_at,
                "base_model": meta.get("base_model", ""),
                "quality": meta.get("quality", ""),
                "is_current": d.name == active_source,
                "active_source": "",
            })
    return artifacts


_QUALITY_COLORS = {
    "Excellent": ("#28a745", "#d4edda", "#155724"),
    "Good":      ("#34a853", "#d4edda", "#1b6b3a"),
    "Okay":      ("#ffc107", "#fff3cd", "#7d5a00"),
    "Fair":      ("#fd7e14", "#ffe5d0", "#7d3200"),
    "Poor":      ("#dc3545", "#f8d7da", "#721c24"),
}


def _make_artifacts_html(artifacts: list) -> str:
    if not artifacts:
        return (
            '<div style="padding:16px;background:#f8f9fa;border:1px solid #dee2e6;'
            'border-radius:8px;font-family:monospace;font-size:13px;color:#6c757d;">'
            'No saved models found. Train a model and click <b>Save Model</b> to preserve it.'
            '</div>'
        )
    cards = []
    for a in artifacts:
        # Active badge
        if a["is_current"]:
            src = a.get("active_source", "")
            label = f"Active · {src}" if src else "Active"
            active_badge = (
                f'<span style="background:#0d6efd;color:white;font-size:10px;padding:2px 8px;'
                f'border-radius:3px;margin-left:6px;">{label}</span>'
            )
        else:
            active_badge = ""
        # Quality badge
        quality = a.get("quality", "")
        if quality and quality in _QUALITY_COLORS:
            qc, _, _ = _QUALITY_COLORS[quality]
            quality_badge = (
                f'<span style="background:{qc};color:white;font-size:10px;padding:2px 8px;'
                f'border-radius:3px;margin-left:6px;">{quality}</span>'
            )
        else:
            quality_badge = ""
        base_html = (
            f'<span style="color:#555;">Base:&nbsp;{_html.escape(a["base_model"])}</span>&nbsp;&middot;&nbsp;'
            if a.get("base_model") else ""
        )
        border_color = "#0d6efd" if a["is_current"] else "#dee2e6"
        bg_color = "#f0f5ff" if a["is_current"] else "white"
        cards.append(
            f'<div style="border:1px solid {border_color};border-radius:8px;padding:10px 14px;'
            f'margin-bottom:6px;background:{bg_color};">'
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;margin-bottom:4px;">'
            f'<span style="font-family:monospace;font-size:14px;font-weight:bold;color:#222;">'
            f'{_html.escape(a["name"])}</span>{active_badge}{quality_badge}</div>'
            f'<div style="font-family:monospace;font-size:11px;color:#777;">'
            f'{base_html}'
            f'<span>{a["size"]}</span>&nbsp;&middot;&nbsp;'
            f'<span>Saved&nbsp;{a["saved_at"]}</span></div></div>'
        )
    return '<div>' + ''.join(cards) + '</div>'


def _artifact_choices(artifacts: list) -> list:
    return [a["name"] for a in artifacts]


def _save_current_model(name: str) -> Tuple[str, str, object]:
    """Copy output_model → saved_models/<name>. Returns (status, html, dropdown_update)."""
    name = name.strip()
    if not name:
        arts = _list_all_artifacts()
        return "Enter a name for the model.", _make_artifacts_html(arts), gr.update()
    name = re.sub(r"[^\w\-\.]", "_", name)
    if not (_OUTPUT_MODEL_DIR / "config.json").exists():
        arts = _list_all_artifacts()
        return (
            "No trained model found in output_model/ — complete training first.",
            _make_artifacts_html(arts), gr.update(),
        )
    dest = _SAVED_MODELS_DIR / name
    if dest.exists():
        arts = _list_all_artifacts()
        return (
            f"A model named '{name}' already exists. Choose a different name.",
            _make_artifacts_html(arts), gr.update(),
        )
    try:
        _SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(_OUTPUT_MODEL_DIR), str(dest))
        base_model = ""
        try:
            cfg = json.loads((_OUTPUT_MODEL_DIR / "config.json").read_text())
            base_model = cfg.get("_name_or_path", "")
        except Exception:
            pass
        quality = _compute_quality_rating(
            _training_state.get("last_loss"),
            _training_state.get("last_eval_loss"),
            _training_state.get("initial_loss"),
        )
        (dest / "_meta.json").write_text(json.dumps({
            "name": name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "base_model": base_model,
            "quality": quality,
        }, indent=2))
        arts = _list_all_artifacts()
        return (
            f"Saved as '{name}' ({_dir_size_str(dest)}).",
            _make_artifacts_html(arts),
            gr.update(choices=_artifact_choices(arts), value=None),
        )
    except Exception as exc:
        arts = _list_all_artifacts()
        return f"Error: {exc}", _make_artifacts_html(arts), gr.update()


def _rename_artifact(old_name: str, new_name: str) -> Tuple[str, str, object]:
    """Rename a saved model artifact. Returns (status, html, dropdown_update)."""
    new_name = new_name.strip()
    if not old_name:
        arts = _list_all_artifacts()
        return "Select an artifact to rename.", _make_artifacts_html(arts), gr.update()
    if not new_name:
        arts = _list_all_artifacts()
        return "Enter a new name.", _make_artifacts_html(arts), gr.update()
    if old_name == "output_model":
        arts = _list_all_artifacts()
        return (
            "Cannot rename 'output_model' directly. Save it first, then rename the copy.",
            _make_artifacts_html(arts), gr.update(),
        )
    new_name = re.sub(r"[^\w\-\.]", "_", new_name)
    src = _SAVED_MODELS_DIR / old_name
    if not src.exists():
        arts = _list_all_artifacts()
        return f"'{old_name}' not found.", _make_artifacts_html(arts), gr.update()
    dest = _SAVED_MODELS_DIR / new_name
    if dest.exists():
        arts = _list_all_artifacts()
        return f"Name '{new_name}' is already in use.", _make_artifacts_html(arts), gr.update()
    try:
        src.rename(dest)
        meta_file = dest / "_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta["name"] = new_name
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass
        arts = _list_all_artifacts()
        return (
            f"Renamed '{old_name}' → '{new_name}'.",
            _make_artifacts_html(arts),
            gr.update(choices=_artifact_choices(arts), value=None),
        )
    except Exception as exc:
        arts = _list_all_artifacts()
        return f"Error: {exc}", _make_artifacts_html(arts), gr.update()


def _set_active_model(artifact_name: str) -> Tuple[str, str, object]:
    """Copy a saved model to _OUTPUT_MODEL_DIR to make it the active model for Chat."""
    global _model, _tokenizer
    if not artifact_name:
        arts = _list_all_artifacts()
        return "Select a saved model to set as active.", _make_artifacts_html(arts), gr.update()
    if artifact_name == "output_model":
        arts = _list_all_artifacts()
        return "output_model is already the active model.", _make_artifacts_html(arts), gr.update()
    src = _SAVED_MODELS_DIR / artifact_name
    if not src.exists():
        arts = _list_all_artifacts()
        return f"'{artifact_name}' not found.", _make_artifacts_html(arts), gr.update()
    try:
        _model = None
        _tokenizer = None
        if _OUTPUT_MODEL_DIR.exists():
            shutil.rmtree(str(_OUTPUT_MODEL_DIR))
        shutil.copytree(str(src), str(_OUTPUT_MODEL_DIR))
        # Record which saved model is the active source so the UI can display it
        try:
            meta_path = _OUTPUT_MODEL_DIR / "_meta.json"
            meta: dict = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            meta["active_source"] = artifact_name
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception:
            pass
        _pipeline_status["train"] = "complete"
        arts = _list_all_artifacts()
        return (
            f"'{artifact_name}' is now the active model. Go to Step 5 · Chat and click Load Model.",
            _make_artifacts_html(arts),
            gr.update(choices=_artifact_choices(arts), value=None),
        )
    except Exception as exc:
        arts = _list_all_artifacts()
        return f"Error: {exc}", _make_artifacts_html(arts), gr.update()


def _delete_artifact(artifact_name: str) -> Tuple[str, str, object]:
    """Delete a named artifact. Returns (status, html, dropdown_update)."""
    global _model, _tokenizer
    if not artifact_name:
        arts = _list_all_artifacts()
        return "No artifact selected.", _make_artifacts_html(arts), gr.update()
    target = _OUTPUT_MODEL_DIR if artifact_name == "output_model" else _SAVED_MODELS_DIR / artifact_name
    if not target.exists():
        arts = _list_all_artifacts()
        return f"'{artifact_name}' not found.", _make_artifacts_html(arts), gr.update()
    try:
        if artifact_name == "output_model":
            _model = None
            _tokenizer = None
            _pipeline_status["train"] = "pending"
        shutil.rmtree(str(target))
        arts = _list_all_artifacts()
        return (
            f"Deleted '{artifact_name}'.",
            _make_artifacts_html(arts),
            gr.update(choices=_artifact_choices(arts), value=None),
        )
    except Exception as exc:
        arts = _list_all_artifacts()
        return f"Error deleting '{artifact_name}': {exc}", _make_artifacts_html(arts), gr.update()


def _refresh_artifacts() -> Tuple[str, object]:
    arts = _list_all_artifacts()
    return _make_artifacts_html(arts), gr.update(choices=_artifact_choices(arts), value=None)


# ---------------------------------------------------------------------------
# Subprocess streaming helper
# ---------------------------------------------------------------------------

def _iter_lines(proc, heartbeat: float = 1.0) -> Generator[Optional[str], None, None]:
    """Read subprocess stdout in a background thread, yielding lines as they arrive.
    Yields None roughly every *heartbeat* seconds when the process is silent.
    """
    q: "queue.Queue[Optional[str]]" = queue.Queue()

    def _reader():
        buf = b""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\r", b"\n"):
                line = buf.decode("utf-8", errors="replace").strip()
                if line:
                    q.put(line + "\n")
                buf = b""
            else:
                buf += ch
        if buf:
            line = buf.decode("utf-8", errors="replace").strip()
            if line:
                q.put(line + "\n")
        proc.wait()
        q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    while True:
        try:
            item = q.get(timeout=heartbeat)
            if item is None:
                break
            yield item
        except queue.Empty:
            yield None


def _popen(cmd: list) -> subprocess.Popen:
    """Launch cmd as a subprocess with unbuffered, merged stdout+stderr."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )


# ---------------------------------------------------------------------------
# Step 1 — Upload Documents
# ---------------------------------------------------------------------------

def _upload_file_choices() -> list:
    """Return list of filenames currently in _DATA_DIR."""
    if not _DATA_DIR.exists():
        return []
    return sorted(
        f.name for f in _DATA_DIR.iterdir()
        if f.suffix.lower() in (".pdf", ".csv")
    )


def _save_uploads(files) -> Tuple[str, str, str, object]:
    """Copy uploaded files to _DATA_DIR. Returns (status, files_html, pipeline_html, dropdown_update)."""
    if not files:
        return (
            "No files selected. Please select at least one PDF or CSV.",
            _make_uploaded_files_html(),
            _make_pipeline_html(),
            gr.update(choices=_upload_file_choices()),
        )
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdfs, csvs = [], []
    for f in files:
        src = Path(f)
        dest = _DATA_DIR / src.name
        shutil.copy2(src, dest)
        if src.suffix.lower() == ".csv":
            csvs.append(src.name)
        else:
            pdfs.append(src.name)
    _pipeline_status["upload"] = "complete"
    parts = []
    if pdfs:
        parts.append(f"{len(pdfs)} PDF(s): {', '.join(pdfs)}")
    if csvs:
        parts.append(f"{len(csvs)} CSV(s): {', '.join(csvs)}")
    msg = "Uploaded " + "; ".join(parts) + f" → {_DATA_DIR}"
    return msg, _make_uploaded_files_html(), _make_pipeline_html(), gr.update(choices=_upload_file_choices())


def _remove_file(filename: str) -> Tuple[str, str, str, object]:
    """Delete a file from _DATA_DIR. Returns (status, files_html, pipeline_html, dropdown_update)."""
    if not filename:
        return (
            "No file selected.",
            _make_uploaded_files_html(),
            _make_pipeline_html(),
            gr.update(choices=_upload_file_choices()),
        )
    target = _DATA_DIR / filename
    if not target.exists():
        return (
            f"File '{filename}' not found.",
            _make_uploaded_files_html(),
            _make_pipeline_html(),
            gr.update(choices=_upload_file_choices()),
        )
    try:
        target.unlink()
        remaining = _upload_file_choices()
        if not remaining:
            _pipeline_status["upload"] = "pending"
        return (
            f"Removed '{filename}'.",
            _make_uploaded_files_html(),
            _make_pipeline_html(),
            gr.update(choices=remaining, value=None),
        )
    except Exception as exc:
        return (
            f"Error removing '{filename}': {exc}",
            _make_uploaded_files_html(),
            _make_pipeline_html(),
            gr.update(choices=_upload_file_choices()),
        )


# ---------------------------------------------------------------------------
# Step 2 — Extract Training Data
# ---------------------------------------------------------------------------

def _run_extraction(
    chunk_size: int,
    chunk_overlap: int,
    val_ratio: float,
    fmt: str,
    manual_mode: bool,
    no_multiturn: bool,
    include_memory: bool = True,
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    """
    Run the extraction pipeline. Yields:
        (log_text, elapsed_html, pipeline_html, qa_review_html, activity_text)
    """
    start_time = time.time()
    _pipeline_status["extract"] = "running"
    _pipeline_status["approve"] = "pending"
    log_text = "Starting training data extraction…\n"
    blank = '<p style="color:#888;font-family:monospace;font-size:13px;">Running extraction…</p>'
    yield log_text, _make_elapsed_html(0.0), _make_pipeline_html(), blank, _data_activity_text(log_text)

    if not _DATA_DIR.exists():
        log_text += f"ERROR: {_DATA_DIR} does not exist. Upload files in Step 1 first.\n"
        _pipeline_status["extract"] = "failed"
        yield (log_text, _make_elapsed_html(time.time() - start_time, failed=True),
               _make_pipeline_html(), "", _data_activity_text(log_text))
        return

    pdfs = sorted(_DATA_DIR.glob("*.pdf"))
    csvs = sorted(_DATA_DIR.glob("*.csv"))
    if not pdfs and not csvs:
        log_text += f"ERROR: No PDF or CSV files in {_DATA_DIR}. Upload files in Step 1 first.\n"
        _pipeline_status["extract"] = "failed"
        yield (log_text, _make_elapsed_html(time.time() - start_time, failed=True),
               _make_pipeline_html(), "", _data_activity_text(log_text))
        return

    log_text += f"Found {len(pdfs)} PDF(s) and {len(csvs)} CSV(s) in {_DATA_DIR}\n"
    yield (log_text, _make_elapsed_html(time.time() - start_time),
           _make_pipeline_html(), blank, _data_activity_text(log_text))

    _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        if manual_mode:
            cmd.append("--manual")
        if no_multiturn and manual_mode:
            cmd.append("--no-multiturn")
    if csvs:
        cmd += ["--csv", str(csvs[0])]
    # Auto-include the knowledge library if it has any entries
    library_included = _LIBRARY_DIR.exists() and any(_LIBRARY_DIR.glob("*.yaml"))
    if library_included:
        cmd += ["--yaml-dir", str(_LIBRARY_DIR)]
    # Include approved conversation memory if requested and available
    mem_file = _MEMORY_DIR / "interactions.jsonl"
    if include_memory and mem_file.exists():
        cmd += ["--memory-dir", str(_MEMORY_DIR)]
    # Log knowledge library stats when the library was included (not gated by memory)
    if library_included:
        lib = library_stats(_LIBRARY_DIR)
        log_text += (
            f"Knowledge library: {lib['total_patterns']} pattern(s), "
            f"{lib['total_qa_pairs']} Q&A pairs — auto-included.\n"
        )
        yield (log_text, _make_elapsed_html(time.time() - start_time),
               _make_pipeline_html(), blank, _data_activity_text(log_text))

    log_text += f"Running: {' '.join(cmd)}\n\n"
    yield (log_text, _make_elapsed_html(time.time() - start_time),
           _make_pipeline_html(), blank, _data_activity_text(log_text))

    proc = _popen(cmd)
    for line in _iter_lines(proc):
        if line:
            log_text += line
        yield (log_text, _make_elapsed_html(time.time() - start_time),
               _make_pipeline_html(), blank, _data_activity_text(log_text))

    elapsed = time.time() - start_time
    if proc.returncode == 0:
        log_text += "\nExtraction complete.\n"
        out_files = (
            sorted(_TRAINING_DATA_DIR.glob("*.jsonl")) +
            sorted(_TRAINING_DATA_DIR.glob("*/*.jsonl"))
        )
        if out_files:
            log_text += "Output files:\n" + "\n".join(f"  {f}" for f in out_files) + "\n"

        train_jsonl = _find_train_jsonl()
        n_ex = 0
        if train_jsonl:
            try:
                n_ex = sum(1 for ln in train_jsonl.open(encoding="utf-8") if ln.strip())
            except Exception:
                pass
        if n_ex < 20:
            _pipeline_status["extract"] = "warning"
            log_text += f"WARNING: Only {n_ex} training examples generated — quality may be poor.\n"
        else:
            _pipeline_status["extract"] = "complete"

        review_html = _make_qa_review_html()
        yield (log_text,
               _make_elapsed_html(elapsed, done=(_pipeline_status["extract"] == "complete")),
               _make_pipeline_html(), review_html, _data_activity_text(log_text))
    else:
        log_text += f"\nExtraction FAILED (exit code {proc.returncode}).\n"
        _pipeline_status["extract"] = "failed"
        yield (log_text, _make_elapsed_html(elapsed, failed=True),
               _make_pipeline_html(), "", _data_activity_text(log_text))


def _approve_training_data() -> Tuple[str, str]:
    """Mark training data as approved. Returns (status_text, pipeline_html)."""
    train_jsonl = _find_train_jsonl()
    if not train_jsonl:
        return "No training data found. Run extraction first.", _make_pipeline_html()
    try:
        n = sum(1 for ln in train_jsonl.open(encoding="utf-8") if ln.strip())
    except Exception:
        n = 0
    if n == 0:
        return "Training data file is empty. Re-run extraction.", _make_pipeline_html()
    _pipeline_status["approve"] = "complete"
    return f"✓ Approved {n:,} training pairs. Proceed to Step 3 · Train.", _make_pipeline_html()


# ---------------------------------------------------------------------------
# Step 3 — Training
# ---------------------------------------------------------------------------

def _run_training(
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_len: int,
    resume_ckpt: bool = False,
    save_steps: int = 50,
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    """Yields (status_card, progress_bar, pipeline, quality, raw_log, chart, activity)."""
    device_label = _get_device_label()
    use_unsloth = _unsloth_available()
    script = "train.finetune_unsloth" if use_unsloth else "train.finetune_cpu"

    _pipeline_status["train"] = "running"

    phase = "loading"
    current_msg = f"Loading model {model_name}…"
    warnings: List[str] = []
    error: Optional[str] = None

    if "CPU" in device_label and not use_unsloth:
        warnings.append(
            "Training on CPU is very slow. "
            "For a 1.1B model expect several minutes per step."
        )

    current_epoch = 0
    total_epochs_parsed = epochs
    global_step = 0
    total_steps = 0
    last_pct = 0.0
    initial_loss: Optional[float] = None
    last_loss: Optional[float] = None
    last_eval_loss: Optional[float] = None
    n_examples = 0
    loss_history: List[dict] = []

    _training_state.update({
        "active": True, "done": False, "failed": False,
        "log": "", "phase": phase, "current_msg": current_msg,
        "warnings": list(warnings), "error": None,
        "current_epoch": 0, "total_epochs": epochs,
        "global_step": 0, "total_steps": 0, "last_pct": 0.0,
        "initial_loss": None, "last_loss": None, "last_eval_loss": None,
        "n_examples": 0, "loss_history": [], "start_time": None, "elapsed": 0.0,
    })

    def _sync() -> None:
        _training_state.update({
            "log": log_text,
            "phase": phase,
            "current_msg": current_msg,
            "warnings": list(warnings),
            "error": error,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs_parsed,
            "global_step": global_step,
            "total_steps": total_steps,
            "last_pct": last_pct,
            "initial_loss": initial_loss,
            "last_loss": last_loss,
            "last_eval_loss": last_eval_loss,
            "n_examples": n_examples,
            "loss_history": list(loss_history),
        })

    log_text = (
        f"Device: {device_label}\n"
        f"Backend: {'Unsloth (GPU fast path)' if use_unsloth else 'HuggingFace Trainer (CPU/MPS)'}\n\n"
    )
    progress_html = ""
    pipe_html = _make_pipeline_html()

    def _card(done: bool = False, failed: bool = False) -> str:
        return _make_train_status_html(phase, current_msg, warnings, error, done, failed)

    _no_chart = gr.update(visible=False)

    yield _card(), progress_html, pipe_html, "", log_text, _no_chart, _activity_text(log_text)

    # ── Validate training files ───────────────────────────────────────────
    train_file, val_file = _find_training_files()
    if not train_file:
        error = "Training data not found. Complete Steps 1–3 (Upload, Extract, Approve) first."
        _pipeline_status["train"] = "failed"
        log_text += f"ERROR: {error}\n"
        yield _card(failed=True), "", _make_pipeline_html(), "", log_text, _no_chart, ""
        return
    if not val_file:
        error = "Validation data not found. Complete Steps 1–3 first."
        _pipeline_status["train"] = "failed"
        log_text += f"ERROR: {error}\n"
        yield _card(failed=True), "", _make_pipeline_html(), "", log_text, _no_chart, ""
        return

    _OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = _latest_checkpoint(_OUTPUT_MODEL_DIR) if resume_ckpt else None
    if resume_ckpt:
        if ckpt:
            log_text += f"Resuming from checkpoint: {ckpt.name}\n\n"
        else:
            log_text += "Resume requested but no checkpoint found — starting from scratch.\n\n"

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
        "--save-steps", str(save_steps),
    ]
    if ckpt:
        cmd += ["--resume", str(ckpt)]
    log_text += f"Running: {' '.join(cmd)}\n\n"
    yield _card(), progress_html, _make_pipeline_html(), "", log_text, _no_chart, _activity_text(log_text)

    start_time = time.time()
    _training_state["start_time"] = start_time
    proc = _popen(cmd)

    for line in _iter_lines(proc):
        elapsed = time.time() - start_time
        if line:
            log_text += line
            clean = line.strip()

            if "Loading base model:" in clean:
                phase = "loading"
                current_msg = f"Loading model weights for {model_name}…"
            elif m := _TQDM_WEIGHTS_RE.search(clean):
                current_msg = f"Loading model weights… {m.group(1)}%"
            elif "Model loaded." in clean:
                current_msg = "Model weights loaded."
            elif "Adding LoRA adapters" in clean:
                phase = "lora"
                current_msg = "Adding LoRA adapters for fine-tuning…"
            elif m := _TRAINABLE_RE.search(clean):
                current_msg = (
                    f"LoRA ready — {m.group(1)} trainable / {m.group(2)} total "
                    f"params ({m.group(3)}% of model)"
                )
            elif "Loading dataset" in clean:
                phase = "dataset"
                current_msg = "Loading training dataset…"
            elif m := _EXAMPLES_RE.search(clean):
                n_examples = int(m.group(1))
                phase = "training"
                current_msg = f"Training with {n_examples:,} examples…"
            elif "Merging LoRA weights" in clean:
                phase = "saving"
                current_msg = "Merging LoRA weights and saving model…"
            elif "Saved to" in clean:
                current_msg = "Model saved successfully!"

            if "WARNING:" in clean and "very slow" not in clean:
                w = clean.split("WARNING:", 1)[-1].strip()
                if w and w not in warnings:
                    warnings.append(w)

            if "Traceback (most recent call last)" in clean:
                if not error:
                    error = "An error occurred — see developer log for details."
            if clean.startswith("Error") and ":" in clean[:40]:
                if not error:
                    error = clean

            m = _EPOCH_RE.search(clean)
            if m:
                current_epoch = int(m.group(1))
                total_epochs_parsed = int(m.group(2))
                current_msg = f"Training — epoch {current_epoch} of {total_epochs_parsed}…"

            m = _STEP_RE.search(clean)
            if m:
                global_step = int(m.group(1))
                total_steps = int(m.group(2))
                last_pct = float(m.group(3))

            m = _LOSS_RE.search(clean)
            if m:
                v = float(m.group(1))
                if initial_loss is None:
                    initial_loss = v
                last_loss = v
                loss_history.append({"step": global_step, "loss": v, "metric": "train"})

            m = _EVAL_LOSS_RE.search(clean)
            if m:
                last_eval_loss = float(m.group(1))
                loss_history.append({"step": global_step, "loss": last_eval_loss, "metric": "eval"})

        if phase in ("training", "saving") or global_step > 0:
            progress_html = _make_training_progress_html(
                max(current_epoch, 1) if global_step > 0 else current_epoch,
                total_epochs_parsed, global_step, total_steps, last_pct, elapsed, last_loss,
            )

        chart_update = (
            gr.update(value=pd.DataFrame(loss_history), visible=True)
            if loss_history else _no_chart
        )
        _sync()
        yield _card(), progress_html, _make_pipeline_html(), "", log_text, chart_update, _activity_text(log_text)

    elapsed = time.time() - start_time
    final_chart = (
        gr.update(value=pd.DataFrame(loss_history), visible=True)
        if loss_history else _no_chart
    )
    if proc.returncode == 0:
        _pipeline_status["train"] = "complete"
        current_msg = "Training complete! Go to Step 5 · Chat and load your model."
        log_text += "\nTraining complete.\n"
        progress_html = _make_training_progress_html(
            total_epochs_parsed, total_epochs_parsed,
            total_steps, total_steps, 100.0, elapsed, last_loss, done=True,
        )
        quality_html = _make_quality_html(last_loss, last_eval_loss, initial_loss, n_examples)
        _training_state.update({"active": False, "done": True, "elapsed": elapsed})
        _sync()
        yield (
            _card(done=True), progress_html, _make_pipeline_html(),
            quality_html, log_text, final_chart, _activity_text(log_text),
        )
    else:
        _pipeline_status["train"] = "failed"
        if not error:
            error = f"Training failed (exit code {proc.returncode}). Check the developer log."
        log_text += f"\nTraining FAILED (exit code {proc.returncode}).\n"
        progress_html = _make_training_progress_html(
            total_epochs_parsed, total_epochs_parsed,
            total_steps, total_steps, last_pct, elapsed, last_loss, failed=True,
        )
        _training_state.update({"active": False, "failed": True, "elapsed": elapsed})
        _sync()
        yield (
            _card(failed=True), progress_html, _make_pipeline_html(),
            "", log_text, final_chart, _activity_text(log_text),
        )


# ---------------------------------------------------------------------------
# Step 5 — Chat helpers
# ---------------------------------------------------------------------------

def _load_model_ui() -> Tuple[str, gr.update]:
    global _model, _tokenizer, _chat_session_id
    if not _model_ready():
        return (
            "No trained model found at output_model/. Complete the Train step first.",
            gr.update(interactive=True),
        )
    try:
        _model, _tokenizer = load_model(_OUTPUT_MODEL_DIR)
        _chat_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return "Model loaded. Start chatting!", gr.update(interactive=False, value="Model loaded")
    except Exception as exc:
        return f"Error loading model: {exc}", gr.update(interactive=True)


def _normalize_content(content: Any) -> str:
    """Ensure content is a string (handles Gradio 5.x list-of-parts format)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "\n".join(parts) if parts else ""
    return str(content)


_CHAT_SYSTEM_PROMPT = (
    "You are a concise SQL and Teradata documentation assistant. "
    "Answer questions directly and briefly based on the documentation you were trained on. "
    "Do not repeat the question. Do not include unrelated examples."
)
_CHAT_MAX_HISTORY_TURNS = 3  # keep last N user/assistant turn pairs


def _chat(message: str, history: List) -> str:
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return "Model not loaded. Click 'Load Model' first."

    # Flatten history into message dicts
    history_messages = []
    for entry in history:
        if isinstance(entry, dict):
            history_messages.append({
                "role": entry["role"],
                "content": _normalize_content(entry.get("content")),
            })
        else:
            user_msg, assistant_msg = entry[0], entry[1]
            history_messages.append({"role": "user", "content": _normalize_content(user_msg)})
            if assistant_msg:
                history_messages.append({"role": "assistant", "content": _normalize_content(assistant_msg)})

    # Cap to last N complete turns to avoid context overflow
    # Each turn = 2 messages (user + assistant), so keep last 2*N messages
    if len(history_messages) > _CHAT_MAX_HISTORY_TURNS * 2:
        history_messages = history_messages[-(_CHAT_MAX_HISTORY_TURNS * 2):]

    msg_text = _normalize_content(message["content"] if isinstance(message, dict) else message)

    # Retrieve relevant knowledge library entries and inject into system prompt
    try:
        kb_context = _get_retriever().get_context(msg_text, max_entries=2)
    except Exception:
        kb_context = ""

    system_content = _CHAT_SYSTEM_PROMPT
    if kb_context:
        system_content = _CHAT_SYSTEM_PROMPT + "\n\n" + kb_context

    # Build final message list: system prompt (+ context) + trimmed history + current user message
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": msg_text})

    answer = generate_response(_model, _tokenizer, messages)

    # Log every interaction to conversation memory (fire-and-forget — never block the response)
    try:
        log_interaction(
            _MEMORY_DIR,
            question=msg_text,
            answer=answer,
            session_id=_chat_session_id,
            model_name=_active_model_name(),
            kb_context_used=bool(kb_context),
        )
    except Exception:
        pass

    return answer


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    device_label = _get_device_label()

    # Reflect already-completed steps on startup
    uploaded = _DATA_DIR.exists() and (
        any(_DATA_DIR.glob("*.pdf")) or any(_DATA_DIR.glob("*.csv"))
    )
    train_jsonl = _find_train_jsonl()
    _pipeline_status["upload"]  = "complete" if uploaded else "pending"
    _pipeline_status["extract"] = "complete" if train_jsonl else "pending"
    _pipeline_status["approve"] = "complete" if train_jsonl else "pending"
    _pipeline_status["train"]   = "complete" if _model_ready() else "pending"

    with gr.Blocks(title="SLM Training Demo") as app:
        gr.Markdown(
            f"# SLM Training Demo\n"
            f"Upload documents → extract Q&A training data → train a small model → chat with it.\n\n"
            f"**Runtime device:** `{device_label}`"
        )
        pipeline_status = gr.HTML(value=_make_pipeline_html())

        with gr.Tabs():

            # ── Tab 1 · Upload ─────────────────────────────────────────────
            with gr.Tab("1 · Upload"):
                gr.Markdown(
                    "Upload your source files. **PDFs** are parsed for text content; "
                    "**CSVs** must have `question,answer` columns (or a single `text` column)."
                )
                file_upload = gr.File(
                    label="Upload files (PDF and/or CSV)",
                    file_count="multiple",
                    file_types=[".pdf", ".csv"],
                )
                upload_btn = gr.Button("Upload Files", variant="primary")
                upload_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    value=(
                        f"Previously uploaded files found in {_DATA_DIR}. "
                        "You can re-upload to replace them or proceed to Step 2."
                        if uploaded else "No files uploaded yet."
                    ),
                )
                uploaded_list = gr.HTML(value=_make_uploaded_files_html())

                gr.Markdown("---")
                gr.Markdown("### Remove Uploaded File")
                with gr.Row():
                    with gr.Column(scale=3):
                        remove_file_dd = gr.Dropdown(
                            label="Select file to remove",
                            choices=_upload_file_choices(),
                            value=None,
                        )
                    with gr.Column(scale=1, min_width=160):
                        remove_file_btn = gr.Button("Remove File", variant="stop")

                upload_btn.click(
                    fn=_save_uploads,
                    inputs=[file_upload],
                    outputs=[upload_status, uploaded_list, pipeline_status, remove_file_dd],
                )
                remove_file_btn.click(
                    fn=_remove_file,
                    inputs=[remove_file_dd],
                    outputs=[upload_status, uploaded_list, pipeline_status, remove_file_dd],
                )

            # ── Tab 2 · Extract ────────────────────────────────────────────
            with gr.Tab("2 · Extract Training Data"):
                gr.Markdown(
                    "Configure extraction settings and click **Extract Training Data**. "
                    "Review the generated Q&A pairs, then click **✓ Approve** to proceed."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        manual_mode = gr.Checkbox(
                            label="Manual / documentation mode",
                            value=True,
                            info=(
                                "Best for SQL manuals and technical docs. "
                                "Filters TOC, headers/footers, and index pages; "
                                "extracts section-aware Q&A with typed questions."
                            ),
                        )
                        no_multiturn = gr.Checkbox(
                            label="Skip multi-turn conversations",
                            value=False,
                            info="When unchecked, generates multi-turn ShareGPT conversations in addition to single-turn pairs.",
                        )
                        include_memory = gr.Checkbox(
                            label="Include approved conversation memory",
                            value=True,
                            info=(
                                "Include approved interactions from Tab 7 · Memory in this extraction run. "
                                "Closes the continuous learning loop."
                            ),
                        )
                    with gr.Column(scale=1):
                        chunk_size = gr.Slider(
                            200, 2000, value=800, step=50, label="Chunk size (chars)"
                        )
                        chunk_overlap = gr.Slider(
                            0, 400, value=150, step=25, label="Chunk overlap"
                        )
                    with gr.Column(scale=1):
                        val_ratio = gr.Slider(
                            0.05, 0.4, value=0.15, step=0.05, label="Validation split ratio"
                        )
                        fmt_choice = gr.Radio(
                            ["sharegpt", "alpaca", "both"],
                            value="sharegpt",
                            label="Output format",
                        )

                extract_btn = gr.Button("Extract Training Data", variant="primary")
                extract_progress = gr.HTML(value="")
                extract_activity = gr.Textbox(
                    label="Processing Status",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                    placeholder="Per-file extraction progress will appear here…",
                )
                with gr.Accordion("Developer log", open=False):
                    extract_log = gr.Textbox(
                        label="Raw log", lines=10, max_lines=20, interactive=False,
                    )

                gr.Markdown("---")
                gr.Markdown("### Review Extracted Q&A Pairs")
                qa_review = gr.HTML(
                    value=(
                        _make_qa_review_html()
                        if train_jsonl else
                        '<p style="color:#888;font-family:monospace;font-size:13px;">'
                        'No training data yet. Click <b>Extract Training Data</b> above.</p>'
                    )
                )

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=2):
                        approve_btn = gr.Button("✓ Approve Training Data", variant="primary")
                    with gr.Column(scale=3):
                        approve_status = gr.Textbox(
                            label="Approval status",
                            lines=1,
                            interactive=False,
                            value=(
                                "Training data approved (previously extracted data found)."
                                if train_jsonl else
                                "Run extraction first, then click Approve to proceed to training."
                            ),
                        )

                gr.Markdown("---")
                gr.Markdown(
                    "**Save Dataset Snapshot** — download the JSONL files as a zip "
                    "to reuse later without re-extracting."
                )
                with gr.Row():
                    snap_btn = gr.Button("Save Dataset Snapshot", variant="secondary")
                    snap_status = gr.Textbox(
                        label="Snapshot status", lines=1, interactive=False, value=""
                    )
                snap_file = gr.File(label="Download snapshot", visible=False)

                def _do_save_snapshot():
                    p, msg = _save_dataset_snapshot()
                    if p:
                        return msg, gr.update(value=str(p), visible=True)
                    return msg, gr.update(visible=False)

                extract_btn.click(
                    fn=_run_extraction,
                    inputs=[chunk_size, chunk_overlap, val_ratio, fmt_choice, manual_mode, no_multiturn, include_memory],
                    outputs=[extract_log, extract_progress, pipeline_status, qa_review, extract_activity],
                )
                approve_btn.click(
                    fn=_approve_training_data,
                    inputs=[],
                    outputs=[approve_status, pipeline_status],
                )
                snap_btn.click(
                    fn=_do_save_snapshot,
                    inputs=[],
                    outputs=[snap_status, snap_file],
                )

            # ── Tab 3 · Train ──────────────────────────────────────────────
            with gr.Tab("3 · Train"):
                gr.Markdown(
                    "Configure fine-tuning parameters and click **Start Training**. "
                    "Logs stream live. Complete Steps 1 and 2 first.\n\n"
                    + (
                        "✓ Training data approved — ready to train."
                        if _pipeline_status["approve"] == "complete" else
                        "⚠ Complete Step 2 (Extract & Approve) before training."
                    )
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
                        save_steps = gr.Slider(
                            10, 200, value=50, step=10,
                            label="Checkpoint every N steps",
                            info="Lower = safer against interruption, higher = faster.",
                        )

                with gr.Accordion("Load saved dataset / Resume training", open=False):
                    gr.Markdown(
                        "**Load Dataset** — upload a snapshot zip (from Step 2) or individual `.jsonl` "
                        "files to skip extraction.\n\n"
                        "**Resume** — continue from the last saved checkpoint instead of starting over."
                    )
                    with gr.Row():
                        dataset_upload = gr.File(
                            label="Upload dataset snapshot (.zip or .jsonl)",
                            file_types=[".zip", ".jsonl"],
                        )
                    with gr.Row():
                        load_dataset_btn = gr.Button("Load Dataset", variant="secondary")
                        load_dataset_status = gr.Textbox(
                            label="Load status", lines=1, interactive=False, value=""
                        )
                    resume_ckpt = gr.Checkbox(
                        label="Resume from last checkpoint (if available)", value=False,
                    )

                    def _do_load_dataset(f):
                        if not f:
                            return "No file selected."
                        return _load_dataset_snapshot(f)

                    load_dataset_btn.click(
                        fn=_do_load_dataset,
                        inputs=[dataset_upload],
                        outputs=[load_dataset_status],
                    )

                train_btn = gr.Button("Start Training", variant="primary")
                train_status_card = gr.HTML(value="")
                train_progress = gr.HTML(value="")
                train_activity = gr.Textbox(
                    label="Training Progress",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    placeholder="Epoch / step updates will appear here once training starts…",
                )
                train_loss_plot = gr.LinePlot(
                    value=pd.DataFrame({"step": [], "loss": [], "metric": []}),
                    x="step", y="loss", color="metric",
                    title="Training Loss", y_title="Loss", x_title="Step",
                    height=220, visible=False,
                )
                train_quality = gr.HTML(value="")
                with gr.Accordion("Developer log", open=False):
                    train_log_raw = gr.Textbox(
                        label="Raw training log", lines=12, max_lines=30, interactive=False,
                    )

                train_btn.click(
                    fn=_run_training,
                    inputs=[model_name, epochs, batch_size, lr, max_seq_len, resume_ckpt, save_steps],
                    outputs=[
                        train_status_card, train_progress, pipeline_status,
                        train_quality, train_log_raw, train_loss_plot, train_activity,
                    ],
                )

            # ── Tab 4 · Model Manager ──────────────────────────────────────
            with gr.Tab("4 · Model Manager"):
                gr.Markdown(
                    "Select a saved model, then use **Set as Active** to load it for chat "
                    "or **Delete** to remove it. Save the current trained model with a name "
                    "to keep it for later."
                )
                _init_arts = _list_all_artifacts()

                artifacts_html = gr.HTML(value=_make_artifacts_html(_init_arts))

                with gr.Row():
                    artifact_dropdown = gr.Dropdown(
                        label="Select model",
                        choices=_artifact_choices(_init_arts),
                        value=None,
                        scale=3,
                    )
                    refresh_btn = gr.Button("Refresh", variant="secondary", min_width=100, scale=1)

                manage_status = gr.Textbox(
                    label="Status", interactive=False, lines=1, value=""
                )

                with gr.Row():
                    set_active_btn = gr.Button("Set as Active Model for Chat", variant="primary", scale=2)
                    delete_btn = gr.Button("Delete Selected Model", variant="stop", scale=1)

                gr.Markdown("---")
                gr.Markdown("### Save Current Output Model")
                with gr.Row():
                    with gr.Column(scale=3):
                        save_name = gr.Textbox(
                            label="Save as name",
                            placeholder="e.g. td17-analytic-v1",
                            max_lines=1,
                        )
                    with gr.Column(scale=1, min_width=140):
                        save_btn = gr.Button("Save Model", variant="primary")

                with gr.Accordion("Rename Selected Model", open=False):
                    with gr.Row():
                        with gr.Column(scale=3):
                            rename_input = gr.Textbox(
                                label="New name",
                                placeholder="e.g. td17-analytic-v2",
                                max_lines=1,
                            )
                        with gr.Column(scale=1, min_width=140):
                            rename_btn = gr.Button("Rename", variant="secondary")

                save_btn.click(
                    fn=_save_current_model,
                    inputs=[save_name],
                    outputs=[manage_status, artifacts_html, artifact_dropdown],
                )
                rename_btn.click(
                    fn=_rename_artifact,
                    inputs=[artifact_dropdown, rename_input],
                    outputs=[manage_status, artifacts_html, artifact_dropdown],
                )
                set_active_btn.click(
                    fn=_set_active_model,
                    inputs=[artifact_dropdown],
                    outputs=[manage_status, artifacts_html, artifact_dropdown],
                )
                delete_btn.click(
                    fn=_delete_artifact,
                    inputs=[artifact_dropdown],
                    outputs=[manage_status, artifacts_html, artifact_dropdown],
                )
                refresh_btn.click(
                    fn=_refresh_artifacts,
                    inputs=[],
                    outputs=[artifacts_html, artifact_dropdown],
                )

            # ── Tab 5 · Chat ───────────────────────────────────────────────
            with gr.Tab("5 · Chat"):
                gr.Markdown(
                    "Once training is complete, load the model and start asking questions. "
                    "Use Step 4 · Model Manager to switch between saved models."
                )
                load_btn = gr.Button(
                    "Load Model" if not _model_ready() else "Load Model (ready)",
                    variant="primary",
                )
                load_status = gr.Textbox(
                    label="Status",
                    value=(
                        "Model found — click Load Model to begin."
                        if _model_ready() else
                        "No trained model yet. Complete Steps 1–3 first."
                    ),
                    interactive=False,
                    lines=1,
                )
                _chat_kwargs: dict = dict(
                    fn=_chat,
                    examples=[
                        "What is this document about?",
                        "Show me an example SQL query.",
                        "What is the syntax for RANK?",
                        "What does CSUM do?",
                        "What are the arguments to QUANTILE?",
                    ],
                )
                try:
                    chat_interface = gr.ChatInterface(type="messages", **_chat_kwargs)  # noqa: F841
                except TypeError:
                    chat_interface = gr.ChatInterface(**_chat_kwargs)  # noqa: F841

                load_btn.click(
                    fn=_load_model_ui,
                    inputs=[],
                    outputs=[load_status, load_btn],
                )

            # ── Tab 6 · Knowledge Library ───────────────────────────────────
            with gr.Tab("6 · Knowledge Library"):
                gr.Markdown(
                    "**Teach the model what you know** — no YAML or coding required.\n\n"
                    "Fill in the form below in plain English. The system generates training Q&A pairs "
                    "from your answers and saves them to a persistent library. "
                    "You don't need all the answers at once — add what you know now, "
                    "come back and add more later. Every saved entry is automatically "
                    "included the next time you run **Step 2 · Extract Training Data**."
                )

                with gr.Accordion("Add / Edit a Pattern", open=True):
                    with gr.Row():
                        with gr.Column(scale=2):
                            kb_title = gr.Textbox(
                                label="Name / Title *",
                                placeholder="e.g. nPath, CSUM, Sessionize, QUANTILE",
                            )
                            kb_description = gr.Textbox(
                                label="What does it do? *",
                                placeholder=(
                                    "Explain in plain English what this function or feature does, "
                                    "what data it works on, and what result it produces..."
                                ),
                                lines=4,
                            )
                            kb_use_cases = gr.Textbox(
                                label="When would you use this? (one per line)",
                                placeholder="User journey analysis\nFunnel analysis\nChurn prediction",
                                lines=4,
                            )
                            kb_params = gr.Textbox(
                                label="Inputs / Parameters (name: description (example))",
                                placeholder=(
                                    "partition_columns: Columns to group by, e.g. user_id (user_id)\n"
                                    "order_columns: Columns to sort by, e.g. timestamp (event_ts)\n"
                                    "pattern: Sequence to match (A.B+.C)"
                                ),
                                lines=5,
                            )
                            kb_category = gr.Textbox(
                                label="Category (optional)",
                                placeholder="analytics, data_quality, ml, timeseries, text…",
                            )
                        with gr.Column(scale=2):
                            kb_sql = gr.Textbox(
                                label="SQL example (paste real SQL — concrete values are fine)",
                                placeholder=(
                                    "SELECT * FROM nPath (\n"
                                    "  ON clickstream PARTITION BY user_id ORDER BY ts\n"
                                    "  USING\n"
                                    "    SYMBOLS (event IN ('LOGIN') AS A, event IN ('BUY') AS B)\n"
                                    "    PATTERN ('A.B')\n"
                                    "    MODE (NONOVERLAPPING)\n"
                                    "    RESULT (ACCUMULATE (event OF ANY (A,B)) AS path)\n"
                                    ")"
                                ),
                                lines=8,
                            )
                            kb_sql_desc = gr.Textbox(
                                label="What does this SQL example do?",
                                placeholder="Find all users who logged in then made a purchase",
                            )
                            kb_output = gr.Textbox(
                                label="What does the result look like? (sample rows or description)",
                                placeholder=(
                                    "path                    | count\n"
                                    "LOGIN|BUY               | 1234\n"
                                    "LOGIN|BROWSE|BUY        |  876"
                                ),
                                lines=4,
                            )
                    kb_errors = gr.Textbox(
                        label="Common errors or gotchas (Problem: solution — one per line)",
                        placeholder=(
                            "Spaces in pattern string: Remove all spaces, use A.B.C not 'A B C'\n"
                            "Reserved keyword as symbol name: Use Exited not Exit, Counted not Count"
                        ),
                        lines=3,
                    )
                    kb_bp = gr.Textbox(
                        label="Tips and best practices (free text — write what you know)",
                        placeholder="Always run EXPLAIN before executing. Start with a simple pattern like A.B.C...",
                        lines=3,
                    )
                    with gr.Row():
                        kb_preview_btn = gr.Button("Preview Q&A pairs", variant="secondary")
                        kb_save_btn = gr.Button("Save to Library", variant="primary")

                kb_status = gr.Textbox(label="Status", interactive=False, lines=2)
                kb_preview_out = gr.HTML(
                    value='<div style="color:#888;padding:8px;">Click Preview to see generated Q&A pairs.</div>'
                )

                gr.Markdown("---")
                gr.Markdown("### Knowledge Library")
                kb_library_html = gr.HTML(value=_make_library_html())
                kb_refresh_btn = gr.Button("Refresh Library", variant="secondary", size="sm")

                with gr.Accordion("Delete an entry", open=False):
                    with gr.Row():
                        kb_delete_slug = gr.Textbox(
                            label="Slug (from the table above)",
                            placeholder="e.g. npath_sequence_analysis",
                            scale=3,
                        )
                        kb_delete_btn = gr.Button("Delete", variant="stop", scale=1)

                # Wire up events
                _kb_inputs = [
                    kb_title, kb_description, kb_use_cases, kb_params,
                    kb_sql, kb_sql_desc, kb_output, kb_errors, kb_bp, kb_category,
                ]
                kb_preview_btn.click(
                    fn=_kb_preview,
                    inputs=_kb_inputs,
                    outputs=[kb_preview_out],
                )
                kb_save_btn.click(
                    fn=_kb_save,
                    inputs=_kb_inputs,
                    outputs=[kb_status, kb_library_html],
                )
                kb_refresh_btn.click(
                    fn=lambda: _make_library_html(),
                    inputs=[],
                    outputs=[kb_library_html],
                )
                kb_delete_btn.click(
                    fn=_kb_delete,
                    inputs=[kb_delete_slug],
                    outputs=[kb_status, kb_library_html],
                )

            # ── Tab 7 · Memory ───────────────────────────────────────────────
            with gr.Tab("7 · Memory"):
                gr.Markdown(
                    "**Conversation Memory** — every chat interaction is logged here automatically.\n\n"
                    "Review answers, **Approve** the good ones and **Reject** the bad ones. "
                    "Approved interactions are exported as training data and fed into the next "
                    "fine-tuning run, closing the continuous-learning loop. "
                    "Frequently-asked questions surface as candidate patterns for the Knowledge Library."
                )

                mem_stats_html = gr.HTML(value=_make_memory_stats_html())

                gr.Markdown("---")
                gr.Markdown("### Review Interactions")
                with gr.Row():
                    with gr.Column(scale=3):
                        mem_dd = gr.Dropdown(
                            label="Select an interaction to review",
                            choices=_memory_interaction_choices(),
                            value=None,
                        )
                    with gr.Column(scale=1, min_width=120):
                        mem_refresh_btn = gr.Button("Refresh", variant="secondary")

                mem_current_id = gr.Textbox(visible=False, value="")
                mem_detail_html = gr.HTML(value="")
                mem_action_status = gr.Textbox(label="Status", lines=1, interactive=False, value="")

                with gr.Row():
                    mem_approve_btn = gr.Button("✓ Approve — Good answer", variant="primary", scale=2)
                    mem_reject_btn  = gr.Button("✗ Reject — Bad answer",  variant="stop",    scale=1)

                gr.Markdown("---")
                gr.Markdown(
                    "### Export to Training Data\n"
                    "Exports all **Approved** interactions as ShareGPT JSONL. "
                    "Check **Include conversation memory** in Step 3 · Train to include them in the next run."
                )
                with gr.Row():
                    mem_export_btn = gr.Button("Export Approved to Training Data", variant="primary")
                    mem_export_status = gr.Textbox(label="Status", lines=1, interactive=False, value="")
                mem_export_file = gr.File(label="Download exported file", visible=False)

                gr.Markdown("---")
                gr.Markdown("### Frequently-Asked Questions")
                gr.Markdown(
                    "Questions asked 2+ times — these are strong candidates to add to the "
                    "Knowledge Library as structured patterns."
                )
                mem_freq_html = gr.HTML(value=_make_frequent_questions_html())

                # ── Event wiring ──────────────────────────────────────────
                mem_dd.change(
                    fn=_memory_select,
                    inputs=[mem_dd],
                    outputs=[mem_current_id, mem_detail_html],
                )
                mem_refresh_btn.click(
                    fn=_memory_refresh,
                    inputs=[],
                    outputs=[mem_stats_html, mem_freq_html, mem_dd],
                )
                mem_approve_btn.click(
                    fn=_memory_approve,
                    inputs=[mem_current_id],
                    outputs=[mem_action_status, mem_stats_html, mem_detail_html, mem_dd],
                )
                mem_reject_btn.click(
                    fn=_memory_reject,
                    inputs=[mem_current_id],
                    outputs=[mem_action_status, mem_stats_html, mem_detail_html, mem_dd],
                )
                mem_export_btn.click(
                    fn=_memory_export,
                    inputs=[],
                    outputs=[mem_export_status, mem_export_file],
                )

        # ── Auto-reconnect: restore training UI on page load / browser crash ──
        _reconnect_outputs = [
            train_status_card, train_progress, pipeline_status,
            train_quality, train_log_raw, train_loss_plot, train_activity,
        ]
        app.load(fn=_rebuild_training_ui, outputs=_reconnect_outputs)
        try:
            gr.Timer(value=2).tick(fn=_rebuild_training_ui, outputs=_reconnect_outputs)
        except AttributeError:
            pass

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="End-to-end SLM training and demo UI")
    p.add_argument(
        "--model-dir", type=Path, default=None,
        help="Trained model directory (default: <root>/output_model or /app/output_model in Docker)",
    )
    p.add_argument(
        "--training-data-dir", type=Path, default=None,
        help="Training data directory (default: <root>/training_data or /app/training_data in Docker)",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Raw uploaded data directory (default: <root>/data or /app/data in Docker)",
    )
    p.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    p.add_argument(
        "--host", type=str,
        default=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        help="Bind address. Default: 0.0.0.0 (all interfaces). "
             "Set GRADIO_HOST=127.0.0.1 to restrict to localhost only.",
    )
    p.add_argument("--share", action="store_true", default=False)
    args = p.parse_args()

    global _DATA_DIR, _TRAINING_DATA_DIR, _OUTPUT_MODEL_DIR, _SAVED_MODELS_DIR, _LIBRARY_DIR
    if args.data_dir:
        _DATA_DIR = args.data_dir
    if args.training_data_dir:
        _TRAINING_DATA_DIR = args.training_data_dir
    # Library dir alongside training data dir
    _LIBRARY_DIR = _TRAINING_DATA_DIR.parent / "knowledge_library"

    model_dir = args.model_dir or Path(os.environ.get("MODEL_DIR", "")) or None
    if model_dir and str(model_dir) not in ("", "."):
        _OUTPUT_MODEL_DIR = model_dir
        _SAVED_MODELS_DIR = model_dir.parent / "saved_models"

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        allowed_paths=[str(_TRAINING_DATA_DIR), str(_OUTPUT_MODEL_DIR), str(_SAVED_MODELS_DIR)],
    )


if __name__ == "__main__":
    main()