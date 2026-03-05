"""
End-to-end Gradio UI: Prepare Data → Train → Chat.
All three stages run inside the same container; training streams live logs.

Usage:
  python run_gradio_ui.py --host 0.0.0.0          # Docker (all defaults)
  python run_gradio_ui.py --model-dir output_model  # local override
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

# ---------------------------------------------------------------------------
# Default paths (Docker layout; overridable via CLI)
# ---------------------------------------------------------------------------
_DATA_DIR = Path("/app/data")
_TRAINING_DATA_DIR = Path("/app/training_data")
_OUTPUT_MODEL_DIR = Path("/app/output_model")
_SAVED_MODELS_DIR = Path("/app/saved_models")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None

# Pipeline completion tracking (updated by each tab's generator)
_pipeline_status: dict = {"data": "pending", "train": "pending"}

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
    """Zip all JSONL files in training_data/ into a snapshot archive.
    Returns (zip_path_or_None, status_message).
    """
    files = sorted(_TRAINING_DATA_DIR.glob("*.jsonl"))
    if not files:
        return None, "No training data found — run Prepare Data first."
    snap_path = _TRAINING_DATA_DIR / "dataset_snapshot.zip"
    with zipfile.ZipFile(snap_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    size_kb = snap_path.stat().st_size / 1024
    return snap_path, (
        f"Snapshot ready: {len(files)} JSONL files, {size_kb:.1f} KB — "
        "download it using the file widget below."
    )


def _load_dataset_snapshot(upload_path: str) -> str:
    """Extract an uploaded zip (or copy a single JSONL) into training_data/.
    Returns a status message.
    """
    src = Path(upload_path)
    _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            names = zf.namelist()
            zf.extractall(_TRAINING_DATA_DIR)
        return (
            f"Loaded snapshot: extracted {len(names)} file(s) to {_TRAINING_DATA_DIR}. "
            "You can now start training without running Prepare Data."
        )
    elif src.suffix.lower() == ".jsonl":
        dest = _TRAINING_DATA_DIR / src.name
        shutil.copy2(src, dest)
        return (
            f"Loaded {src.name} → {_TRAINING_DATA_DIR}. "
            "You can now start training without running Prepare Data."
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
    # Detect Apple Silicon even when running inside Docker (Linux/aarch64)
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
    """Return the last *n* meaningful training-progress lines from raw log.
    Filters out transformers/tqdm noise so non-technical users see only what matters.
    """
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
    """Return meaningful data-prep progress lines, filtering out command/path noise."""
    keep = []
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (
            line.startswith("[")          # [1/16] per-file progress
            or line.startswith("Found ")
            or line.startswith("Loading CSV")
            or line.startswith("Total:")
            or line.startswith("Alpaca:")
            or line.startswith("ShareGPT:")
            or "complete" in line.lower()
            or "Q&A pairs" in line
            or "Data preparation" in line
        ):
            keep.append(line)
    return "\n".join(keep[-15:])


def _rebuild_training_ui():
    """Reconstruct the training tab UI from _training_state.

    Called on page load and by the auto-refresh timer so the UI automatically
    recovers after a browser crash or tab reload while training is running.
    Returns a 7-tuple matching the outputs list of train_btn.click().
    """
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

    pipe_html = _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"])
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


# Ordered phases shown in the training status breadcrumb
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
    """User-friendly training status card: phase breadcrumb + message + warnings/errors."""
    phase_keys = [p for p, _ in _TRAIN_PHASES]
    phase_idx = phase_keys.index(phase) if phase in phase_keys else 0

    # --- Breadcrumb ---
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

    # --- Current message ---
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

    # --- Warnings ---
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


def _make_pipeline_html(data_status: str = "pending", train_status: str = "pending") -> str:
    _CFG = {
        "pending":  ("○", "#adb5bd", "#6c757d"),
        "running":  ("↻", "#4a90d9", "#4a90d9"),
        "complete": ("✓", "#28a745", "#28a745"),
        "warning":  ("⚠", "#ffc107", "#856404"),
        "failed":   ("✗", "#dc3545", "#dc3545"),
    }
    chat_status = "complete" if train_status == "complete" else "pending"
    steps = [
        ("1", "Prepare Data", data_status),
        ("2", "Train", train_status),
        ("3", "Chat", chat_status),
    ]
    parts = []
    for num, name, status in steps:
        icon, bg, tc = _CFG.get(status, _CFG["pending"])
        parts.append(
            f'<div style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="width:22px;height:22px;border-radius:50%;background:{bg};color:white;'
            f'display:inline-flex;align-items:center;justify-content:center;font-size:11px;'
            f'font-weight:bold;flex-shrink:0;">{icon}</span>'
            f'<span style="color:#333;"><b>Step&nbsp;{num}</b>&nbsp;{name}</span>'
            f'&nbsp;<span style="color:{tc};font-weight:bold;font-size:12px;">[{status.capitalize()}]</span>'
            f'</div>'
        )
    sep = '&nbsp;<span style="color:#adb5bd;font-size:18px;">&#8594;</span>&nbsp;'
    return (
        f'<div style="display:flex;align-items:center;gap:4px;padding:10px 16px;'
        f'background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;'
        f'font-family:monospace;font-size:13px;flex-wrap:wrap;margin-bottom:6px;">'
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

    # Pick 3 indices spread across the middle third of the file
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


# ---------------------------------------------------------------------------
# Model artifact management (Tab 4)
# ---------------------------------------------------------------------------

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
    # Current training output
    if _OUTPUT_MODEL_DIR.exists() and (_OUTPUT_MODEL_DIR / "config.json").exists():
        base_model = ""
        try:
            cfg = json.loads((_OUTPUT_MODEL_DIR / "config.json").read_text())
            base_model = cfg.get("_name_or_path", "")
        except Exception:
            pass
        artifacts.append({
            "name": "output_model",
            "path": _OUTPUT_MODEL_DIR,
            "size": _dir_size_str(_OUTPUT_MODEL_DIR),
            "saved_at": datetime.fromtimestamp(_OUTPUT_MODEL_DIR.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "base_model": base_model,
            "is_current": True,
        })
    # Named saved models
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
                "is_current": False,
            })
    return artifacts


def _make_artifacts_html(artifacts: list) -> str:
    if not artifacts:
        return (
            '<div style="padding:16px;background:#f8f9fa;border:1px solid #dee2e6;'
            'border-radius:8px;font-family:monospace;font-size:13px;color:#6c757d;">'
            'No model artifacts found. Train a model and click <b>Save Model</b> to preserve it.'
            '</div>'
        )
    cards = []
    for a in artifacts:
        badge = (
            '<span style="background:#6c757d;color:white;font-size:10px;padding:1px 8px;'
            'border-radius:3px;margin-left:8px;">Current Output</span>'
            if a["is_current"] else ""
        )
        base_html = (
            f'&nbsp;&middot;&nbsp;Base:&nbsp;<span style="color:#444;">'
            f'{_html.escape(a["base_model"])}</span>'
            if a.get("base_model") else ""
        )
        cards.append(
            f'<div style="border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;'
            f'margin-bottom:8px;background:{"#fff8e1" if a["is_current"] else "white"};">'
            f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
            f'<span style="font-family:monospace;font-size:14px;font-weight:bold;color:#222;">'
            f'{_html.escape(a["name"])}</span>{badge}</div>'
            f'<div style="font-family:monospace;font-size:12px;color:#666;">'
            f'Size:&nbsp;<b>{a["size"]}</b>&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
            f'Saved:&nbsp;{a["saved_at"]}{base_html}</div></div>'
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
        (dest / "_meta.json").write_text(json.dumps({
            "name": name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "base_model": base_model,
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
    Yields None roughly every *heartbeat* seconds when the process is silent, so
    callers can update an elapsed-time display even when no output is produced
    (e.g. while processing large PDFs).
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
        q.put(None)  # sentinel

    threading.Thread(target=_reader, daemon=True).start()
    while True:
        try:
            item = q.get(timeout=heartbeat)
            if item is None:
                break
            yield item
        except queue.Empty:
            yield None  # heartbeat tick — no new log line


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
# Tab 1 — Data Preparation
# ---------------------------------------------------------------------------

def _run_data_prep(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
    val_ratio: float,
    fmt: str,
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    start_time = time.time()
    _pipeline_status["data"] = "running"
    log_text = "Starting data preparation...\n"
    pipe_html = _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"])
    yield log_text, _make_elapsed_html(0.0), pipe_html, "", _data_activity_text(log_text)

    if not uploaded_files:
        log_text += "ERROR: No files uploaded. Please upload at least one PDF or CSV.\n"
        _pipeline_status["data"] = "failed"
        yield log_text, _make_elapsed_html(time.time() - start_time, failed=True), \
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", \
            _data_activity_text(log_text)
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
    log_text += f"Copied {len(pdfs)} PDF(s) and {len(csvs)} CSV file(s) to {_DATA_DIR}\n"
    yield log_text, _make_elapsed_html(time.time() - start_time), \
        _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", \
        _data_activity_text(log_text)

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

    log_text += f"Running: {' '.join(cmd)}\n\n"
    yield log_text, _make_elapsed_html(time.time() - start_time), \
        _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", \
        _data_activity_text(log_text)

    proc = _popen(cmd)
    for line in _iter_lines(proc):
        if line:
            log_text += line
        yield log_text, _make_elapsed_html(time.time() - start_time), \
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", \
            _data_activity_text(log_text)

    elapsed = time.time() - start_time
    train_jsonl = _TRAINING_DATA_DIR / "train_sharegpt.jsonl"
    if proc.returncode == 0:
        log_text += "\nData preparation complete.\n"
        files = sorted(_TRAINING_DATA_DIR.glob("*.jsonl"))
        if files:
            log_text += "Output files:\n" + "\n".join(f"  {f.name}" for f in files) + "\n"
        # Warn if the training split is very small
        n_prep = sum(1 for _ in train_jsonl.open()) if train_jsonl.exists() else 0
        if n_prep < 20:
            _pipeline_status["data"] = "warning"
            log_text += f"WARNING: Only {n_prep} training examples generated — quality may be poor.\n"
        else:
            _pipeline_status["data"] = "complete"
        sample_html = _make_data_sample_html(train_jsonl)
        yield log_text, _make_elapsed_html(elapsed, done=(_pipeline_status["data"] == "complete")), \
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), sample_html, \
            _data_activity_text(log_text)
    else:
        log_text += f"\nData preparation FAILED (exit code {proc.returncode}).\n"
        _pipeline_status["data"] = "failed"
        yield log_text, _make_elapsed_html(elapsed, failed=True), \
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", \
            _data_activity_text(log_text)


# ---------------------------------------------------------------------------
# Tab 2 — Training
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
    """Yields (status_card, progress_bar, pipeline, quality, raw_log) on every update."""
    device_label = _get_device_label()
    use_unsloth = _unsloth_available()
    script = "train.finetune_unsloth" if use_unsloth else "train.finetune_cpu"

    _pipeline_status["train"] = "running"

    # ── User-facing status state ──────────────────────────────────────────
    phase = "loading"
    current_msg = f"Loading model {model_name}…"
    warnings: List[str] = []
    error: Optional[str] = None

    if "CPU" in device_label and not use_unsloth:
        warnings.append(
            "Training on CPU is very slow. "
            "For a 1.1B model expect several minutes per step."
        )

    # ── Training progress state ───────────────────────────────────────────
    current_epoch = 0
    total_epochs_parsed = epochs
    global_step = 0
    total_steps = 0
    last_pct = 0.0
    initial_loss: Optional[float] = None
    last_loss: Optional[float] = None
    last_eval_loss: Optional[float] = None
    n_examples = 0
    loss_history: List[dict] = []  # [{step, loss, metric}] for live chart

    # ── Initialise persistent reconnect state ────────────────────────────
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
        """Snapshot current local training variables into _training_state."""
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
    progress_html = ""   # hidden until training phase starts
    pipe_html = _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"])

    def _card(done: bool = False, failed: bool = False) -> str:
        return _make_train_status_html(phase, current_msg, warnings, error, done, failed)

    _no_chart = gr.update(visible=False)

    yield _card(), progress_html, pipe_html, "", log_text, _no_chart, _activity_text(log_text)

    # ── Validate files ────────────────────────────────────────────────────
    train_file = _TRAINING_DATA_DIR / "train_sharegpt.jsonl"
    val_file = _TRAINING_DATA_DIR / "val_sharegpt.jsonl"
    if not train_file.exists():
        error = "Training data not found. Complete the 'Prepare Data' step first."
        _pipeline_status["train"] = "failed"
        log_text += f"ERROR: {error}\n"
        yield _card(failed=True), "", \
            _make_pipeline_html(_pipeline_status["data"], "failed"), "", log_text, _no_chart, ""
        return
    if not val_file.exists():
        error = "Validation data not found. Complete the 'Prepare Data' step first."
        _pipeline_status["train"] = "failed"
        log_text += f"ERROR: {error}\n"
        yield _card(failed=True), "", \
            _make_pipeline_html(_pipeline_status["data"], "failed"), "", log_text, _no_chart, ""
        return

    _OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Checkpoint resume
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
    yield _card(), progress_html, pipe_html, "", log_text, _no_chart, _activity_text(log_text)

    start_time = time.time()
    _training_state["start_time"] = start_time
    proc = _popen(cmd)

    for line in _iter_lines(proc):
        elapsed = time.time() - start_time
        if line:
            log_text += line
            clean = line.strip()

            # ── Phase / message transitions ───────────────────────────────
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

            # ── Warnings from subprocess (skip the CPU one we already added) ──
            if "WARNING:" in clean and "very slow" not in clean:
                w = clean.split("WARNING:", 1)[-1].strip()
                if w and w not in warnings:
                    warnings.append(w)

            # ── Errors / tracebacks ───────────────────────────────────────
            if "Traceback (most recent call last)" in clean:
                if not error:
                    error = "An error occurred — see developer log for details."
            if clean.startswith("Error") and ":" in clean[:40]:
                if not error:
                    error = clean

            # ── Training progress metrics ─────────────────────────────────
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

        # Show training progress bar once we enter the training/saving phase
        if phase in ("training", "saving") or global_step > 0:
            progress_html = _make_training_progress_html(
                max(current_epoch, 1) if global_step > 0 else current_epoch,
                total_epochs_parsed,
                global_step,
                total_steps,
                last_pct,
                elapsed,
                last_loss,
            )

        chart_update = (
            gr.update(value=pd.DataFrame(loss_history), visible=True)
            if loss_history else _no_chart
        )
        _sync()
        yield _card(), progress_html, \
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]), "", log_text, \
            chart_update, _activity_text(log_text)

    elapsed = time.time() - start_time
    final_chart = (
        gr.update(value=pd.DataFrame(loss_history), visible=True)
        if loss_history else _no_chart
    )
    success = proc.returncode == 0
    if success:
        _pipeline_status["train"] = "complete"
        current_msg = "Training complete! Switch to the Chat tab and load your model."
        log_text += "\nTraining complete.\n"
        progress_html = _make_training_progress_html(
            total_epochs_parsed, total_epochs_parsed,
            total_steps, total_steps, 100.0, elapsed, last_loss, done=True,
        )
        quality_html = _make_quality_html(last_loss, last_eval_loss, initial_loss, n_examples)
        _training_state.update({"active": False, "done": True, "elapsed": elapsed})
        _sync()
        yield (
            _card(done=True),
            progress_html,
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]),
            quality_html,
            log_text,
            final_chart,
            _activity_text(log_text),
        )
    else:
        _pipeline_status["train"] = "failed"
        if not error:
            error = f"Training failed (exit code {proc.returncode}). Check the developer log for details."
        log_text += f"\nTraining FAILED (exit code {proc.returncode}).\n"
        progress_html = _make_training_progress_html(
            total_epochs_parsed, total_epochs_parsed,
            total_steps, total_steps, last_pct, elapsed, last_loss, failed=True,
        )
        _training_state.update({"active": False, "failed": True, "elapsed": elapsed})
        _sync()
        yield (
            _card(failed=True),
            progress_html,
            _make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"]),
            "",
            log_text,
            final_chart,
            _activity_text(log_text),
        )


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


def _normalize_content(content: Any) -> str:
    """Ensure content is a string for tokenizer.apply_chat_template (no list concatenation)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gradio 5.x can send content as list of parts e.g. [{"type": "text", "text": "..."}]
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


def _chat(message: str, history: List) -> str:
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return "Model not loaded. Click 'Load Model' first."
    messages = []
    for entry in history:
        if isinstance(entry, dict):
            # Gradio 5.x messages format: {"role": ..., "content": ..., ...}
            messages.append({
                "role": entry["role"],
                "content": _normalize_content(entry.get("content")),
            })
        else:
            # Gradio 4.x tuples format: (user_msg, assistant_msg)
            user_msg, assistant_msg = entry[0], entry[1]
            messages.append({"role": "user", "content": _normalize_content(user_msg)})
            if assistant_msg:
                messages.append({"role": "assistant", "content": _normalize_content(assistant_msg)})
    messages.append({"role": "user", "content": _normalize_content(message)})
    return generate_response(_model, _tokenizer, messages)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    device_label = _get_device_label()

    # Reflect any already-completed steps on startup
    _pipeline_status["data"] = (
        "complete" if (_TRAINING_DATA_DIR / "train_sharegpt.jsonl").exists() else "pending"
    )
    _pipeline_status["train"] = "complete" if _model_ready() else "pending"

    with gr.Blocks(title="SLM Training Demo") as app:
        gr.Markdown(
            f"# SLM Training Demo\n"
            f"Train a small language model on your own data and chat with it — "
            f"all in the browser.\n\n"
            f"**Runtime device:** `{device_label}`"
        )
        pipeline_status = gr.HTML(
            value=_make_pipeline_html(_pipeline_status["data"], _pipeline_status["train"])
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
                prep_progress = gr.HTML(value="")
                prep_activity = gr.Textbox(
                    label="Processing Status",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    placeholder="Per-file progress will appear here once processing starts…",
                )
                with gr.Accordion("Developer log", open=False):
                    prep_log = gr.Textbox(
                        label="Raw log",
                        lines=10,
                        max_lines=20,
                        interactive=False,
                    )
                data_sample = gr.HTML(value="")

                gr.Markdown("---")
                gr.Markdown(
                    "**Save Dataset Snapshot** — Download the prepared JSONL files as a zip. "
                    "Re-upload this zip in the Train tab to skip Prepare Data next time."
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

                snap_btn.click(
                    fn=_do_save_snapshot,
                    inputs=[],
                    outputs=[snap_status, snap_file],
                )
                prep_btn.click(
                    fn=_run_data_prep,
                    inputs=[file_upload, chunk_size, chunk_overlap, val_ratio, fmt_choice],
                    outputs=[prep_log, prep_progress, pipeline_status, data_sample, prep_activity],
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
                        save_steps = gr.Slider(
                            10, 200, value=50, step=10,
                            label="Checkpoint every N steps",
                            info="Save a checkpoint this often. Lower = safer, higher = faster.",
                        )

                with gr.Accordion("Load saved dataset / Resume training", open=False):
                    gr.Markdown(
                        "**Load Dataset** — upload a snapshot zip (saved from Prepare Data tab) "
                        "or individual `.jsonl` files to skip the data preparation step.\n\n"
                        "**Resume** — if training was interrupted, tick this to continue from "
                        "the last saved checkpoint instead of starting over."
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
                        label="Resume from last checkpoint (if available)",
                        value=False,
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
                    x="step",
                    y="loss",
                    color="metric",
                    title="Training Loss",
                    y_title="Loss",
                    x_title="Step",
                    height=220,
                    visible=False,
                )
                train_quality = gr.HTML(value="")
                with gr.Accordion("Developer log", open=False):
                    train_log_raw = gr.Textbox(
                        label="Raw training log",
                        lines=12,
                        max_lines=30,
                        interactive=False,
                    )
                train_btn.click(
                    fn=_run_training,
                    inputs=[model_name, epochs, batch_size, lr, max_seq_len, resume_ckpt, save_steps],
                    outputs=[train_status_card, train_progress, pipeline_status, train_quality, train_log_raw, train_loss_plot, train_activity],
                )

            # ── Tab 3 ──────────────────────────────────────────────────────
            with gr.Tab("3 · Model Manager"):
                gr.Markdown(
                    "Save the current trained model under a name to preserve it across runs, "
                    "then manage or delete artifacts to free disk space."
                )

                gr.Markdown("### Save Current Model")
                with gr.Row():
                    with gr.Column(scale=3):
                        save_name = gr.Textbox(
                            label="Model name",
                            placeholder="e.g. tinyllama-qa-v1",
                            max_lines=1,
                        )
                    with gr.Column(scale=1, min_width=140):
                        save_btn = gr.Button("Save Model", variant="primary")
                save_status = gr.Textbox(
                    label="Save status", interactive=False, lines=1, value=""
                )

                gr.Markdown("### Artifact List")
                _init_arts = _list_all_artifacts()
                artifacts_html = gr.HTML(value=_make_artifacts_html(_init_arts))
                refresh_btn = gr.Button("Refresh List", variant="secondary")

                gr.Markdown("### Delete Artifact")
                gr.Markdown(
                    "Select an artifact from the dropdown and click **Delete Selected** "
                    "to permanently remove it from disk. Deleting `output_model` also "
                    "unloads the model from memory."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        delete_dropdown = gr.Dropdown(
                            label="Select artifact to delete",
                            choices=_artifact_choices(_init_arts),
                            value=None,
                        )
                    with gr.Column(scale=1, min_width=160):
                        delete_btn = gr.Button("Delete Selected", variant="stop")
                delete_status = gr.Textbox(
                    label="Delete status", interactive=False, lines=1, value=""
                )

                save_btn.click(
                    fn=_save_current_model,
                    inputs=[save_name],
                    outputs=[save_status, artifacts_html, delete_dropdown],
                )
                delete_btn.click(
                    fn=_delete_artifact,
                    inputs=[delete_dropdown],
                    outputs=[delete_status, artifacts_html, delete_dropdown],
                )
                refresh_btn.click(
                    fn=_refresh_artifacts,
                    inputs=[],
                    outputs=[artifacts_html, delete_dropdown],
                )

            # ── Tab 4 ──────────────────────────────────────────────────────
            with gr.Tab("4 · Chat"):
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
                _chat_kwargs: dict = dict(
                    fn=_chat,
                    examples=[
                        "What is this document about?",
                        "Summarize the main points.",
                        "How do I get started?",
                    ],
                )
                try:
                    # Gradio 5.x: explicit messages format avoids tuple-unpack errors
                    chat_interface = gr.ChatInterface(type="messages", **_chat_kwargs)  # noqa: F841
                except TypeError:
                    # Gradio 4.x: type parameter not supported
                    chat_interface = gr.ChatInterface(**_chat_kwargs)  # noqa: F841
                load_btn.click(
                    fn=_load_model_ui,
                    inputs=[],
                    outputs=[load_status, load_btn],
                )

        # ── Auto-reconnect: restore training UI on page load / browser crash ──
        # app.load fires once when the browser (re)connects; gr.Timer polls every 2 s
        # so live progress stays visible even if the tab was reloaded mid-training.
        _reconnect_outputs = [
            train_status_card, train_progress, pipeline_status,
            train_quality, train_log_raw, train_loss_plot, train_activity,
        ]
        app.load(fn=_rebuild_training_ui, outputs=_reconnect_outputs)
        try:
            gr.Timer(value=2).tick(fn=_rebuild_training_ui, outputs=_reconnect_outputs)
        except AttributeError:
            pass  # Gradio < 4.20 — page-load restore still works via app.load

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
    p.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_PORT", "7860")))
    p.add_argument(
        "--host", type=str,
        default=os.environ.get("GRADIO_HOST", "0.0.0.0"),
        help="Bind address. Default: 0.0.0.0 (all interfaces). "
             "Set GRADIO_HOST=127.0.0.1 to restrict to localhost only.",
    )
    p.add_argument("--share", action="store_true", default=False)
    args = p.parse_args()

    global _DATA_DIR, _TRAINING_DATA_DIR, _OUTPUT_MODEL_DIR, _SAVED_MODELS_DIR
    if args.data_dir:
        _DATA_DIR = args.data_dir
    if args.training_data_dir:
        _TRAINING_DATA_DIR = args.training_data_dir

    # Support old --model-dir flag and MODEL_DIR env var for backwards compat
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
