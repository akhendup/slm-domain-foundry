"""
Load trained model for inference: use Unsloth when NVIDIA GPU is available,
otherwise use Hugging Face transformers (CPU or Mac MPS).
Handles Unsloth/PEFT saves where config.json may lack model_type.
"""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Tuple

import torch

# Unsloth only when we have a CUDA GPU (avoid NotImplementedError on Mac/CPU)
_UNSLOTH_AVAILABLE = False
try:
    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        _UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    pass

# ONNX Runtime via optimum — CPU inference only (optional)
_ORT_AVAILABLE = False
_ORTModelForCausalLM = None
try:
    from optimum.onnxruntime import ORTModelForCausalLM as _ORTModelForCausalLM
    _ORT_AVAILABLE = True
except (ImportError, Exception):
    pass

# Transformers always available for fallback (CPU / MPS)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Mac Apple Silicon
    return torch.device("cpu")


def _dtype_for_device(device: torch.device) -> torch.dtype:
    """Return the best float dtype for the given device.
    CUDA: bfloat16 | MPS: float16 (bfloat16 unsupported) | CPU: float32
    """
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _resolve_model_dir(model_dir: Path) -> Path:
    """Return the directory that contains config.json (maybe one level down, e.g. output_model/checkpoint-500)."""
    model_dir = Path(model_dir)
    if (model_dir / "config.json").exists():
        return model_dir
    for sub in model_dir.iterdir():
        if sub.is_dir() and (sub / "config.json").exists():
            return sub
    return model_dir  # caller will handle missing config


def _infer_model_type(model_dir: Path) -> str:
    """Infer model_type when config.json or adapter_config.json omit it (e.g. Unsloth/PEFT)."""
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg.get("model_type"):
            return cfg["model_type"]
    adapter_path = model_dir / "adapter_config.json"
    if adapter_path.exists():
        with open(adapter_path) as f:
            adapter = json.load(f)
        base = adapter.get("base_model_name_or_path", "")
        if "llama" in base.lower() or "Llama" in base:
            return "llama"
        if "qwen" in base.lower():
            return "qwen2"
        if "mistral" in base.lower():
            return "mistral"
    # Default for TinyLlama / common Unsloth base
    return "llama"


def _load_with_patched_config(model_dir: Path, device: torch.device) -> Tuple[Any, Any]:
    """Load tokenizer and model when config.json has no model_type (Unsloth/PEFT save)."""
    model_dir = _resolve_model_dir(Path(model_dir))
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {model_dir}. Looked in that directory and one level of subdirectories. "
            "Point --model-dir at the folder that contains config.json (and the model weights)."
        )

    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict.setdefault("model_type", _infer_model_type(model_dir))

    # Copy config + tokenizer/small metadata to temp (skip large weight files); load model from model_dir with patched config
    weight_suffixes = {".safetensors", ".bin", ".msgpack"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in model_dir.iterdir():
            if item.is_file() and item.suffix not in weight_suffixes and not item.name.startswith("flax_"):
                shutil.copy2(item, tmp_path / item.name)
        (tmp_path / "config.json").write_text(json.dumps(config_dict, indent=2))

        tokenizer = AutoTokenizer.from_pretrained(str(tmp_path), trust_remote_code=True)
        config = AutoConfig.from_pretrained(str(tmp_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=_dtype_for_device(device),
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
    if device.type in ("cpu", "mps") and getattr(model, "device_map", None) is None:
        model = model.to(device)
    return model, tokenizer


def load_model(model_dir: Path) -> Tuple[Any, Any]:
    """
    Load model and tokenizer from model_dir.
    Uses Unsloth if CUDA is available, else transformers (CPU or MPS).
    Handles saved dirs where config.json lacks model_type (e.g. Unsloth/PEFT).
    If config.json is not in model_dir, looks one level down (e.g. output_model/checkpoint-500).
    Returns (model, tokenizer).
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    model_dir = _resolve_model_dir(model_dir)

    device = _get_device()

    if _UNSLOTH_AVAILABLE and device.type == "cuda":
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_dir),
                dtype=None,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except (NotImplementedError, Exception):
            pass  # fall through to transformers

    # CPU: try ONNX Runtime for faster inference, fall through to standard transformers
    if device.type == "cpu" and _ORT_AVAILABLE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
            model = _ORTModelForCausalLM.from_pretrained(str(model_dir), export=True)
            return model, tokenizer
        except Exception:
            pass  # fall through to standard transformers

    # Standard HuggingFace transformers (MPS or CUDA fallback from Unsloth)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=_dtype_for_device(device),
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
    except ValueError as e:
        if "model_type" in str(e) or "Unrecognized model" in str(e):
            model, tokenizer = _load_with_patched_config(model_dir, device)
        else:
            raise
    if device.type in ("cpu", "mps") and getattr(model, "device_map", None) is None:
        model = model.to(device)
    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate assistant reply given messages (e.g. [{"role": "user", "content": "..."}])."""
    # Normalise content to str — Gradio 5.x can pass list-of-parts e.g. [{"type":"text","text":"..."}]
    safe_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p["text"] if isinstance(p, dict) and "text" in p else str(p)
                for p in content
            )
        elif not isinstance(content, str):
            content = str(content) if content else ""
        safe_messages.append({"role": msg["role"], "content": content})

    prompt = tokenizer.apply_chat_template(
        safe_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    device = next(model.parameters()).device if hasattr(model, "parameters") else _get_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    if prompt in response:
        response = response.replace(prompt, "").strip()
    for marker in ["<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>"]:
        if marker in response:
            response = response.split(marker)[-1].strip()
    return response
