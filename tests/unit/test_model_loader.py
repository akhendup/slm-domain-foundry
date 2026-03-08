"""Unit tests for demo/model_loader.py — device/dtype/config helpers only (no model weights)."""
import json
import shutil

import pytest
import torch

from demo.model_loader import (
    _dtype_for_device,
    _get_device,
    _infer_model_type,
    _resolve_model_dir,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        device = _get_device()
        assert isinstance(device, torch.device)

    def test_device_type_is_valid(self):
        device = _get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_cpu_when_no_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        # Also patch MPS if present
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        device = _get_device()
        assert device.type == "cpu"

    def test_cuda_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        device = _get_device()
        assert device.type == "cuda"


# ---------------------------------------------------------------------------
# _dtype_for_device
# ---------------------------------------------------------------------------

class TestDtypeForDevice:
    def test_cuda_returns_bfloat16(self):
        assert _dtype_for_device(torch.device("cuda")) == torch.bfloat16

    def test_mps_returns_float16(self):
        assert _dtype_for_device(torch.device("mps")) == torch.float16

    def test_cpu_returns_float32(self):
        assert _dtype_for_device(torch.device("cpu")) == torch.float32


# ---------------------------------------------------------------------------
# _resolve_model_dir
# ---------------------------------------------------------------------------

class TestResolveModelDir:
    def test_returns_dir_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        result = _resolve_model_dir(tmp_path)
        assert result == tmp_path

    def test_finds_config_one_level_down(self, tmp_path):
        sub = tmp_path / "checkpoint-500"
        sub.mkdir()
        (sub / "config.json").write_text("{}", encoding="utf-8")
        result = _resolve_model_dir(tmp_path)
        assert result == sub

    def test_returns_original_when_no_config(self, tmp_path):
        # No config.json anywhere — returns original path (caller handles missing config)
        result = _resolve_model_dir(tmp_path)
        assert result == tmp_path

    def test_prefers_top_level_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        sub = tmp_path / "checkpoint-100"
        sub.mkdir()
        (sub / "config.json").write_text("{}", encoding="utf-8")
        result = _resolve_model_dir(tmp_path)
        assert result == tmp_path


# ---------------------------------------------------------------------------
# _infer_model_type
# ---------------------------------------------------------------------------

class TestInferModelType:
    def test_reads_model_type_from_config(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "mistral"}), encoding="utf-8"
        )
        assert _infer_model_type(tmp_path) == "mistral"

    def test_infers_llama_from_adapter(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta-llama/Llama-2-7b-hf"}),
            encoding="utf-8",
        )
        assert _infer_model_type(tmp_path) == "llama"

    def test_infers_qwen_from_adapter(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen2-1.5B"}),
            encoding="utf-8",
        )
        assert _infer_model_type(tmp_path) == "qwen2"

    def test_infers_mistral_from_adapter(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "mistralai/Mistral-7B-v0.1"}),
            encoding="utf-8",
        )
        assert _infer_model_type(tmp_path) == "mistral"

    def test_defaults_to_llama(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert _infer_model_type(tmp_path) == "llama"

    def test_load_model_raises_on_missing_dir(self, tmp_path):
        from demo.model_loader import load_model
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_model(missing)
