"""Real model load and generation (tiny HF checkpoint on disk)."""
import json
import shutil

import pytest
import torch

TINY_HF = "sshleifer/tiny-gpt2"

from app.model_loader import (
    _configure_tokenizer_pad,
    _infer_model_type,
    _load_peft_adapter,
    _load_with_patched_config,
    _resolve_model_dir,
    generate_response,
    load_model,
)

pytestmark = [pytest.mark.real, pytest.mark.unit]

peft = pytest.importorskip("peft")


class TestResolveModelDirReal:
    def test_merged_model_dir(self, tiny_lm_dir):
        assert _resolve_model_dir(tiny_lm_dir) == tiny_lm_dir

    def test_peft_adapter_dir(self, tmp_path):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(TINY_HF)
        tok = AutoTokenizer.from_pretrained(TINY_HF)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        peft = get_peft_model(
            base,
            LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["c_attn"]),
        )
        adapter_dir = tmp_path / "adapter_ckpt"
        peft.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        assert _resolve_model_dir(adapter_dir) == adapter_dir


class TestLoadAndGenerateReal:
    def test_load_merged_and_generate(self, tiny_lm_dir):
        model, tokenizer = load_model(tiny_lm_dir)
        _configure_tokenizer_pad(tokenizer)
        reply = generate_response(
            model,
            tokenizer,
            [{"role": "user", "content": "Say OK"}],
            max_new_tokens=8,
            temperature=0.0,
        )
        assert isinstance(reply, str)

    def test_generate_list_content(self, tiny_lm_dir):
        model, tokenizer = load_model(tiny_lm_dir)
        reply = generate_response(
            model,
            tokenizer,
            [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            max_new_tokens=6,
            temperature=0.0,
        )
        assert isinstance(reply, str)

    def test_load_peft_adapter_real(self, tmp_path):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_name = TINY_HF
        base = AutoModelForCausalLM.from_pretrained(base_name)
        tok = AutoTokenizer.from_pretrained(base_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        peft = get_peft_model(
            base,
            LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["c_attn"]),
        )
        adapter_dir = tmp_path / "peft_out"
        peft.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        with open(adapter_dir / "adapter_config.json") as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = base_name
        (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")

        device = torch.device("cpu")
        model, tokenizer = _load_peft_adapter(adapter_dir, device)
        out = generate_response(
            model,
            tokenizer,
            [{"role": "user", "content": "Test"}],
            max_new_tokens=5,
            temperature=0.0,
        )
        assert isinstance(out, str)

    def test_patched_config_load(self, tiny_lm_dir, tmp_path):
        broken = tmp_path / "broken"
        broken.mkdir()
        cfg = json.loads((tiny_lm_dir / "config.json").read_text(encoding="utf-8"))
        del cfg["model_type"]
        (broken / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            src = tiny_lm_dir / name
            if src.exists():
                shutil.copy2(src, broken / name)
        shutil.copy2(tiny_lm_dir / "model.safetensors", broken / "model.safetensors")
        model, tok = _load_with_patched_config(broken, torch.device("cpu"))
        assert model is not None
        assert _infer_model_type(broken) in ("gpt2", "llama", "qwen2", "mistral")
