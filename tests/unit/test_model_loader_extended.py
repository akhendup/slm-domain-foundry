"""
Extended unit tests for demo/model_loader.py.
Covers: _load_with_patched_config, load_model, generate_response.
"""
import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_dir(tmp_path, config_dict=None):
    """Create a minimal model directory with config.json and dummy files."""
    if config_dict is None:
        config_dict = {"model_type": "llama"}
    (tmp_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    # Weight file — should be skipped by _load_with_patched_config
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    return tmp_path


def _make_real_tokenizer_mock():
    """Tokenizer mock whose apply_chat_template returns a real tensor."""
    mock_tok = MagicMock()
    mock_tok.eos_token_id = 2
    input_ids = torch.zeros(1, 5, dtype=torch.long)
    mock_tok.apply_chat_template.return_value = input_ids
    mock_tok.decode.return_value = "  Generated answer.  "
    return mock_tok


def _make_model_mock():
    """Model mock whose generate() returns a real tensor."""
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])
    # generate returns shape (1, 10) — slicing works naturally
    mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)
    mock_model.device_map = None
    return mock_model


# ---------------------------------------------------------------------------
# _load_with_patched_config
# ---------------------------------------------------------------------------

class TestLoadWithPatchedConfig:
    def test_raises_when_no_config_json(self, tmp_path):
        from demo.model_loader import _load_with_patched_config
        with pytest.raises(FileNotFoundError):
            _load_with_patched_config(tmp_path, torch.device("cpu"))

    def test_returns_model_and_tokenizer(self, tmp_path):
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path)

        mock_tokenizer = _make_real_tokenizer_mock()
        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            mock_model_cls.from_pretrained.return_value = mock_model

            model, tokenizer = _load_with_patched_config(model_dir, torch.device("cpu"))

        assert model is not None
        assert tokenizer is not None

    def test_model_moved_to_cpu_device(self, tmp_path):
        """On CPU, model.to(device) is called when device_map is None."""
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path)

        mock_model = _make_model_mock()
        mock_model.device_map = None

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            _load_with_patched_config(model_dir, torch.device("cpu"))

        mock_model.to.assert_called_once_with(torch.device("cpu"))

    def test_model_not_moved_when_has_device_map(self, tmp_path):
        """Model with device_map is not moved manually."""
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path)

        mock_model = _make_model_mock()
        mock_model.device_map = {"": 0}  # truthy — skip .to()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            _load_with_patched_config(model_dir, torch.device("cpu"))

        mock_model.to.assert_not_called()

    def test_injects_model_type_into_temp_config(self, tmp_path):
        """Empty config gets model_type injected via _infer_model_type."""
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path, {})  # no model_type

        written_config = {}

        def capture(path, **kwargs):
            cfg = Path(path) / "config.json"
            if cfg.exists():
                written_config.update(json.loads(cfg.read_text()))
            return _make_real_tokenizer_mock()

        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.side_effect = capture
            mock_model_cls.from_pretrained.return_value = mock_model

            _load_with_patched_config(model_dir, torch.device("cpu"))

        assert written_config.get("model_type") == "llama"  # default from _infer_model_type

    def test_weight_files_not_copied_to_tmp(self, tmp_path):
        """Weight files (.safetensors, .bin) are not copied to temp directory."""
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path)
        (model_dir / "model.bin").write_bytes(b"more weights")

        tmp_dir_check = {}  # capture result during live temp dir

        def capture(path, **kwargs):
            # Check the temp dir while it's still alive
            tmp_dir = Path(path)
            if tmp_dir.is_dir():
                tmp_dir_check["has_weights"] = any(
                    f.suffix in (".safetensors", ".bin")
                    for f in tmp_dir.iterdir()
                    if f.is_file()
                )
            return _make_real_tokenizer_mock()

        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.side_effect = capture
            mock_model_cls.from_pretrained.return_value = mock_model

            _load_with_patched_config(model_dir, torch.device("cpu"))

        assert tmp_dir_check.get("has_weights") is False

    def test_mps_device_moves_model(self, tmp_path):
        """On MPS, model.to(device) is called when device_map is None."""
        from demo.model_loader import _load_with_patched_config
        model_dir = _make_model_dir(tmp_path)

        mock_model = _make_model_mock()
        mock_model.device_map = None

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            _load_with_patched_config(model_dir, torch.device("mps"))

        mock_model.to.assert_called_once_with(torch.device("mps"))

    def test_resolves_subdir_config(self, tmp_path):
        """If config.json is in a subdirectory, it is found and used."""
        from demo.model_loader import _load_with_patched_config
        sub = tmp_path / "checkpoint-100"
        sub.mkdir()
        _make_model_dir(sub)

        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoConfig"), \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            model, tokenizer = _load_with_patched_config(tmp_path, torch.device("cpu"))

        assert model is not None


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModelExtended:
    def _setup(self, tmp_path, monkeypatch):
        """Prepare a model dir and disable GPU/Unsloth/ORT."""
        import demo.model_loader as ml
        model_dir = _make_model_dir(tmp_path)
        monkeypatch.setattr(ml, "_UNSLOTH_AVAILABLE", False)
        monkeypatch.setattr(ml, "_ORT_AVAILABLE", False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        return model_dir

    def test_standard_transformers_path_cpu(self, tmp_path, monkeypatch):
        from demo.model_loader import load_model
        model_dir = self._setup(tmp_path, monkeypatch)

        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            model, tokenizer = load_model(model_dir)

        assert model is not None
        assert tokenizer is not None

    def test_model_moved_to_cpu_after_load(self, tmp_path, monkeypatch):
        from demo.model_loader import load_model
        model_dir = self._setup(tmp_path, monkeypatch)

        mock_model = _make_model_mock()
        mock_model.device_map = None

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            mock_tok_cls.from_pretrained.return_value = _make_real_tokenizer_mock()
            mock_model_cls.from_pretrained.return_value = mock_model

            load_model(model_dir)

        mock_model.to.assert_called_once()

    def test_value_error_model_type_triggers_patched_config(self, tmp_path, monkeypatch):
        """ValueError about model_type falls back to _load_with_patched_config."""
        from demo.model_loader import load_model
        model_dir = self._setup(tmp_path, monkeypatch)

        mock_model = _make_model_mock()
        mock_model.device_map = {"": 0}  # prevent .to() call in load_model
        mock_tokenizer = _make_real_tokenizer_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM"), \
             patch("demo.model_loader._load_with_patched_config",
                   return_value=(mock_model, mock_tokenizer)) as mock_patched:
            mock_tok_cls.from_pretrained.side_effect = ValueError("Unrecognized model in config: unknown")

            model, tokenizer = load_model(model_dir)

        mock_patched.assert_called_once()
        assert model is mock_model

    def test_value_error_about_model_type_keyword(self, tmp_path, monkeypatch):
        """ValueError containing 'model_type' also falls back to _load_with_patched_config."""
        from demo.model_loader import load_model
        model_dir = self._setup(tmp_path, monkeypatch)

        mock_model = _make_model_mock()
        mock_tokenizer = _make_real_tokenizer_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM"), \
             patch("demo.model_loader._load_with_patched_config",
                   return_value=(mock_model, mock_tokenizer)) as mock_patched:
            mock_tok_cls.from_pretrained.side_effect = ValueError("missing model_type in config")

            model, tokenizer = load_model(model_dir)

        mock_patched.assert_called_once()

    def test_other_value_error_is_reraised(self, tmp_path, monkeypatch):
        """ValueError unrelated to model_type propagates."""
        from demo.model_loader import load_model
        model_dir = self._setup(tmp_path, monkeypatch)

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM"):
            mock_tok_cls.from_pretrained.side_effect = ValueError("something completely different")

            with pytest.raises(ValueError, match="something completely different"):
                load_model(model_dir)

    def test_ort_path_used_when_available(self, tmp_path, monkeypatch):
        """ORT path is taken on CPU when _ORT_AVAILABLE is True."""
        from demo.model_loader import load_model
        import demo.model_loader as ml
        model_dir = _make_model_dir(tmp_path)

        monkeypatch.setattr(ml, "_UNSLOTH_AVAILABLE", False)
        monkeypatch.setattr(ml, "_ORT_AVAILABLE", True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        mock_ort_model = MagicMock()
        mock_tokenizer = _make_real_tokenizer_mock()
        mock_ort_cls = MagicMock()
        mock_ort_cls.from_pretrained.return_value = mock_ort_model

        monkeypatch.setattr(ml, "_ORTModelForCausalLM", mock_ort_cls)

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls:
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            model, tokenizer = load_model(model_dir)

        assert model is mock_ort_model

    def test_ort_exception_falls_through(self, tmp_path, monkeypatch):
        """ORT path exception falls through to standard transformers."""
        from demo.model_loader import load_model
        import demo.model_loader as ml
        model_dir = _make_model_dir(tmp_path)

        monkeypatch.setattr(ml, "_UNSLOTH_AVAILABLE", False)
        monkeypatch.setattr(ml, "_ORT_AVAILABLE", True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        mock_ort_cls = MagicMock()
        mock_ort_cls.from_pretrained.side_effect = Exception("ORT load failed")
        monkeypatch.setattr(ml, "_ORTModelForCausalLM", mock_ort_cls)

        mock_model = _make_model_mock()

        with patch("demo.model_loader.AutoTokenizer") as mock_tok_cls, \
             patch("demo.model_loader.AutoModelForCausalLM") as mock_model_cls:
            # first call (ORT path) raises, second call (standard) succeeds
            mock_tok_cls.from_pretrained.side_effect = [
                Exception("ORT tokenizer"),  # ORT path raises
                _make_real_tokenizer_mock(),  # standard path succeeds
            ]
            mock_model_cls.from_pretrained.return_value = mock_model

            model, tokenizer = load_model(model_dir)

        assert model is not None


# ---------------------------------------------------------------------------
# generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    def test_returns_string(self):
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        result = generate_response(model, tokenizer, [{"role": "user", "content": "What is CSUM?"}])
        assert isinstance(result, str)

    def test_strips_whitespace(self):
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        tokenizer.decode.return_value = "  padded response  "
        result = generate_response(model, tokenizer, [{"role": "user", "content": "Q"}])
        assert result == "padded response"

    def test_string_content_passed_through(self):
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [{"role": "user", "content": "Plain question"}]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["content"] == "Plain question"

    def test_list_content_joined_to_string(self):
        """Gradio 5.x passes content as [{type, text}, ...] — should be joined."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}],
            }
        ]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        content = call_args[0]["content"]
        assert "Hello" in content
        assert "World" in content

    def test_list_content_non_dict_items(self):
        """List content with non-dict items: str() is called on each element."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [{"role": "user", "content": ["part1", "part2"]}]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        content = call_args[0]["content"]
        assert "part1" in content

    def test_non_string_int_content_converted(self):
        """Non-string, non-list content is converted via str()."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [{"role": "user", "content": 42}]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["content"] == "42"

    def test_none_content_becomes_empty_string(self):
        """None content converts to empty string."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [{"role": "user", "content": None}]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["content"] == ""

    def test_tpl_out_with_input_ids_attribute(self):
        """BatchEncoding-like output: uses .input_ids attribute."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()

        # Simulate BatchEncoding with input_ids
        tpl_out = MagicMock()
        tpl_out.input_ids = torch.zeros(1, 5, dtype=torch.long)
        tokenizer.apply_chat_template.return_value = tpl_out

        result = generate_response(model, tokenizer, [{"role": "user", "content": "Q"}])
        assert isinstance(result, str)

    def test_generate_called_with_custom_params(self):
        """max_new_tokens and temperature are passed to model.generate."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        generate_response(
            model, tokenizer,
            [{"role": "user", "content": "Q?"}],
            max_new_tokens=512,
            temperature=0.7,
        )
        kw = model.generate.call_args[1]
        assert kw["max_new_tokens"] == 512
        assert kw["temperature"] == 0.7

    def test_do_sample_false_when_temperature_zero(self):
        """temperature=0.0 → do_sample=False."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        generate_response(model, tokenizer, [{"role": "user", "content": "Q"}], temperature=0.0)
        kw = model.generate.call_args[1]
        assert kw["do_sample"] is False

    def test_do_sample_true_when_temperature_positive(self):
        """temperature > 0 → do_sample=True."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        generate_response(model, tokenizer, [{"role": "user", "content": "Q"}], temperature=0.5)
        kw = model.generate.call_args[1]
        assert kw["do_sample"] is True

    def test_no_model_parameters_uses_get_device(self):
        """When model has no parameters(), device is inferred from _get_device."""
        from demo.model_loader import generate_response
        # SimpleNamespace: no 'parameters' attribute, has 'generate'
        model = types.SimpleNamespace(
            generate=MagicMock(return_value=torch.zeros(1, 10, dtype=torch.long))
        )
        tokenizer = _make_real_tokenizer_mock()
        result = generate_response(model, tokenizer, [{"role": "user", "content": "Q"}])
        assert isinstance(result, str)

    def test_multiturn_messages(self):
        """Multiple messages (multi-turn) are forwarded to apply_chat_template."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        generate_response(model, tokenizer, messages)
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert len(call_args) == 3

    def test_decode_called_on_new_tokens_only(self):
        """Only the newly generated tokens (after input) are decoded."""
        from demo.model_loader import generate_response
        model = _make_model_mock()
        tokenizer = _make_real_tokenizer_mock()
        # input_ids has 5 tokens; generate returns 10 tokens total
        # so 5 new tokens should be decoded
        decoded_slices = []

        def capture_decode(tokens, **kwargs):
            decoded_slices.append(tokens)
            return "result"

        tokenizer.decode.side_effect = capture_decode
        generate_response(model, tokenizer, [{"role": "user", "content": "Q"}])

        assert len(decoded_slices) == 1
        # The decoded slice should have length 10 - 5 = 5
        assert decoded_slices[0].shape[0] == 5
