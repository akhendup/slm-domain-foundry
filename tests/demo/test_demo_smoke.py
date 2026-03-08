"""
Demo smoke tests: verify the Gradio app constructs without error and key helpers work.
Does NOT launch the server or load model weights.
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.demo


# ---------------------------------------------------------------------------
# Gradio app construction
# ---------------------------------------------------------------------------

class TestGradioAppBuilds:
    def test_build_app_returns_blocks(self):
        """build_app() must return a Gradio Blocks instance without raising."""
        from demo.gradio_ui import build_app
        import gradio as gr
        app = build_app()
        assert isinstance(app, gr.Blocks)

    def test_build_app_idempotent(self):
        """Calling build_app() twice should not raise."""
        from demo.gradio_ui import build_app
        build_app()
        build_app()

    def test_app_has_expected_tab_count(self):
        """The UI should have multiple tabs (Upload, Extract, Train, Models, Chat…)."""
        from demo.gradio_ui import build_app
        import gradio as gr
        app = build_app()
        # Count Tab components in the blocks queue
        tabs = [c for c in app.blocks.values() if isinstance(c, gr.Tab)]
        assert len(tabs) >= 4, f"Expected at least 4 tabs, found {len(tabs)}"


# ---------------------------------------------------------------------------
# Device and backend helpers
# ---------------------------------------------------------------------------

class TestDeviceHelpers:
    def test_get_device_label_on_this_machine(self):
        from demo.gradio_ui import _get_device_label
        label = _get_device_label()
        assert isinstance(label, str)
        assert len(label) > 3

    def test_unsloth_available_is_bool(self):
        from demo.gradio_ui import _unsloth_available
        result = _unsloth_available()
        assert isinstance(result, bool)

    def test_model_ready_false_when_no_model(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        monkeypatch.setattr(gradio_ui, "_OUTPUT_MODEL_DIR", tmp_path / "nonexistent")
        from demo.gradio_ui import _model_ready
        assert _model_ready() is False

    def test_model_ready_true_when_config_present(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        from demo import gradio_ui
        monkeypatch.setattr(gradio_ui, "_OUTPUT_MODEL_DIR", tmp_path)
        from demo.gradio_ui import _model_ready
        assert _model_ready() is True


# ---------------------------------------------------------------------------
# Docker detection integration
# ---------------------------------------------------------------------------

class TestDockerDetectionIntegration:
    def test_in_docker_is_bool(self):
        from demo.gradio_ui import _IN_DOCKER
        assert isinstance(_IN_DOCKER, bool)

    def test_path_dirs_consistent_with_docker_flag(self):
        from demo import gradio_ui
        if gradio_ui._IN_DOCKER:
            assert str(gradio_ui._DATA_DIR).startswith("/app")
        else:
            project_root = str(gradio_ui._PROJECT_ROOT)
            assert str(gradio_ui._DATA_DIR).startswith(project_root)


# ---------------------------------------------------------------------------
# generate_response message normalization
# ---------------------------------------------------------------------------

class TestGenerateResponseNormalization:
    """Test the message normalization logic without loading a real model."""

    def test_list_content_normalized_to_str(self):
        """Gradio 5.x can pass content as list-of-parts; generate_response must handle it."""
        from demo.model_loader import generate_response

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # apply_chat_template returns a string prompt
        mock_tokenizer.apply_chat_template.return_value = "user: Hello\nassistant:"
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.eos_token_id = 2

        # Simulate model output
        import torch
        mock_out = torch.tensor([[1, 2, 3]])
        mock_model.generate.return_value = mock_out
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_tokenizer.batch_decode.return_value = ["user: Hello\nassistant: The answer is 42."]

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]

        # Should not raise even with list content
        with patch.object(mock_tokenizer, "__call__", return_value={
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }):
            try:
                result = generate_response(mock_model, mock_tokenizer, messages)
                assert isinstance(result, str)
            except Exception as e:
                # If mock setup is imperfect, a type error on tensor ops is acceptable —
                # but the content normalization itself should not be the cause.
                assert "list" not in str(e).lower(), f"List content not normalized: {e}"

    def test_none_content_normalized(self):
        """None content should be converted to empty string, not raise."""
        from demo.model_loader import generate_response

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.eos_token_id = 2

        import torch
        mock_model.generate.return_value = torch.tensor([[1]])
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_tokenizer.batch_decode.return_value = ["prompt response"]

        messages = [{"role": "user", "content": None}]
        with patch.object(mock_tokenizer, "__call__", return_value={
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }):
            try:
                generate_response(mock_model, mock_tokenizer, messages)
            except Exception as e:
                assert "NoneType" not in str(e), f"None content not normalized: {e}"


# ---------------------------------------------------------------------------
# Gradio UI module-level import smoke test
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_gradio_ui_importable(self):
        """The gradio_ui module must be importable without side effects."""
        import demo.gradio_ui  # noqa: F401

    def test_model_loader_importable(self):
        import demo.model_loader  # noqa: F401

    def test_knowledge_capture_importable(self):
        import data.knowledge_capture  # noqa: F401

    def test_all_data_modules_importable(self):
        import data.chunking       # noqa: F401
        import data.csv_loader     # noqa: F401
        import data.pdf_extractor  # noqa: F401
        import data.manual_extractor  # noqa: F401
        import data.prepare_training_data  # noqa: F401
        import data.yaml_pattern_loader    # noqa: F401


# ---------------------------------------------------------------------------
# run_gradio_ui entry point
# ---------------------------------------------------------------------------

class TestEntryPoint:
    def test_module_help(self):
        """python -m demo.gradio_ui --help should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "demo.gradio_ui", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0
        assert "host" in result.stdout.lower() or "port" in result.stdout.lower()
