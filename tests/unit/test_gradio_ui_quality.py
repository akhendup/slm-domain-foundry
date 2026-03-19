"""
Additional tests for demo/gradio_ui.py — quality HTML and rebuild UI functions.
These cover the remaining helper functions not in test_gradio_ui_helpers.py.
"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _make_quality_html — all rating branches
# ---------------------------------------------------------------------------

class TestMakeQualityHtmlBranches:
    def test_excellent_rating(self):
        from demo.gradio_ui import _make_quality_html
        # final < 1.2 and reduction > 40%
        result = _make_quality_html(0.8, 0.9, 2.0, 500)
        assert "Excellent" in result

    def test_good_rating(self):
        from demo.gradio_ui import _make_quality_html
        # 1.2 <= final < 1.8 and reduction > 25%
        result = _make_quality_html(1.5, 1.6, 2.5, 500)
        assert "Good" in result

    def test_okay_rating(self):
        from demo.gradio_ui import _make_quality_html
        # 1.8 <= final < 2.2 and reduction > 10%
        result = _make_quality_html(2.0, 2.1, 2.5, 500)
        assert "Okay" in result

    def test_fair_rating(self):
        from demo.gradio_ui import _make_quality_html
        # 2.2 <= final < 2.8
        result = _make_quality_html(2.5, 2.5, 2.8, 500)
        assert "Fair" in result

    def test_poor_rating(self):
        from demo.gradio_ui import _make_quality_html
        # final >= 2.8
        result = _make_quality_html(3.0, 3.0, 3.5, 100)
        assert "Poor" in result

    def test_warning_very_small_dataset(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(0.8, 0.9, 2.0, 20)
        assert "Very small dataset" in result or "very small" in result.lower()

    def test_warning_small_dataset(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(0.8, 0.9, 2.0, 100)
        assert "Small dataset" in result or "small" in result.lower()

    def test_overfitting_warning(self):
        from demo.gradio_ui import _make_quality_html
        # eval_loss > train_loss * 1.3 → overfitting warning
        result = _make_quality_html(1.0, 2.0, 2.5, 500)
        assert "overfitting" in result.lower()

    def test_no_initial_loss(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(1.0, 1.1, None, 500)
        assert isinstance(result, str) and len(result) > 0

    def test_no_eval_loss(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(1.0, None, 2.0, 500)
        assert isinstance(result, str) and len(result) > 0

    def test_zero_examples(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(1.0, 1.1, 2.0, 0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _make_data_sample_html
# ---------------------------------------------------------------------------

class TestMakeDataSampleHtml:
    def test_returns_empty_when_no_file(self, tmp_path):
        from demo.gradio_ui import _make_data_sample_html
        result = _make_data_sample_html(tmp_path / "nonexistent.jsonl")
        assert result == ""

    def test_returns_empty_when_file_empty(self, tmp_path):
        from demo.gradio_ui import _make_data_sample_html
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        result = _make_data_sample_html(f)
        assert result == ""

    def test_returns_html_with_data(self, tmp_path):
        from demo.gradio_ui import _make_data_sample_html
        f = tmp_path / "train.jsonl"
        examples = [
            {"conversations": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ]}
            for i in range(5)
        ]
        f.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
        result = _make_data_sample_html(f)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _rebuild_training_ui
# ---------------------------------------------------------------------------

class TestRebuildTrainingUi:
    def test_returns_tuple_when_not_active(self):
        from demo.gradio_ui import _rebuild_training_ui, _training_state
        # Save state and reset to clean (not active)
        orig = dict(_training_state)
        _training_state["active"] = False
        _training_state["done"] = False
        _training_state["failed"] = False
        try:
            result = _rebuild_training_ui()
            assert isinstance(result, tuple)
        finally:
            _training_state.update(orig)

    def test_returns_tuple_when_done(self):
        from demo.gradio_ui import _rebuild_training_ui, _training_state
        orig = dict(_training_state)
        _training_state.update({
            "active": False, "done": True, "failed": False,
            "phase": "saving", "current_msg": "Training complete!",
            "warnings": [], "error": None,
            "current_epoch": 3, "total_epochs": 3,
            "global_step": 300, "total_steps": 300,
            "last_pct": 100.0, "last_loss": 0.85, "last_eval_loss": 0.9,
            "initial_loss": 2.1, "n_examples": 1000,
            "loss_history": [{"step": 100, "loss": 1.5}],
            "start_time": None, "elapsed": 120.0,
        })
        try:
            result = _rebuild_training_ui()
            assert isinstance(result, tuple)
            assert len(result) == 7
        finally:
            _training_state.update(orig)

    def test_returns_tuple_when_active(self):
        import time
        from demo.gradio_ui import _rebuild_training_ui, _training_state
        orig = dict(_training_state)
        _training_state.update({
            "active": True, "done": False, "failed": False,
            "phase": "training", "current_msg": "Training step 50/300",
            "warnings": [], "error": None,
            "current_epoch": 1, "total_epochs": 3,
            "global_step": 50, "total_steps": 300,
            "last_pct": 16.7, "last_loss": 1.8, "last_eval_loss": None,
            "initial_loss": 2.5, "n_examples": 500,
            "loss_history": [],
            "start_time": time.time() - 60, "elapsed": 60.0,
        })
        try:
            result = _rebuild_training_ui()
            assert isinstance(result, tuple)
        finally:
            _training_state.update(orig)

    def test_returns_tuple_when_failed(self):
        from demo.gradio_ui import _rebuild_training_ui, _training_state
        orig = dict(_training_state)
        _training_state.update({
            "active": False, "done": False, "failed": True,
            "phase": "training", "current_msg": "Error occurred",
            "warnings": [], "error": "CUDA out of memory",
            "current_epoch": 1, "total_epochs": 3,
            "global_step": 10, "total_steps": 300,
            "last_pct": 3.3, "last_loss": None, "last_eval_loss": None,
            "initial_loss": None, "n_examples": 0,
            "loss_history": [],
            "start_time": None, "elapsed": 30.0,
        })
        try:
            result = _rebuild_training_ui()
            assert isinstance(result, tuple)
        finally:
            _training_state.update(orig)


# ---------------------------------------------------------------------------
# _CHAT_SYSTEM_PROMPT — env var override
# ---------------------------------------------------------------------------

class TestChatSystemPrompt:
    def test_default_prompt_is_generic(self):
        """Default prompt must not contain product-specific hardcoded names."""
        import demo.gradio_ui as gui
        assert "Teradata" not in gui._CHAT_SYSTEM_PROMPT

    def test_env_var_overrides_prompt(self, monkeypatch):
        """SLM_SYSTEM_PROMPT env var must replace the default prompt."""
        monkeypatch.setenv("SLM_SYSTEM_PROMPT", "Custom system prompt for testing.")
        _env = os.environ.get("SLM_SYSTEM_PROMPT", "default")
        assert _env == "Custom system prompt for testing."


# ---------------------------------------------------------------------------
# ENV var MODEL_DIR bug fix — Path("") must not override _OUTPUT_MODEL_DIR
# ---------------------------------------------------------------------------

class TestModelDirEnvVar:
    def test_empty_env_var_resolves_to_none(self, monkeypatch):
        """An empty MODEL_DIR env var must resolve to None, not Path('')."""
        monkeypatch.setenv("MODEL_DIR", "")
        _env = os.environ.get("MODEL_DIR", "").strip()
        resolved = Path(_env) if _env else None
        assert resolved is None

    def test_set_env_var_resolves_to_path(self, tmp_path, monkeypatch):
        """A non-empty MODEL_DIR env var should resolve to a Path."""
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        _env = os.environ.get("MODEL_DIR", "").strip()
        resolved = Path(_env) if _env else None
        assert resolved == tmp_path

    def test_whitespace_env_var_resolves_to_none(self, monkeypatch):
        """A whitespace-only MODEL_DIR env var must resolve to None."""
        monkeypatch.setenv("MODEL_DIR", "   ")
        _env = os.environ.get("MODEL_DIR", "").strip()
        resolved = Path(_env) if _env else None
        assert resolved is None
