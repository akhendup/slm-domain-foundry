"""Unit tests for helper functions in demo/gradio_ui.py that don't require a running server."""
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    def test_seconds_only(self):
        from demo.gradio_ui import _format_time
        assert _format_time(45) == "00:45"

    def test_minutes_and_seconds(self):
        from demo.gradio_ui import _format_time
        assert _format_time(90) == "01:30"

    def test_hours(self):
        from demo.gradio_ui import _format_time
        assert _format_time(3661) == "01:01:01"

    def test_zero(self):
        from demo.gradio_ui import _format_time
        assert _format_time(0) == "00:00"

    def test_negative_clamped_to_zero(self):
        from demo.gradio_ui import _format_time
        result = _format_time(-5)
        assert result == "00:00"


# ---------------------------------------------------------------------------
# _activity_text
# ---------------------------------------------------------------------------

class TestActivityText:
    def test_extracts_epoch_lines(self):
        from demo.gradio_ui import _activity_text
        log = "--- Epoch 1/3 ---\nSome other noise\n--- Epoch 2/3 ---"
        result = _activity_text(log)
        assert "Epoch 1/3" in result
        assert "Epoch 2/3" in result

    def test_extracts_step_lines(self):
        from demo.gradio_ui import _activity_text
        log = "step 10/100 10%\nnoise line\nstep 20/100 20%"
        result = _activity_text(log)
        assert "step" in result

    def test_extracts_loss_lines(self):
        from demo.gradio_ui import _activity_text
        log = "Training loss=1.2345 here"
        result = _activity_text(log)
        assert "loss=1.2345" in result

    def test_empty_log(self):
        from demo.gradio_ui import _activity_text
        assert _activity_text("") == ""

    def test_n_limit_respected(self):
        from demo.gradio_ui import _activity_text
        lines = "\n".join(f"--- Epoch {i}/100 ---" for i in range(50))
        result = _activity_text(lines, n=5)
        assert len(result.splitlines()) <= 5

    def test_skips_blank_lines(self):
        from demo.gradio_ui import _activity_text
        log = "\n\n--- Epoch 1/3 ---\n\n"
        result = _activity_text(log)
        assert "Epoch 1/3" in result

    def test_model_loaded_extracted(self):
        from demo.gradio_ui import _activity_text
        log = "Model loaded successfully\nother stuff"
        result = _activity_text(log)
        assert "Model loaded" in result

    def test_trainable_params_extracted(self):
        from demo.gradio_ui import _activity_text
        log = "trainable params: 1,000,000 || all params: 100,000,000"
        result = _activity_text(log)
        assert "trainable params" in result


# ---------------------------------------------------------------------------
# _data_activity_text
# ---------------------------------------------------------------------------

class TestDataActivityText:
    def test_extracts_bracket_lines(self):
        from demo.gradio_ui import _data_activity_text
        log = "[1/3] manual.pdf — 50 Q&A pairs\nother noise"
        result = _data_activity_text(log)
        assert "[1/3]" in result

    def test_extracts_found_lines(self):
        from demo.gradio_ui import _data_activity_text
        log = "Found 3 PDF(s) to process."
        result = _data_activity_text(log)
        assert "Found" in result

    def test_extracts_loading_csv_lines(self):
        from demo.gradio_ui import _data_activity_text
        log = "Loading CSV: data.csv"
        result = _data_activity_text(log)
        assert "Loading CSV" in result

    def test_extracts_total_lines(self):
        from demo.gradio_ui import _data_activity_text
        log = "Total: 120 Q&A pairs — shuffling and splitting…"
        result = _data_activity_text(log)
        assert "Total:" in result

    def test_empty_log(self):
        from demo.gradio_ui import _data_activity_text
        assert _data_activity_text("") == ""

    def test_extracts_qa_pairs_lines(self):
        from demo.gradio_ui import _data_activity_text
        log = "YAML patterns: 45 Q&A pairs, 3 multi-turn conversations"
        result = _data_activity_text(log)
        assert "Q&A pairs" in result


# ---------------------------------------------------------------------------
# _make_train_status_html
# ---------------------------------------------------------------------------

class TestMakeTrainStatusHtml:
    def test_returns_string(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("loading", "Loading model...", [], None, False, False)
        assert isinstance(result, str)

    def test_contains_phase_label(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("training", "Training step 10/100", [], None, False, False)
        assert "Train" in result

    def test_done_state(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("saving", "Training complete!", [], None, True, False)
        assert isinstance(result, str)
        assert "✓" in result

    def test_failed_state(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("training", "", [], "Out of memory", False, True)
        assert "Out of memory" in result or "✗" in result

    def test_warnings_included(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("training", "msg", ["Low memory warning"], None, False, False)
        assert "Low memory warning" in result

    def test_all_phases_rendered(self):
        from demo.gradio_ui import _make_train_status_html
        result = _make_train_status_html("dataset", "Loading data...", [], None, False, False)
        # All 5 phase labels should appear
        for label in ("Load model", "Setup LoRA", "Load data", "Train", "Save"):
            assert label in result


# ---------------------------------------------------------------------------
# _make_training_progress_html
# ---------------------------------------------------------------------------

class TestMakeTrainingProgressHtml:
    def test_returns_string(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(1, 3, 50, 300, 16.7, 120.0)
        assert isinstance(result, str)

    def test_contains_epoch_info(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(2, 5, 100, 500, 40.0, 60.0)
        assert "2" in result and "5" in result

    def test_contains_loss_when_provided(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(1, 3, 50, 300, 16.7, 120.0, loss=1.2345)
        assert "1.2345" in result

    def test_done_state(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(3, 3, 300, 300, 100.0, 300.0, done=True)
        assert "Complete" in result

    def test_failed_state(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(1, 3, 50, 300, 16.7, 120.0, failed=True)
        assert "Failed" in result

    def test_eta_shown_when_progress(self):
        from demo.gradio_ui import _make_training_progress_html
        result = _make_training_progress_html(1, 3, 50, 300, 50.0, 120.0)
        assert "ETA" in result or "eta" in result.lower() or ":" in result


# ---------------------------------------------------------------------------
# _make_pipeline_html
# ---------------------------------------------------------------------------

class TestMakePipelineHtml:
    def test_returns_string(self):
        from demo.gradio_ui import _make_pipeline_html
        result = _make_pipeline_html()
        assert isinstance(result, str)

    def test_contains_pipeline_steps(self):
        from demo.gradio_ui import _make_pipeline_html
        result = _make_pipeline_html()
        for step in ("Upload", "Extract", "Train", "Chat"):
            assert step in result

    def test_pending_status_shown(self, monkeypatch):
        from demo import gradio_ui
        # Force a pending state to verify the HTML renders pending steps
        monkeypatch.setitem(gradio_ui._pipeline_status, "train", "pending")
        result = gradio_ui._make_pipeline_html()
        assert "Pending" in result or "pending" in result


# ---------------------------------------------------------------------------
# _make_quality_html
# ---------------------------------------------------------------------------

class TestMakeQualityHtml:
    def test_returns_empty_when_no_final_loss(self):
        from demo.gradio_ui import _make_quality_html
        assert _make_quality_html(None, None, None, 0) == ""

    def test_returns_string_with_loss(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(0.85, 0.92, 2.1, 1000)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_example_count(self):
        from demo.gradio_ui import _make_quality_html
        result = _make_quality_html(0.85, 0.92, 2.1, 1234)
        assert "1234" in result or "1,234" in result


# ---------------------------------------------------------------------------
# _make_elapsed_html
# ---------------------------------------------------------------------------

class TestMakeElapsedHtml:
    def test_returns_string(self):
        from demo.gradio_ui import _make_elapsed_html
        result = _make_elapsed_html(120.0)
        assert isinstance(result, str)

    def test_done_state(self):
        from demo.gradio_ui import _make_elapsed_html
        result = _make_elapsed_html(300.0, done=True)
        assert "Complete" in result

    def test_failed_state(self):
        from demo.gradio_ui import _make_elapsed_html
        result = _make_elapsed_html(60.0, failed=True)
        assert "Failed" in result


# ---------------------------------------------------------------------------
# _save_dataset_snapshot / _load_dataset_snapshot
# ---------------------------------------------------------------------------

class TestDatasetSnapshot:
    def test_save_returns_none_when_no_files(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", tmp_path / "empty_training")
        from demo.gradio_ui import _save_dataset_snapshot
        path, msg = _save_dataset_snapshot()
        assert path is None
        assert "No training data" in msg

    def test_save_creates_zip(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        (training_dir / "train.jsonl").write_text('{"conversations": []}', encoding="utf-8")
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        from demo.gradio_ui import _save_dataset_snapshot
        path, msg = _save_dataset_snapshot()
        assert path is not None
        assert path.suffix == ".zip"
        assert "Snapshot ready" in msg

    def test_load_unsupported_file_type(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        fake_file = tmp_path / "data.txt"
        fake_file.write_text("data", encoding="utf-8")
        from demo.gradio_ui import _load_dataset_snapshot
        result = _load_dataset_snapshot(str(fake_file))
        assert "Unsupported" in result

    def test_load_jsonl_file(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        jsonl_file = tmp_path / "train.jsonl"
        jsonl_file.write_text('{"conversations": []}', encoding="utf-8")
        from demo.gradio_ui import _load_dataset_snapshot
        result = _load_dataset_snapshot(str(jsonl_file))
        assert "Loaded" in result or "train.jsonl" in result

    def test_load_zip_file(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        zip_path = tmp_path / "snapshot.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.jsonl", '{"conversations": []}')
        from demo.gradio_ui import _load_dataset_snapshot
        result = _load_dataset_snapshot(str(zip_path))
        assert "Loaded" in result or "snapshot" in result.lower()


# ---------------------------------------------------------------------------
# _latest_checkpoint
# ---------------------------------------------------------------------------

class TestLatestCheckpoint:
    def test_returns_none_when_no_checkpoints(self, tmp_path):
        from demo.gradio_ui import _latest_checkpoint
        assert _latest_checkpoint(tmp_path) is None

    def test_returns_latest_checkpoint(self, tmp_path):
        from demo.gradio_ui import _latest_checkpoint
        ckpt1 = tmp_path / "checkpoint-100"
        ckpt1.mkdir()
        ckpt2 = tmp_path / "checkpoint-200"
        ckpt2.mkdir()
        result = _latest_checkpoint(tmp_path)
        assert result is not None
        assert result.name.startswith("checkpoint-")


# ---------------------------------------------------------------------------
# _find_train_jsonl / _merge_jsonl / _find_training_files
# ---------------------------------------------------------------------------

class TestTrainingFileHelpers:
    def test_find_train_jsonl_root(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        (training_dir / "train_sharegpt.jsonl").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        from demo.gradio_ui import _find_train_jsonl
        result = _find_train_jsonl()
        assert result is not None
        assert result.name == "train_sharegpt.jsonl"

    def test_find_train_jsonl_none_when_missing(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        from demo.gradio_ui import _find_train_jsonl
        assert _find_train_jsonl() is None

    def test_merge_jsonl(self, tmp_path):
        from demo.gradio_ui import _merge_jsonl
        src1 = tmp_path / "a.jsonl"
        src2 = tmp_path / "b.jsonl"
        src1.write_text('{"x": 1}\n', encoding="utf-8")
        src2.write_text('{"x": 2}\n', encoding="utf-8")
        dest = tmp_path / "merged.jsonl"
        _merge_jsonl([src1, src2], dest)
        lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_merge_jsonl_skips_missing(self, tmp_path):
        from demo.gradio_ui import _merge_jsonl
        src1 = tmp_path / "exists.jsonl"
        src1.write_text('{"x": 1}\n', encoding="utf-8")
        dest = tmp_path / "merged.jsonl"
        # src2 does not exist
        _merge_jsonl([src1, tmp_path / "missing.jsonl"], dest)
        lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1

    def test_find_training_files_root_exists(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        (training_dir / "train_sharegpt.jsonl").write_text("{}", encoding="utf-8")
        (training_dir / "val_sharegpt.jsonl").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        from demo.gradio_ui import _find_training_files
        train, val = _find_training_files()
        assert train is not None
        assert val is not None

    def test_find_training_files_none_when_missing(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        training_dir = tmp_path / "training_data"
        training_dir.mkdir()
        monkeypatch.setattr(gradio_ui, "_TRAINING_DATA_DIR", training_dir)
        from demo.gradio_ui import _find_training_files
        train, val = _find_training_files()
        assert train is None


# ---------------------------------------------------------------------------
# _active_model_name
# ---------------------------------------------------------------------------

class TestActiveModelName:
    def test_returns_default_when_no_meta(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        monkeypatch.setattr(gradio_ui, "_OUTPUT_MODEL_DIR", tmp_path / "no_model")
        from demo.gradio_ui import _active_model_name
        result = _active_model_name()
        assert result == "output_model"

    def test_reads_from_meta_json(self, tmp_path, monkeypatch):
        from demo import gradio_ui
        monkeypatch.setattr(gradio_ui, "_OUTPUT_MODEL_DIR", tmp_path)
        meta = {"name": "my-trained-model"}
        (tmp_path / "_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        from demo.gradio_ui import _active_model_name
        result = _active_model_name()
        assert result == "my-trained-model"


# ---------------------------------------------------------------------------
# _get_retriever
# ---------------------------------------------------------------------------

class TestGetRetriever:
    def test_returns_knowledge_retriever(self):
        from demo.gradio_ui import _get_retriever
        from data.knowledge_retriever import KnowledgeRetriever
        # Reset global state
        import demo.gradio_ui as ui
        ui._knowledge_retriever = None
        result = _get_retriever()
        assert isinstance(result, KnowledgeRetriever)

    def test_same_instance_returned(self):
        from demo.gradio_ui import _get_retriever
        import demo.gradio_ui as ui
        ui._knowledge_retriever = None
        r1 = _get_retriever()
        r2 = _get_retriever()
        assert r1 is r2
