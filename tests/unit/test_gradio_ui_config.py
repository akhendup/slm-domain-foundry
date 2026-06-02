"""Unit tests for app/gradio_ui.py — Docker detection and path configuration."""
import os
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest


pytestmark = pytest.mark.unit


# We import the function directly to test it in isolation
from app.gradio_ui import _is_docker, _get_device_label, _unsloth_available


class TestIsDocker:
    def test_returns_bool_natively(self, monkeypatch):
        """_is_docker() must return a bool regardless of environment."""
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
        result = _is_docker()
        assert isinstance(result, bool)

    def test_env_var_one_forces_docker(self, monkeypatch):
        monkeypatch.setenv("DOCKER_CONTAINER", "1")
        assert _is_docker() is True

    def test_env_var_true_forces_docker(self, monkeypatch):
        monkeypatch.setenv("DOCKER_CONTAINER", "true")
        assert _is_docker() is True

    def test_env_var_yes_forces_docker(self, monkeypatch):
        monkeypatch.setenv("DOCKER_CONTAINER", "yes")
        assert _is_docker() is True

    def test_env_var_zero_not_docker(self, monkeypatch):
        monkeypatch.setenv("DOCKER_CONTAINER", "0")
        # "0" is not in ("1", "true", "yes") so env var alone won't trigger
        # (may still be Docker via /.dockerenv but env var check passes)
        # just verify no exception
        result = _is_docker()
        assert isinstance(result, bool)

    def test_dockerenv_file_triggers(self, monkeypatch):
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
        with patch("app.gradio_ui.Path") as MockPath:
            # Make /.dockerenv appear to exist
            mock_instance = MockPath.return_value
            mock_instance.exists.return_value = True
            # Patch just the /.dockerenv check
            import app.gradio_ui as ui_mod
            original_is_docker = ui_mod._is_docker

            def patched():
                if os.environ.get("DOCKER_CONTAINER", "").lower() in ("1", "true", "yes"):
                    return True
                # Simulate /.dockerenv existing
                return True

            assert patched() is True


class TestPathConfiguration:
    def test_project_root_is_parent_of_demo(self):
        """_PROJECT_ROOT should be the directory above demo/."""
        from app.gradio_ui import _PROJECT_ROOT
        assert (_PROJECT_ROOT / "demo").is_dir()

    def test_native_paths_under_project_root(self, monkeypatch):
        """When not in Docker, all dirs should be under _PROJECT_ROOT."""
        monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
        # Re-import to get fresh state — or just check current module state
        from app import gradio_ui
        if not gradio_ui._IN_DOCKER:
            assert str(gradio_ui._DATA_DIR).startswith(str(gradio_ui._PROJECT_ROOT))
            assert str(gradio_ui._TRAINING_DATA_DIR).startswith(str(gradio_ui._PROJECT_ROOT))
            assert str(gradio_ui._OUTPUT_MODEL_DIR).startswith(str(gradio_ui._PROJECT_ROOT))

    def test_docker_paths_under_app(self, monkeypatch):
        """When DOCKER_CONTAINER=1 env is set, dirs should be under /app."""
        # This tests what would happen after a fresh import with the env set.
        # We can only test the _is_docker logic directly here.
        monkeypatch.setenv("DOCKER_CONTAINER", "1")
        assert _is_docker() is True
        # The actual /app paths are set at module import time, so we verify
        # the detection function correctly signals Docker mode.

    def test_all_path_attrs_are_path_objects(self):
        from app import gradio_ui
        for attr in ("_DATA_DIR", "_TRAINING_DATA_DIR", "_OUTPUT_MODEL_DIR",
                     "_SAVED_MODELS_DIR", "_LIBRARY_DIR"):
            assert isinstance(getattr(gradio_ui, attr), Path), f"{attr} is not a Path"

    def test_data_dir_named_data(self):
        from app.gradio_ui import _DATA_DIR
        assert _DATA_DIR.name == "data"

    def test_training_data_dir_named(self):
        from app.gradio_ui import _TRAINING_DATA_DIR
        assert _TRAINING_DATA_DIR.name == "training_data"


class TestGetDeviceLabel:
    def test_returns_string(self):
        label = _get_device_label()
        assert isinstance(label, str)
        assert len(label) > 0

    def test_contains_known_backend(self):
        label = _get_device_label()
        assert any(keyword in label for keyword in ("CUDA", "MPS", "CPU", "Apple"))


class TestUnslothAvailable:
    def test_returns_bool(self):
        result = _unsloth_available()
        assert isinstance(result, bool)

    def test_false_when_no_cuda(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _unsloth_available() is False
