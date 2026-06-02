"""Unit tests for app/swarm.py — no real model weights required."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.swarm import SwarmManager

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_load(model_dir):
    """Return mock (model, tokenizer) pair instead of loading real weights."""
    return MagicMock(name="model"), MagicMock(name="tokenizer")


def _mock_generate(model, tokenizer, messages, **kwargs):
    return "mock answer"


# ---------------------------------------------------------------------------
# SwarmManager.load
# ---------------------------------------------------------------------------

class TestSwarmLoad:
    def test_load_success_returns_ok_message(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            msg = s.load("m1", tmp_path)
        assert "Loaded" in msg and "m1" in msg

    def test_load_adds_to_pool(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        assert "m1" in s.names()

    def test_load_missing_dir_returns_error(self, tmp_path):
        s = SwarmManager()
        msg = s.load("m1", tmp_path / "does_not_exist")
        assert "not found" in msg.lower()
        assert "m1" not in s.names()

    def test_load_empty_name_returns_error(self, tmp_path):
        s = SwarmManager()
        msg = s.load("  ", tmp_path)
        assert "empty" in msg.lower() or "name" in msg.lower()

    def test_load_duplicate_name_returns_error(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
            msg = s.load("m1", tmp_path)
        assert "already" in msg.lower()

    def test_load_exception_returns_error_message(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=RuntimeError("gpu oom")):
            msg = s.load("bad", tmp_path)
        assert "Failed" in msg or "gpu oom" in msg
        assert "bad" not in s.names()

    def test_load_multiple_different_names(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("a", tmp_path)
            s.load("b", tmp_path)
        assert s.names() == ["a", "b"]


# ---------------------------------------------------------------------------
# SwarmManager.unload
# ---------------------------------------------------------------------------

class TestSwarmUnload:
    def test_unload_removes_from_pool(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        s.unload("m1")
        assert "m1" not in s.names()

    def test_unload_returns_ok_message(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        msg = s.unload("m1")
        assert "Unloaded" in msg

    def test_unload_nonexistent_returns_error(self):
        s = SwarmManager()
        msg = s.unload("ghost")
        assert "not in" in msg.lower()

    def test_unload_does_not_affect_other_models(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("keep", tmp_path)
            s.load("remove", tmp_path)
        s.unload("remove")
        assert "keep" in s.names()
        assert "remove" not in s.names()


# ---------------------------------------------------------------------------
# SwarmManager.clear
# ---------------------------------------------------------------------------

class TestSwarmClear:
    def test_clear_removes_all(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
            s.load("m2", tmp_path)
        removed = s.clear()
        assert removed == 2
        assert s.size() == 0

    def test_clear_empty_swarm_returns_zero(self):
        s = SwarmManager()
        assert s.clear() == 0


# ---------------------------------------------------------------------------
# SwarmManager introspection
# ---------------------------------------------------------------------------

class TestSwarmIntrospection:
    def test_names_sorted_alphabetically(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("zebra", tmp_path)
            s.load("alpha", tmp_path)
            s.load("beta", tmp_path)
        assert s.names() == ["alpha", "beta", "zebra"]

    def test_empty_swarm_names_returns_empty_list(self):
        assert SwarmManager().names() == []

    def test_size_increments_on_load(self, tmp_path):
        s = SwarmManager()
        assert s.size() == 0
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        assert s.size() == 1
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m2", tmp_path)
        assert s.size() == 2

    def test_size_decrements_on_unload(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        s.unload("m1")
        assert s.size() == 0

    def test_is_loaded_true_after_load(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        assert s.is_loaded("m1") is True

    def test_is_loaded_false_before_load(self):
        assert SwarmManager().is_loaded("nope") is False

    def test_is_loaded_false_after_unload(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
        s.unload("m1")
        assert s.is_loaded("m1") is False


# ---------------------------------------------------------------------------
# SwarmManager.generate_one
# ---------------------------------------------------------------------------

class TestGenerateOne:
    def test_calls_generate_response(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load), \
             patch("app.swarm.generate_response", return_value="the answer"):
            s.load("m1", tmp_path)
            result = s.generate_one("m1", [{"role": "user", "content": "q"}])
        assert result == "the answer"

    def test_missing_model_returns_error_string(self):
        s = SwarmManager()
        result = s.generate_one("ghost", [])
        assert "ghost" in result
        assert "not in" in result.lower() or "not" in result.lower()

    def test_passes_kwargs_to_generate(self, tmp_path):
        s = SwarmManager()
        captured = {}

        def _cap(m, t, msgs, **kw):
            captured.update(kw)
            return "ok"

        with patch("app.swarm.load_model", side_effect=_mock_load), \
             patch("app.swarm.generate_response", side_effect=_cap):
            s.load("m1", tmp_path)
            s.generate_one("m1", [], max_new_tokens=64, temperature=0.1)

        assert captured.get("max_new_tokens") == 64
        assert captured.get("temperature") == 0.1


# ---------------------------------------------------------------------------
# SwarmManager.generate_all
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_empty_swarm_returns_empty_dict(self):
        assert SwarmManager().generate_all([]) == {}

    def test_all_models_receive_query(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("a", tmp_path)
            s.load("b", tmp_path)
        with patch("app.swarm.generate_response", return_value="resp"):
            results = s.generate_all([{"role": "user", "content": "hi"}])
        assert set(results.keys()) == {"a", "b"}
        assert all(v == "resp" for v in results.values())

    def test_all_three_models_get_results(self, tmp_path):
        s = SwarmManager()
        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("x", tmp_path)
            s.load("y", tmp_path)
            s.load("z", tmp_path)
        with patch("app.swarm.generate_response", return_value="r"):
            results = s.generate_all([])
        assert set(results.keys()) == {"x", "y", "z"}

    def test_failing_model_produces_error_string(self, tmp_path):
        s = SwarmManager()

        def _always_fail(m, t, msgs, **kw):
            raise RuntimeError("oom")

        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("bad", tmp_path)

        with patch("app.swarm.generate_response", side_effect=_always_fail):
            results = s.generate_all([])

        assert "bad" in results
        assert "[Error:" in results["bad"]

    def test_one_fails_others_still_succeed(self, tmp_path):
        s = SwarmManager()

        with patch("app.swarm.load_model", side_effect=_mock_load):
            s.load("m1", tmp_path)
            s.load("m2", tmp_path)

        # One model's ID fails, the other succeeds — use a set to track which
        # model objects were created so we can make exactly one fail
        call_ids: list = []
        lock = __import__("threading").Lock()

        def _selective(m, t, msgs, **kw):
            with lock:
                call_ids.append(id(m))
                # Only the very first call fails
                is_first = len(call_ids) == 1
            if is_first:
                raise RuntimeError("first fails")
            return "ok"

        with patch("app.swarm.generate_response", side_effect=_selective):
            results = s.generate_all([])

        assert set(results.keys()) == {"m1", "m2"}
        errors = [v for v in results.values() if v.startswith("[Error:")]
        oks = [v for v in results.values() if v == "ok"]
        assert len(errors) == 1
        assert len(oks) == 1


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestGetSwarm:
    def test_returns_swarm_manager_instance(self):
        from app.swarm import get_swarm
        assert isinstance(get_swarm(), SwarmManager)

    def test_is_singleton(self):
        from app.swarm import get_swarm
        assert get_swarm() is get_swarm()
