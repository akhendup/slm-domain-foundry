"""
Extended unit tests for data/knowledge_capture.py.
Covers: load_pattern_for_edit, preview_qa, and edge-cases not in the primary test file.
"""
import pytest

from data.knowledge_capture import (
    delete_from_library,
    form_to_pattern,
    load_library_entries,
    load_pattern_for_edit,
    preview_qa,
    save_to_library,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# load_pattern_for_edit
# ---------------------------------------------------------------------------

class TestLoadPatternForEdit:
    def _save_full_pattern(self, tmp_path):
        """Save a complete pattern and return its slug."""
        form = {
            "title": "EditTest",
            "description": "A function used for testing pattern editing.",
            "category": "analytics",
            "use_cases_text": "Running totals\nCumulative revenue",
            "parameters_text": "value_col: The value column (amount)",
            "sql_example": "SELECT CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) FROM t;",
            "sql_description": "Basic cumulative sum",
            "example_output": "id | total\n1  | 100",
            "common_errors_text": "Missing ORDER BY: Add ORDER BY inside the OVER clause",
            "best_practices": "Always use ORDER BY.",
        }
        pattern = form_to_pattern(form)
        lib_dir = tmp_path / "library"
        save_to_library(pattern, lib_dir)
        return pattern["name"], lib_dir

    def test_returns_dict(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert isinstance(result, dict)

    def test_none_when_missing_slug(self, tmp_path):
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        assert load_pattern_for_edit("nonexistent", lib_dir) is None

    def test_none_when_library_missing(self, tmp_path):
        assert load_pattern_for_edit("any", tmp_path / "no_library") is None

    def test_title_and_description_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert result["title"] == "EditTest"
        assert "testing" in result["description"].lower()

    def test_category_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert result.get("category") == "analytics"

    def test_use_cases_as_text(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        uc_text = result.get("use_cases_text", "")
        assert "Running" in uc_text or "Cumulative" in uc_text

    def test_parameters_as_text(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        params_text = result.get("parameters_text", "")
        assert "value_col" in params_text

    def test_sql_example_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert "CSUM" in result.get("sql_example", "")

    def test_sql_description_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert "cumulative" in result.get("sql_description", "").lower()

    def test_best_practices_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert "ORDER BY" in result.get("best_practices", "")

    def test_errors_as_text(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        errors_text = result.get("common_errors_text", "")
        assert "ORDER BY" in errors_text or "Missing" in errors_text

    def test_example_output_loaded(self, tmp_path):
        slug, lib_dir = self._save_full_pattern(tmp_path)
        result = load_pattern_for_edit(slug, lib_dir)
        assert result.get("example_output") is not None

    def test_minimal_pattern_loads(self, tmp_path):
        pattern = form_to_pattern({"title": "Minimal", "description": "A minimal pattern here."})
        lib_dir = tmp_path / "library"
        save_to_library(pattern, lib_dir)
        result = load_pattern_for_edit(pattern["name"], lib_dir)
        assert result is not None
        assert result["title"] == "Minimal"

    def test_use_cases_empty_when_none(self, tmp_path):
        pattern = form_to_pattern({"title": "NoUC", "description": "No use cases pattern here."})
        lib_dir = tmp_path / "library"
        save_to_library(pattern, lib_dir)
        result = load_pattern_for_edit(pattern["name"], lib_dir)
        assert result.get("use_cases_text", "") == ""

    def test_parameters_empty_when_none(self, tmp_path):
        pattern = form_to_pattern({"title": "NoParams", "description": "No parameters pattern."})
        lib_dir = tmp_path / "library"
        save_to_library(pattern, lib_dir)
        result = load_pattern_for_edit(pattern["name"], lib_dir)
        assert result.get("parameters_text", "") == ""


# ---------------------------------------------------------------------------
# preview_qa
# ---------------------------------------------------------------------------

class TestPreviewQa:
    def _minimal_form(self):
        return {
            "title": "CSUM",
            "description": "Computes a cumulative sum over an ordered window partition.",
        }

    def test_returns_tuple(self):
        qa, mt = preview_qa(self._minimal_form())
        assert isinstance(qa, list)

    def test_qa_pairs_are_tuples(self):
        qa, _ = preview_qa(self._minimal_form())
        for pair in qa:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_generates_at_least_one_pair(self):
        qa, _ = preview_qa(self._minimal_form())
        assert len(qa) >= 1

    def test_multiturn_is_list_or_none(self):
        _, mt = preview_qa(self._minimal_form())
        assert mt is None or isinstance(mt, list)

    def test_with_use_cases(self):
        form = {**self._minimal_form(), "use_cases_text": "Running totals\nCumulative revenue"}
        qa, _ = preview_qa(form)
        assert len(qa) >= 2  # Should have description + use case pairs

    def test_with_parameters(self):
        form = {**self._minimal_form(), "parameters_text": "value_col: The value column (amount)"}
        qa, _ = preview_qa(form)
        assert len(qa) >= 1

    def test_with_sql_example(self):
        form = {
            **self._minimal_form(),
            "sql_example": "SELECT CSUM(x, t) OVER (PARTITION BY id ORDER BY t) FROM tbl;",
            "sql_description": "Running total",
        }
        qa, _ = preview_qa(form)
        qs = [q for q, _ in qa]
        # Should generate SQL-related questions
        assert any("sql" in q.lower() or "example" in q.lower() or "query" in q.lower() for q in qs)

    def test_no_save_side_effects(self, tmp_path):
        """preview_qa must not write any files."""
        form = self._minimal_form()
        import os
        files_before = set(os.listdir(tmp_path))
        preview_qa(form)
        files_after = set(os.listdir(tmp_path))
        assert files_before == files_after

    def test_questions_nonempty(self):
        qa, _ = preview_qa(self._minimal_form())
        for q, a in qa:
            assert q.strip()
            assert a.strip()


# ---------------------------------------------------------------------------
# delete_from_library — edge cases
# ---------------------------------------------------------------------------

class TestDeleteFromLibraryEdgeCases:
    def test_delete_returns_true_when_only_index_entry(self, tmp_path):
        """Even if the YAML file is missing but the index has the entry, delete returns True."""
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        import json
        index = {"entries": {"test_slug": {"title": "Test", "category": "", "created": "", "qa_count": 0, "file": "test_slug.yaml"}}}
        (lib_dir / "_index.json").write_text(json.dumps(index), encoding="utf-8")
        result = delete_from_library("test_slug", lib_dir)
        assert result is True

    def test_delete_nonexistent_slug_returns_false(self, tmp_path):
        lib_dir = tmp_path / "library"
        lib_dir.mkdir()
        assert delete_from_library("ghost_slug", lib_dir) is False

    def test_delete_removes_yaml_file(self, tmp_path):
        lib_dir = tmp_path / "library"
        pattern = form_to_pattern({"title": "ToDelete", "description": "Will be deleted."})
        save_to_library(pattern, lib_dir)
        slug = pattern["name"]
        yaml_path = lib_dir / f"{slug}.yaml"
        assert yaml_path.exists()
        delete_from_library(slug, lib_dir)
        assert not yaml_path.exists()


# ---------------------------------------------------------------------------
# save_to_library — edge cases
# ---------------------------------------------------------------------------

class TestSaveToLibraryEdgeCases:
    def test_save_updates_existing_slug(self, tmp_path):
        lib_dir = tmp_path / "library"
        pattern = form_to_pattern({"title": "UpdateTest", "description": "Initial description here."})
        save_to_library(pattern, lib_dir)

        # Save again with same name/slug but different description
        pattern["description"] = "Updated description content here."
        save_to_library(pattern, lib_dir)

        entries = load_library_entries(lib_dir)
        matching = [e for e in entries if e.get("name") == pattern["name"]]
        # Only one entry should exist for this slug
        assert len(matching) == 1

    def test_save_qa_count_in_index(self, tmp_path):
        lib_dir = tmp_path / "library"
        pattern = form_to_pattern({"title": "QACountTest", "description": "Testing QA count storage."})
        path, qa_count = save_to_library(pattern, lib_dir)
        assert qa_count >= 1

        import json
        index = json.loads((lib_dir / "_index.json").read_text())
        slug = pattern["name"]
        assert index["entries"][slug]["qa_count"] == qa_count
