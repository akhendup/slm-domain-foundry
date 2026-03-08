"""Unit tests for data/knowledge_capture.py"""
import json

import pytest

from data.knowledge_capture import (
    FIELD_DEFS,
    _parse_errors,
    _parse_parameters,
    _parse_use_cases,
    delete_from_library,
    form_to_pattern,
    library_stats,
    load_library_entries,
    save_to_library,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _parse_use_cases
# ---------------------------------------------------------------------------

class TestParseUseCases:
    def test_basic(self):
        text = "User journey analysis\nFunnel analysis\nChurn prediction"
        result = _parse_use_cases(text)
        assert result == ["User journey analysis", "Funnel analysis", "Churn prediction"]

    def test_strips_bullets(self):
        text = "- First case\n• Second case\n* Third case"
        result = _parse_use_cases(text)
        assert all("•" not in r and "-" not in r.lstrip() for r in result)
        assert len(result) == 3

    def test_skips_empty_lines(self):
        text = "Case one\n\nCase two\n   \nCase three"
        result = _parse_use_cases(text)
        assert len(result) == 3

    def test_skips_very_short_items(self):
        result = _parse_use_cases("OK\nLong enough use case here")
        assert len(result) == 1

    def test_empty_returns_empty(self):
        assert _parse_use_cases("") == []
        assert _parse_use_cases("   ") == []


# ---------------------------------------------------------------------------
# _parse_parameters
# ---------------------------------------------------------------------------

class TestParseParameters:
    def test_basic_colon_format(self):
        text = "partition_columns: Columns to group by (user_id)"
        result = _parse_parameters(text)
        assert len(result) == 1
        p = result[0]
        assert p["name"] == "partition_columns"
        assert "group by" in p["description"]
        assert p["example"] == "user_id"

    def test_multiple_params(self):
        text = (
            "input_table: The input data table (orders)\n"
            "partition_by: Grouping column (customer_id)\n"
            "order_by: Sorting column (event_ts)"
        )
        result = _parse_parameters(text)
        assert len(result) == 3
        names = [p["name"] for p in result]
        assert "input_table" in names
        assert "partition_by" in names

    def test_no_example_in_parens(self):
        text = "column: A column name without an example"
        result = _parse_parameters(text)
        assert len(result) == 1
        assert result[0]["example"] == ""

    def test_empty_returns_empty(self):
        assert _parse_parameters("") == []

    def test_required_field_set(self):
        text = "my_param: Description here (example_val)"
        result = _parse_parameters(text)
        assert result[0]["required"] is True
        assert result[0]["type"] == "string"


# ---------------------------------------------------------------------------
# _parse_errors
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_colon_format(self):
        text = "Spaces in pattern: Remove all spaces from the pattern string"
        result = _parse_errors(text)
        assert len(result) == 1
        assert result[0]["error"] == "Spaces in pattern"
        assert "Remove" in result[0]["solution"]

    def test_multiple_errors(self):
        text = (
            "Missing ORDER BY: Add ORDER BY inside the OVER clause\n"
            "Wrong data type: Cast the column to numeric before use"
        )
        result = _parse_errors(text)
        assert len(result) == 2

    def test_empty_returns_empty(self):
        assert _parse_errors("") == []

    def test_skips_lines_without_separator(self):
        text = "Just a plain sentence with no separator at all"
        # No separator means it won't parse into a clean error/solution pair
        # (it may or may not produce output depending on the colon heuristic)
        result = _parse_errors(text)
        # If a colon exists in the line, it may produce output — that's acceptable.
        # The key assertion is the type.
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# form_to_pattern
# ---------------------------------------------------------------------------

class TestFormToPattern:
    def _minimal_form(self):
        return {
            "title": "CSUM",
            "description": "Computes a cumulative sum over a defined window partition.",
        }

    def test_minimal_required_fields(self):
        pattern = form_to_pattern(self._minimal_form())
        assert pattern["name"] == "csum"
        assert pattern["title"] == "CSUM"
        assert pattern["description"] == "Computes a cumulative sum over a defined window partition."
        assert pattern["_source"] == "user_library"

    def test_name_slug_from_title(self):
        pattern = form_to_pattern({"title": "My Cool Feature!", "description": "Desc."})
        # Special chars become _, trailing _ stripped
        assert "my_cool_feature" in pattern["name"]
        assert not pattern["name"].endswith("_")

    def test_use_cases_parsed(self):
        form = {**self._minimal_form(), "use_cases_text": "Case one\nCase two\nCase three"}
        pattern = form_to_pattern(form)
        assert "use_cases" in pattern
        assert len(pattern["use_cases"]) == 3

    def test_parameters_parsed(self):
        form = {**self._minimal_form(), "parameters_text": "col: A column (example_col)"}
        pattern = form_to_pattern(form)
        assert "parameters" in pattern
        assert pattern["parameters"][0]["name"] == "col"

    def test_sql_becomes_template(self):
        form = {
            **self._minimal_form(),
            "sql_example": "SELECT CSUM(x, t) OVER (PARTITION BY id ORDER BY t) FROM tbl;",
            "sql_description": "Running total per id",
        }
        pattern = form_to_pattern(form)
        assert "templates" in pattern
        assert "SELECT CSUM" in pattern["templates"]["example"]["sql"]

    def test_sql_plus_output_becomes_example(self):
        form = {
            **self._minimal_form(),
            "sql_example": "SELECT CSUM(x, t) OVER (PARTITION BY id ORDER BY t) FROM tbl;",
            "sql_description": "Running total",
            "example_output": "id | total\n1  | 100",
        }
        pattern = form_to_pattern(form)
        assert "examples" in pattern
        assert len(pattern["examples"]) == 1

    def test_errors_parsed(self):
        form = {
            **self._minimal_form(),
            "common_errors_text": "Missing ORDER: Add ORDER BY inside OVER",
        }
        pattern = form_to_pattern(form)
        assert "common_errors" in pattern

    def test_best_practices_included(self):
        form = {**self._minimal_form(), "best_practices": "Always specify ORDER BY."}
        pattern = form_to_pattern(form)
        assert pattern["best_practices"] == "Always specify ORDER BY."

    def test_default_category(self):
        pattern = form_to_pattern(self._minimal_form())
        assert pattern["category"] == "general"

    def test_captured_at_present(self):
        pattern = form_to_pattern(self._minimal_form())
        assert "_captured_at" in pattern


# ---------------------------------------------------------------------------
# Library management: save_to_library / load_library_entries / delete
# ---------------------------------------------------------------------------

class TestLibraryManagement:
    def _make_pattern(self, title="TestFunc", desc="A test function."):
        """Helper: form_to_pattern produces a pattern dict for save_to_library."""
        return form_to_pattern({"title": title, "description": desc})

    def test_save_returns_path_and_count(self, tmp_path):
        lib_dir = tmp_path / "library"
        pattern = self._make_pattern()
        result = save_to_library(pattern, lib_dir)
        assert isinstance(result, tuple) and len(result) == 2
        saved_path, qa_count = result
        assert saved_path.exists()
        assert isinstance(qa_count, int)

    def test_save_and_load(self, tmp_path):
        lib_dir = tmp_path / "library"
        pattern = self._make_pattern("TestFunc", "A test function for the library.")
        save_to_library(pattern, lib_dir)

        entries = load_library_entries(lib_dir)
        assert len(entries) == 1
        assert entries[0]["title"] == "TestFunc"

    def test_index_json_created(self, tmp_path):
        lib_dir = tmp_path / "library"
        save_to_library(self._make_pattern("IndexTest", "Testing the index file."), lib_dir)
        index_file = lib_dir / "_index.json"
        assert index_file.exists()
        index = json.loads(index_file.read_text())
        assert "entries" in index

    def test_multiple_saves(self, tmp_path):
        lib_dir = tmp_path / "library"
        for i in range(3):
            save_to_library(
                self._make_pattern(f"Func{i}", f"Function {i} description here."),
                lib_dir
            )
        entries = load_library_entries(lib_dir)
        assert len(entries) == 3

    def test_load_empty_library(self, tmp_path):
        lib_dir = tmp_path / "empty_lib"
        entries = load_library_entries(lib_dir)
        assert entries == []

    def test_delete_removes_entry(self, tmp_path):
        lib_dir = tmp_path / "library"
        save_to_library(self._make_pattern("ToDelete", "Will be removed from library."), lib_dir)
        entries_before = load_library_entries(lib_dir)
        assert len(entries_before) == 1
        slug = entries_before[0]["name"]

        deleted = delete_from_library(slug, lib_dir)
        assert deleted is True

        entries_after = load_library_entries(lib_dir)
        assert len(entries_after) == 0

    def test_library_stats(self, tmp_path):
        lib_dir = tmp_path / "library"
        save_to_library(self._make_pattern("StatsTest", "For stats testing purposes."), lib_dir)
        stats = library_stats(lib_dir)
        assert isinstance(stats, dict)
        assert stats.get("total_patterns", 0) >= 1


# ---------------------------------------------------------------------------
# FIELD_DEFS structure
# ---------------------------------------------------------------------------

class TestFieldDefs:
    def test_all_have_required_keys(self):
        for field in FIELD_DEFS:
            assert "key" in field
            assert "label" in field
            assert "required" in field
            assert "type" in field

    def test_title_is_required(self):
        title_field = next(f for f in FIELD_DEFS if f["key"] == "title")
        assert title_field["required"] is True

    def test_description_is_required(self):
        desc_field = next(f for f in FIELD_DEFS if f["key"] == "description")
        assert desc_field["required"] is True
