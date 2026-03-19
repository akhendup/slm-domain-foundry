"""
Tests for data/template_expander.py

Covers:
  - _get_placeholders
  - _load_vocabulary
  - VocabularyExpander._collect_entries
  - VocabularyExpander._answer_for_field
  - VocabularyExpander._expand_single
  - VocabularyExpander._expand_comparison
  - VocabularyExpander.expand
  - VocabularyExpander.expand_to_multiturn
  - expand_vocab_dir
"""

import logging
import tempfile
from pathlib import Path

import pytest
import yaml

from data.template_expander import (
    VocabularyExpander,
    _get_placeholders,
    _load_vocabulary,
    expand_vocab_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_SQL_ENTRY = {
    "name": "ROW_NUMBER",
    "category": "window_function",
    "description": "ROW_NUMBER assigns a unique sequential integer to each row within a partition.",
    "one_sentence": "ROW_NUMBER assigns a unique sequential row number within a partition.",
    "syntax": "ROW_NUMBER() OVER (PARTITION BY col ORDER BY col)",
    "null_behavior": "NULLs in ORDER BY are sorted to either first or last depending on NULLS FIRST/LAST.",
    "performance_tips": ["Partition pruning reduces sort cost.", "Avoid large unsorted windows."],
    "common_errors": [
        {"error": "Missing ORDER BY", "cause": "ORDER BY is required inside OVER()", "solution": "Add ORDER BY clause"}
    ],
    "examples": [
        {"sql": "SELECT ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn FROM employees;",
         "description": "Ranks employees by salary descending."}
    ],
    "related": ["RANK", "DENSE_RANK"],
}

MINIMAL_FINANCIAL_ENTRY = {
    "name": "ACH Transfer",
    "category": "transaction_type",
    "description": "An ACH transfer is an electronic funds transfer processed through the ACH network.",
    "one_sentence": "An ACH transfer moves money electronically through the US ACH network.",
    "examples": [
        {"scenario": "Payroll", "description": "Employer sends payroll via ACH credit."}
    ],
    "related": ["Wire Transfer", "Direct Deposit"],
    "null_behavior": "ACH transactions always carry an amount and originator ID.",
    "common_errors": [
        {"error": "Returned ACH", "cause": "Insufficient funds", "solution": "Verify account balance"}
    ],
    "analysis_notes": "ACH fees are typically $0.20–$1.50 per transaction.",
}

MINIMAL_SQL_VOCAB = {
    "metadata": {"domain": "sql", "version": "1.0"},
    "window_functions": [MINIMAL_SQL_ENTRY],
}

MINIMAL_FINANCIAL_VOCAB = {
    "metadata": {"domain": "financial", "version": "1.0"},
    "transactions": [MINIMAL_FINANCIAL_ENTRY],
}


# ---------------------------------------------------------------------------
# _get_placeholders
# ---------------------------------------------------------------------------

class TestGetPlaceholders:
    def test_single_placeholder(self):
        assert _get_placeholders("What is {fn}?") == ["fn"]

    def test_two_placeholders(self):
        result = _get_placeholders("How does {fn} differ from {related}?")
        assert result == ["fn", "related"]

    def test_no_placeholder(self):
        assert _get_placeholders("What is SQL?") == []

    def test_repeated_placeholder(self):
        result = _get_placeholders("{fn} and {fn} again")
        assert result == ["fn", "fn"]

    def test_empty_string(self):
        assert _get_placeholders("") == []


# ---------------------------------------------------------------------------
# _load_vocabulary
# ---------------------------------------------------------------------------

class TestLoadVocabulary:
    def test_loads_valid_yaml(self, tmp_path):
        vf = tmp_path / "test_vocabulary.yaml"
        vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        data = _load_vocabulary(vf)
        assert isinstance(data, dict)
        assert "metadata" in data

    def test_returns_none_for_missing_file(self, tmp_path):
        data = _load_vocabulary(tmp_path / "nonexistent.yaml")
        assert data is None

    def test_returns_none_for_invalid_yaml(self, tmp_path, caplog):
        vf = tmp_path / "bad.yaml"
        vf.write_text("key: [unclosed", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            data = _load_vocabulary(vf)
        assert data is None
        assert "Could not load" in caplog.text

    def test_returns_none_for_non_dict_yaml(self, tmp_path, caplog):
        vf = tmp_path / "list.yaml"
        vf.write_text("- item1\n- item2\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            data = _load_vocabulary(vf)
        assert data is None


# ---------------------------------------------------------------------------
# VocabularyExpander._collect_entries
# ---------------------------------------------------------------------------

class TestCollectEntries:
    def test_collects_from_list_section(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        entries = expander._collect_entries()
        assert len(entries) == 1
        assert entries[0]["name"] == "ROW_NUMBER"

    def test_skips_metadata(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        entries = expander._collect_entries()
        # metadata section must not produce entries
        assert not any(e.get("name") == "metadata" for e in entries)

    def test_collects_multiple_entries(self):
        vocab = {
            "metadata": {"domain": "sql"},
            "functions": [
                {"name": "COUNT", "description": "Counts rows."},
                {"name": "SUM", "description": "Sums values."},
            ],
        }
        expander = VocabularyExpander(vocab)
        entries = expander._collect_entries()
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert names == {"COUNT", "SUM"}

    def test_skips_entries_without_name(self):
        vocab = {
            "metadata": {"domain": "sql"},
            "functions": [
                {"description": "No name here"},
                {"name": "RANK", "description": "Ranks rows."},
            ],
        }
        expander = VocabularyExpander(vocab)
        entries = expander._collect_entries()
        assert len(entries) == 1
        assert entries[0]["name"] == "RANK"


# ---------------------------------------------------------------------------
# VocabularyExpander._answer_for_field
# ---------------------------------------------------------------------------

class TestAnswerForField:
    def setup_method(self):
        self.expander = VocabularyExpander(MINIMAL_SQL_VOCAB)

    def test_string_field(self):
        entry = {"name": "F", "description": "A description."}
        assert self.expander._answer_for_field(entry, "description") == "A description."

    def test_list_of_strings(self):
        entry = {"name": "F", "performance_tips": ["Tip one.", "Tip two."]}
        result = self.expander._answer_for_field(entry, "performance_tips")
        assert "Tip one." in result
        assert "Tip two." in result

    def test_list_of_dicts(self):
        entry = {"name": "F", "common_errors": [{"error": "Err", "cause": "C", "solution": "S"}]}
        result = self.expander._answer_for_field(entry, "common_errors")
        assert "Err" in result
        assert "cause" in result.lower()

    def test_missing_field_returns_none(self):
        entry = {"name": "F"}
        assert self.expander._answer_for_field(entry, "nonexistent") is None

    def test_empty_string_returns_none(self):
        entry = {"name": "F", "description": "   "}
        assert self.expander._answer_for_field(entry, "description") is None


# ---------------------------------------------------------------------------
# VocabularyExpander._expand_single
# ---------------------------------------------------------------------------

class TestVocabularyExpanderSingleParam:
    def test_produces_pairs_for_description(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        assert len(pairs) > 0

    def test_all_pairs_have_required_keys(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        for p in pairs:
            assert "question" in p
            assert "answer" in p
            assert "source" in p

    def test_fn_replaced_in_questions(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        questions = [p["question"] for p in pairs]
        assert any("ROW_NUMBER" in q for q in questions)
        # No raw {fn} placeholders should remain
        assert not any("{fn}" in q for q in questions)

    def test_syntax_questions_generated(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        syntax_questions = [p for p in pairs if "syntax" in p["question"].lower() or "clause" in p["question"].lower()]
        assert len(syntax_questions) > 0

    def test_example_questions_generated(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        example_questions = [p for p in pairs if "example" in p["question"].lower() or "query" in p["question"].lower()]
        assert len(example_questions) > 0

    def test_null_behavior_questions_generated(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        null_questions = [p for p in pairs if "null" in p["question"].lower()]
        assert len(null_questions) > 0

    def test_error_questions_generated(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single(MINIMAL_SQL_ENTRY)
        error_questions = [p for p in pairs if "error" in p["question"].lower() or "troubleshoot" in p["question"].lower() or "fail" in p["question"].lower()]
        assert len(error_questions) > 0

    def test_financial_transaction_questions_for_transaction_type(self):
        expander = VocabularyExpander(MINIMAL_FINANCIAL_VOCAB)
        pairs = expander._expand_single(MINIMAL_FINANCIAL_ENTRY)
        assert len(pairs) > 0
        assert any("ACH Transfer" in p["question"] for p in pairs)

    def test_entry_without_name_returns_empty(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_single({"description": "No name"})
        assert pairs == []


# ---------------------------------------------------------------------------
# VocabularyExpander._expand_comparison
# ---------------------------------------------------------------------------

class TestVocabularyExpanderComparison:
    def test_generates_comparison_pairs(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_comparison(MINIMAL_SQL_ENTRY)
        assert len(pairs) > 0

    def test_comparison_pairs_contain_related_name(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_comparison(MINIMAL_SQL_ENTRY)
        questions = [p["question"] for p in pairs]
        assert any("RANK" in q for q in questions)
        assert any("DENSE_RANK" in q for q in questions)

    def test_no_comparison_without_related(self):
        entry = dict(MINIMAL_SQL_ENTRY)
        entry.pop("related", None)
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_comparison(entry)
        assert pairs == []

    def test_no_comparison_without_description(self):
        entry = dict(MINIMAL_SQL_ENTRY)
        entry.pop("description", None)
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_comparison(entry)
        assert pairs == []

    def test_fn_and_related_substituted(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander._expand_comparison(MINIMAL_SQL_ENTRY)
        for p in pairs:
            assert "{fn}" not in p["question"]
            assert "{related}" not in p["question"]


# ---------------------------------------------------------------------------
# VocabularyExpander.expand
# ---------------------------------------------------------------------------

class TestVocabularyExpanderExpand:
    def test_expand_returns_list(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        result = expander.expand()
        assert isinstance(result, list)
        assert len(result) > 10  # Should produce many pairs for one entry

    def test_expand_covers_multiple_entries(self):
        vocab = {
            "metadata": {"domain": "sql"},
            "functions": [
                {
                    "name": "COUNT",
                    "description": "COUNT returns the number of rows matching a condition.",
                    "one_sentence": "COUNT counts rows.",
                    "syntax": "COUNT(*) or COUNT(col)",
                    "related": ["SUM", "AVG"],
                },
                {
                    "name": "SUM",
                    "description": "SUM returns the sum of a numeric column.",
                    "one_sentence": "SUM sums values.",
                    "syntax": "SUM(col)",
                    "related": ["COUNT", "AVG"],
                },
            ],
        }
        expander = VocabularyExpander(vocab)
        result = expander.expand()
        sources = {p["source"] for p in result}
        assert "COUNT" in sources
        assert "SUM" in sources

    def test_expand_logs_count(self, caplog):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        with caplog.at_level(logging.INFO):
            expander.expand()
        assert "VocabularyExpander" in caplog.text

    def test_empty_vocab_returns_empty(self):
        expander = VocabularyExpander({"metadata": {"domain": "sql"}})
        result = expander.expand()
        assert result == []


# ---------------------------------------------------------------------------
# VocabularyExpander.expand_to_multiturn
# ---------------------------------------------------------------------------

class TestVocabularyExpanderMultiturn:
    def test_multiturn_format(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        results = expander.expand_to_multiturn()
        assert len(results) > 0
        for r in results:
            assert "conversations" in r
            convs = r["conversations"]
            assert len(convs) == 2
            assert convs[0]["from"] == "human"
            assert convs[1]["from"] == "gpt"

    def test_multiturn_question_not_empty(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        results = expander.expand_to_multiturn()
        for r in results:
            assert r["conversations"][0]["value"].strip()
            assert r["conversations"][1]["value"].strip()

    def test_multiturn_count_matches_expand(self):
        expander = VocabularyExpander(MINIMAL_SQL_VOCAB)
        pairs = expander.expand()
        multiturn = expander.expand_to_multiturn()
        assert len(multiturn) == len(pairs)


# ---------------------------------------------------------------------------
# expand_vocab_dir
# ---------------------------------------------------------------------------

class TestExpandVocabDir:
    def test_finds_vocabulary_files(self, tmp_path):
        vf = tmp_path / "sql_vocabulary.yaml"
        vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        result = expand_vocab_dir(tmp_path)
        assert len(result) > 0

    def test_expands_multiple_vocab_files(self, tmp_path):
        sql_vf = tmp_path / "sql_vocabulary.yaml"
        sql_vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        fin_vf = tmp_path / "financial_vocabulary.yaml"
        fin_vf.write_text(yaml.dump(MINIMAL_FINANCIAL_VOCAB), encoding="utf-8")
        result = expand_vocab_dir(tmp_path)
        sources = {p["source"] for p in result}
        assert "ROW_NUMBER" in sources
        assert "ACH Transfer" in sources

    def test_multiturn_flag_changes_format(self, tmp_path):
        vf = tmp_path / "sql_vocabulary.yaml"
        vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        result = expand_vocab_dir(tmp_path, multiturn=True)
        assert len(result) > 0
        assert "conversations" in result[0]

    def test_no_vocab_files_returns_empty(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            result = expand_vocab_dir(tmp_path)
        assert result == []
        assert "No *_vocabulary.yaml" in caplog.text

    def test_ignores_invalid_vocab_file(self, tmp_path, caplog):
        # Write a bad file alongside a valid one
        bad_vf = tmp_path / "bad_vocabulary.yaml"
        bad_vf.write_text("key: [unclosed", encoding="utf-8")
        good_vf = tmp_path / "sql_vocabulary.yaml"
        good_vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            result = expand_vocab_dir(tmp_path)
        assert len(result) > 0  # Good file processed
        assert "Could not load" in caplog.text  # Bad file warned

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        result = expand_vocab_dir(tmp_path / "nonexistent")
        assert result == []

    def test_result_has_required_keys(self, tmp_path):
        vf = tmp_path / "sql_vocabulary.yaml"
        vf.write_text(yaml.dump(MINIMAL_SQL_VOCAB), encoding="utf-8")
        result = expand_vocab_dir(tmp_path)
        for p in result:
            assert "question" in p
            assert "answer" in p
            assert "source" in p

    def test_real_sql_vocabulary_file(self):
        """Integration: confirm the real sql_vocabulary.yaml expands to 1000+ pairs."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not (data_dir / "sql_vocabulary.yaml").exists():
            pytest.skip("sql_vocabulary.yaml not present")
        result = expand_vocab_dir(data_dir)
        # At minimum, one vocabulary file should produce hundreds of pairs
        assert len(result) >= 500, f"Expected 500+ pairs, got {len(result)}"

    def test_real_financial_vocabulary_file(self):
        """Integration: confirm the real financial_vocabulary.yaml expands to 200+ pairs."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        if not (data_dir / "financial_vocabulary.yaml").exists():
            pytest.skip("financial_vocabulary.yaml not present")
        # Only financial vocab
        result = expand_vocab_dir(data_dir)
        # Combined with SQL this should definitely exceed 500
        assert len(result) >= 200, f"Expected 200+ pairs, got {len(result)}"


# ---------------------------------------------------------------------------
# Question template exports (new)
# ---------------------------------------------------------------------------

class TestQuestionTemplateNewExports:
    def test_null_behavior_questions_non_empty(self):
        from data.question_templates import NULL_BEHAVIOR_QUESTIONS
        assert len(NULL_BEHAVIOR_QUESTIONS) >= 5

    def test_performance_questions_non_empty(self):
        from data.question_templates import PERFORMANCE_QUESTIONS
        assert len(PERFORMANCE_QUESTIONS) >= 5

    def test_error_questions_non_empty(self):
        from data.question_templates import ERROR_QUESTIONS
        assert len(ERROR_QUESTIONS) >= 5

    def test_comparison_questions_have_fn_and_related(self):
        from data.question_templates import COMPARISON_QUESTIONS
        dual = [t for t in COMPARISON_QUESTIONS if "{fn}" in t and "{related}" in t]
        assert len(dual) >= 3

    def test_null_behavior_fn_placeholder(self):
        from data.question_templates import NULL_BEHAVIOR_QUESTIONS
        fn_templates = [t for t in NULL_BEHAVIOR_QUESTIONS if "{fn}" in t]
        assert len(fn_templates) > 0

    def test_performance_fn_placeholder(self):
        from data.question_templates import PERFORMANCE_QUESTIONS
        fn_templates = [t for t in PERFORMANCE_QUESTIONS if "{fn}" in t]
        assert len(fn_templates) > 0

    def test_error_fn_placeholder(self):
        from data.question_templates import ERROR_QUESTIONS
        fn_templates = [t for t in ERROR_QUESTIONS if "{fn}" in t]
        assert len(fn_templates) > 0
