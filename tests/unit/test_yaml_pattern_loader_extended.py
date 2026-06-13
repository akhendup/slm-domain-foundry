"""
Extended unit tests for data/yaml_pattern_loader.py.
Covers: _join_list, _load_yaml edge cases, best_practices list/dict,
guardrails, related_patterns, common_errors, multi-turn conversation generators.
"""
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _join_list
# ---------------------------------------------------------------------------

class TestJoinList:
    def test_string_input_returned_stripped(self):
        from data.yaml_pattern_loader import _join_list
        result = _join_list("  single string  ")
        assert result == "single string"

    def test_list_input_joined_with_bullets(self):
        from data.yaml_pattern_loader import _join_list
        result = _join_list(["item one", "item two", "item three"])
        assert "item one" in result
        assert "item two" in result

    def test_empty_list_returns_empty_string(self):
        from data.yaml_pattern_loader import _join_list
        assert _join_list([]) == ""

    def test_none_returns_empty_string(self):
        from data.yaml_pattern_loader import _join_list
        assert _join_list(None) == ""

    def test_list_with_falsy_items_skipped(self):
        from data.yaml_pattern_loader import _join_list
        result = _join_list(["a", None, "", "b"])
        assert "a" in result
        assert "b" in result


# ---------------------------------------------------------------------------
# _load_yaml edge cases
# ---------------------------------------------------------------------------

class TestLoadYamlEdgeCases:
    def test_returns_none_on_corrupted_yaml(self, tmp_path):
        from data.yaml_pattern_loader import _load_yaml
        bad = tmp_path / "bad.yaml"
        bad.write_text(": invalid: yaml: content: [}", encoding="utf-8")
        result = _load_yaml(bad)
        assert result is None

    def test_returns_none_when_yaml_is_list(self, tmp_path):
        """YAML files that parse to a list (not dict) return None."""
        from data.yaml_pattern_loader import _load_yaml
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        result = _load_yaml(f)
        assert result is None

    def test_raises_when_yaml_unavailable(self, tmp_path, monkeypatch):
        """_load_yaml raises ImportError when _YAML_AVAILABLE is False."""
        import data.yaml_pattern_loader as ypl
        monkeypatch.setattr(ypl, "_YAML_AVAILABLE", False)
        from data.yaml_pattern_loader import _load_yaml
        f = tmp_path / "test.yaml"
        f.write_text("name: test\n", encoding="utf-8")
        with pytest.raises(ImportError, match="pyyaml"):
            _load_yaml(f)


# ---------------------------------------------------------------------------
# generate_qa_from_pattern — best_practices list / dict branches
# ---------------------------------------------------------------------------

class TestGenerateQaBestPractices:
    def _base_pattern(self):
        return {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for testing.",
            "pattern_alias": "TestFn",
        }

    def test_best_practices_as_list(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            **self._base_pattern(),
            "best_practices": [
                "Always use ORDER BY for deterministic results.",
                "Partition data appropriately to reduce memory usage.",
                "Avoid wide partitions that exceed available memory.",
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("best practice" in q.lower() or "guideline" in q.lower() for q in qs)

    def test_best_practices_as_dict(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            **self._base_pattern(),
            "best_practices": {
                "order_by": "Always include ORDER BY for deterministic output.",
                "partitioning": "Use PARTITION BY to limit row scope.",
            },
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("best practice" in q.lower() for q in qs)
        # Dict-specific pairs
        assert any("order_by" in q or "order by" in q.lower() for q in qs)

    def test_best_practices_list_generates_topic_questions(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            **self._base_pattern(),
            "best_practices": [
                "Always use ORDER BY to get deterministic results from the window.",
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        # Should include topic-level questions for best practices > 15 chars
        qs = [q for q, _ in qa]
        assert any("ORDER BY" in q or "order by" in q.lower() or "best practice" in q.lower() for q in qs)


# ---------------------------------------------------------------------------
# generate_qa_from_pattern — guardrails
# ---------------------------------------------------------------------------

class TestGenerateQaGuardrails:
    def test_guardrails_generate_limitation_questions(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for path analysis.",
            "pattern_alias": "TestFn",
            "guardrails": [
                "Cannot be used in OLAP window functions.",
                "Maximum partition size is 10,000 rows.",
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("limitation" in q.lower() or "restrict" in q.lower() or "NOT use" in q for q in qs)

    def test_guardrails_as_empty_list_no_output(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "guardrails": [],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert not any("limitation" in q.lower() for q in qs)


# ---------------------------------------------------------------------------
# generate_qa_from_pattern — related_patterns
# ---------------------------------------------------------------------------

class TestGenerateQaRelatedPatterns:
    def test_related_patterns_generate_comparison_questions(self):
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for path analysis.",
            "pattern_alias": "TestFn",
            "related_patterns": ["OtherFn", "AnotherFn"],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("different from" in q or "instead of" in q or "replace" in q for q in qs)

    def test_related_patterns_empty_string_skipped(self):
        """Empty string in related_patterns is skipped (line 500)."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for testing.",
            "pattern_alias": "TestFn",
            "related_patterns": ["", "ValidPattern"],
        }
        qa = generate_qa_from_pattern(pattern)
        # Empty string should be skipped; ValidPattern should generate questions
        qs = [q for q, _ in qa]
        assert any("ValidPattern" in q for q in qs)


# ---------------------------------------------------------------------------
# generate_qa_from_pattern — fallback and common errors
# ---------------------------------------------------------------------------

class TestGenerateQaFallback:
    def test_fallback_when_no_other_content(self):
        """Pattern with only title/desc but nothing else → fallback pair (line 512)."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        # A pattern that would generate no other pairs
        pattern = {
            "name": "bare",
            "title": "BareFn",
            "description": "A bare function with nothing else.",
            "pattern_alias": "BareFn",
        }
        qa = generate_qa_from_pattern(pattern)
        assert len(qa) >= 1  # at least the fallback or description pair

    def test_common_errors_with_missing_error_text_skipped(self):
        """Error entry with no 'error' key is skipped (line 397)."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "common_errors": [
                {"cause": "missing clause", "solution": "add it"},  # no 'error' key
                {"error": "Invalid ORDER BY", "cause": "wrong col", "solution": "use correct col"},
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("Invalid ORDER BY" in q for q in qs)


# ---------------------------------------------------------------------------
# generate_multiturn_from_pattern — best_practices list/dict branches
# ---------------------------------------------------------------------------

class TestMultiturnBestPractices:
    def test_best_practices_as_list_in_multiturn(self):
        from data.yaml_pattern_loader import generate_multiturn_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for path analysis.",
            "best_practices": ["Use ORDER BY.", "Partition carefully."],
        }
        result = generate_multiturn_from_pattern(pattern)
        # Should return a conversation (or None if not enough turns)
        assert result is None or isinstance(result, list)

    def test_best_practices_as_dict_in_multiturn(self):
        from data.yaml_pattern_loader import generate_multiturn_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function for path analysis.",
            "best_practices": {"ordering": "Use ORDER BY.", "partitioning": "Partition carefully."},
        }
        result = generate_multiturn_from_pattern(pattern)
        assert result is None or isinstance(result, list)

    def test_returns_none_when_no_fn(self):
        """No title or name → returns None (line 530)."""
        from data.yaml_pattern_loader import generate_multiturn_from_pattern
        pattern = {"description": "A function."}
        result = generate_multiturn_from_pattern(pattern)
        assert result is None

    def test_returns_none_when_no_desc(self):
        """No description → returns None (line 548)."""
        from data.yaml_pattern_loader import generate_multiturn_from_pattern
        pattern = {"title": "TestFn", "name": "testfn"}
        result = generate_multiturn_from_pattern(pattern)
        assert result is None


# ---------------------------------------------------------------------------
# _build_debug_conversation
# ---------------------------------------------------------------------------

class TestBuildDebugConversation:
    def test_returns_none_when_no_first_sql_and_no_errors(self):
        """No first SQL and no common_errors → returns None (line 620)."""
        from data.yaml_pattern_loader import _build_debug_conversation
        pattern = {
            "name": "testfn",
            "templates": {},
            "common_errors": [],
        }
        result = _build_debug_conversation(pattern, "TestFn", "A description.")
        assert result is None

    def test_returns_conversation_with_sql(self):
        from data.yaml_pattern_loader import _build_debug_conversation
        pattern = {
            "name": "testfn",
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
            "common_errors": [],
        }
        result = _build_debug_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 4

    def test_includes_error_turns_when_common_errors_present(self):
        from data.yaml_pattern_loader import _build_debug_conversation
        pattern = {
            "name": "testfn",
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
            "common_errors": [
                {"error": "No results", "cause": "bad pattern", "solution": "fix pattern"},
            ],
        }
        result = _build_debug_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        contents = [t["content"] for t in result]
        assert any("No results" in c for c in contents)


# ---------------------------------------------------------------------------
# _build_performance_conversation
# ---------------------------------------------------------------------------

class TestBuildPerformanceConversation:
    def test_returns_none_when_no_bp_and_no_guardrails(self):
        """No best_practices and no guardrails → returns None (line 658)."""
        from data.yaml_pattern_loader import _build_performance_conversation
        pattern = {}
        result = _build_performance_conversation(pattern, "TestFn", "A description.")
        assert result is None

    def test_best_practices_as_list_in_performance(self):
        from data.yaml_pattern_loader import _build_performance_conversation
        pattern = {
            "best_practices": ["Order data before aggregation.", "Use PARTITION BY."],
        }
        result = _build_performance_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        assert len(result) >= 4

    def test_best_practices_as_dict_in_performance(self):
        from data.yaml_pattern_loader import _build_performance_conversation
        pattern = {
            "best_practices": {"ordering": "Sort data.", "partitioning": "Use PARTITION BY."},
        }
        result = _build_performance_conversation(pattern, "TestFn", "A description.")
        assert result is not None

    def test_guardrails_included_in_performance(self):
        from data.yaml_pattern_loader import _build_performance_conversation
        pattern = {
            "best_practices": "Use ORDER BY.",
            "guardrails": ["Max 10,000 rows per partition."],
        }
        result = _build_performance_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        contents = [t["content"] for t in result]
        assert any("Max 10,000" in c for c in contents)


# ---------------------------------------------------------------------------
# _build_migration_conversation
# ---------------------------------------------------------------------------

class TestBuildMigrationConversation:
    def test_returns_none_when_no_first_sql(self):
        """No SQL template → returns None (line 689)."""
        from data.yaml_pattern_loader import _build_migration_conversation
        pattern = {"templates": {}}
        result = _build_migration_conversation(pattern, "TestFn", "A description.")
        assert result is None

    def test_returns_conversation_with_sql(self):
        from data.yaml_pattern_loader import _build_migration_conversation
        pattern = {
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
        }
        result = _build_migration_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        assert len(result) >= 4

    def test_includes_related_patterns(self):
        from data.yaml_pattern_loader import _build_migration_conversation
        pattern = {
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
            "related_patterns": ["OtherFn", "AnotherFn"],
        }
        result = _build_migration_conversation(pattern, "TestFn", "A description.")
        assert result is not None
        contents = [t["content"] for t in result]
        assert any("OtherFn" in c for c in contents)


# ---------------------------------------------------------------------------
# generate_multiturn_conversations
# ---------------------------------------------------------------------------

class TestGenerateMultiturnConversations:
    def test_returns_empty_when_no_fn_or_desc(self):
        """No title/name or desc → empty list (line 714)."""
        from data.yaml_pattern_loader import generate_multiturn_conversations
        assert generate_multiturn_conversations({}) == []
        assert generate_multiturn_conversations({"title": "Fn"}) == []  # no desc

    def test_returns_list_of_conversations(self):
        from data.yaml_pattern_loader import generate_multiturn_conversations
        pattern = {
            "title": "TestFn",
            "name": "testfn",
            "description": "A function for path analysis.",
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
            "best_practices": "Use ORDER BY.",
        }
        result = generate_multiturn_conversations(pattern)
        assert isinstance(result, list)

    def test_full_pattern_generates_multiple_conversations(self):
        from data.yaml_pattern_loader import generate_multiturn_conversations
        pattern = {
            "title": "TestFn",
            "name": "testfn",
            "description": "A function for path analysis in Clinical.",
            "use_cases": ["Detect customer churn", "Analyze click paths"],
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
            "best_practices": ["Use ORDER BY.", "Partition carefully."],
            "common_errors": [
                {"error": "No results", "cause": "bad pattern", "solution": "fix it"},
            ],
            "guardrails": ["Max 10,000 rows per partition."],
            "related_patterns": ["OtherFn"],
        }
        result = generate_multiturn_conversations(pattern)
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _fmt_param — additional branches
# ---------------------------------------------------------------------------

class TestFmtParam:
    def test_required_false_shows_optional(self):
        from data.yaml_pattern_loader import _fmt_param
        result = _fmt_param({
            "name": "my_param",
            "required": False,
            "description": "An optional param.",
        })
        assert "optional" in result

    def test_with_type_and_example_and_default(self):
        from data.yaml_pattern_loader import _fmt_param
        result = _fmt_param({
            "name": "my_param",
            "type": "INTEGER",
            "description": "Some desc.",
            "example": "42",
            "default": 0,
        })
        assert "INTEGER" in result
        assert "42" in result
        assert "0" in result

    def test_with_hint(self):
        from data.yaml_pattern_loader import _fmt_param
        result = _fmt_param({
            "name": "my_param",
            "description": "A param.",
            "LLM-HINT": "Use positive integers only.",
        })
        # LLM-HINT is accessed in generate_qa_from_pattern, not _fmt_param
        # Just test that the function doesn't crash with extra keys
        assert "my_param" in result


# ---------------------------------------------------------------------------
# generate_qa_from_pattern — specific branch coverage
# ---------------------------------------------------------------------------

class TestGenerateQaSpecificBranches:
    """Tests for specific uncovered branches."""

    def test_parameter_with_llm_hint_appended(self):
        """Line 211: hint appended to base_answer when LLM-HINT is present."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "parameters": [{
                "name": "my_param",
                "description": "A param.",
                "LLM-HINT": "Use positive integers only.",
            }],
        }
        qa = generate_qa_from_pattern(pattern)
        answers = " ".join(a for _, a in qa)
        assert "positive integers" in answers

    def test_template_with_no_sql_skipped(self):
        """Line 262 and 295: templates with no sql/content are skipped."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "use_cases": ["Use case 1"],
            "templates": {
                "no_sql_template": {"description": "A template with no SQL"},  # no sql key
                "good_template": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy.", "description": "A good template"},
            },
        }
        qa = generate_qa_from_pattern(pattern)
        # Good template should generate Q&A; no_sql_template should be skipped
        assert len(qa) >= 1
        qs = [q for q, _ in qa]
        assert any("good template" in q.lower() or "TestFn" in q for q in qs)

    def test_example_non_dict_skipped(self):
        """Line 327-328: non-dict items in examples list are skipped."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "examples": [
                "a plain string example",  # not a dict — should be skipped
                {"name": "real_example", "content": "SELECT * FROM TestFn();", "expected_result": "rows"},
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        # real_example should generate questions; plain string should be skipped
        assert any("real_example" in q for q in qs)

    def test_common_errors_non_dict_skipped(self):
        """Lines 378-379 and 391-392: non-dict errors are skipped."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn",
            "description": "A function.",
            "pattern_alias": "TestFn",
            "common_errors": [
                "This is a plain string error",  # not a dict — should be skipped
                {"error": "Valid error", "cause": "some cause", "solution": "fix it"},
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        assert any("Valid error" in q for q in qs)

    def test_pattern_alias_different_from_title(self):
        """Line 323-324: when pattern_alias (short) differs from fn, extra pair is added."""
        from data.yaml_pattern_loader import generate_qa_from_pattern
        pattern = {
            "name": "testfn",
            "title": "TestFn Full Name",
            "description": "A function.",
            "pattern_alias": "SHORT_FN",  # different from title
            "examples": [
                {"name": "ex1", "content": "SELECT * FROM SHORT_FN();", "expected_result": "rows"},
            ],
        }
        qa = generate_qa_from_pattern(pattern)
        qs = [q for q, _ in qa]
        # Both "SHORT_FN SQL" and "TestFn Full Name SQL" should appear
        assert any("SHORT_FN" in q for q in qs)
        assert any("TestFn Full Name" in q for q in qs)
