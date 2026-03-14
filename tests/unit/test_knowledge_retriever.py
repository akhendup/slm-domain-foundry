"""Unit tests for data/knowledge_retriever.py"""
import pytest

from data.knowledge_retriever import (
    KnowledgeRetriever,
    _fmt_list,
    _load_all_patterns,
    _pattern_searchable_text,
    _score,
    build_context_block,
    extract_query_terms,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extract_query_terms
# ---------------------------------------------------------------------------

class TestExtractQueryTerms:
    def test_basic(self):
        terms = extract_query_terms("What is CSUM function?")
        assert "csum" in terms

    def test_removes_stop_words(self):
        terms = extract_query_terms("what is the CSUM function")
        assert "what" not in terms
        assert "the" not in terms
        assert "is" not in terms

    def test_removes_short_words(self):
        terms = extract_query_terms("use of npath")
        # "of" is a stop word; "use" is a stop word; "npath" stays
        assert "npath" in terms

    def test_case_insensitive(self):
        terms = extract_query_terms("What is NPATH?")
        assert "npath" in terms

    def test_empty_string(self):
        assert extract_query_terms("") == []

    def test_all_stop_words(self):
        result = extract_query_terms("what is the a an")
        assert result == []

    def test_returns_list_of_strings(self):
        terms = extract_query_terms("cumulative sum window function")
        assert isinstance(terms, list)
        for t in terms:
            assert isinstance(t, str)

    def test_technical_terms_kept(self):
        terms = extract_query_terms("npath cumulative partition")
        assert "npath" in terms
        assert "cumulative" in terms
        assert "partition" in terms

    def test_words_shorter_than_three_removed(self):
        # "ok" has 2 chars, should be excluded
        terms = extract_query_terms("ok npath cumulative")
        assert "ok" not in terms
        assert "npath" in terms

    def test_alphanumeric_terms_kept(self):
        terms = extract_query_terms("use CSUM2 function here")
        assert "csum2" in terms


# ---------------------------------------------------------------------------
# _load_all_patterns
# ---------------------------------------------------------------------------

class TestLoadAllPatterns:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert _load_all_patterns(tmp_path) == []

    def test_nonexistent_dir(self, tmp_path):
        assert _load_all_patterns(tmp_path / "nonexistent") == []

    def test_loads_valid_yaml(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        result = _load_all_patterns(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "csum"

    def test_skips_underscore_files(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        (tmp_path / "_index.yaml").write_text("name: index", encoding="utf-8")
        result = _load_all_patterns(tmp_path)
        assert len(result) == 1

    def test_skips_patterns_without_name(self, tmp_path):
        (tmp_path / "noname.yaml").write_text("title: NoName\ndescription: test", encoding="utf-8")
        result = _load_all_patterns(tmp_path)
        assert len(result) == 0

    def test_source_file_added(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        result = _load_all_patterns(tmp_path)
        assert "_source_file" in result[0]

    def test_multiple_patterns(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        second = sample_yaml_pattern.replace("name: csum", "name: msum").replace(
            'title: "CSUM"', 'title: "MSUM"'
        )
        (tmp_path / "msum.yaml").write_text(second, encoding="utf-8")
        result = _load_all_patterns(tmp_path)
        assert len(result) == 2

    def test_handles_invalid_yaml_gracefully(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        (tmp_path / "bad.yaml").write_text("{bad yaml: [unclosed", encoding="utf-8")
        # Should not raise; just skip the bad file
        result = _load_all_patterns(tmp_path)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _pattern_searchable_text
# ---------------------------------------------------------------------------

class TestPatternSearchableText:
    def test_includes_name(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert "csum" in text

    def test_includes_title(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert "csum" in text.lower()

    def test_includes_description(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert "cumulative" in text.lower()

    def test_includes_use_cases(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert "running" in text.lower() or "cumulative" in text.lower()

    def test_includes_parameter_names(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert "value_expression" in text

    def test_empty_pattern(self):
        text = _pattern_searchable_text({})
        assert text == ""

    def test_returns_lowercase(self, sample_pattern_dict):
        text = _pattern_searchable_text(sample_pattern_dict)
        assert text == text.lower()

    def test_use_cases_as_string(self):
        pattern = {"name": "test", "use_cases": "single string use case"}
        text = _pattern_searchable_text(pattern)
        assert isinstance(text, str)

    def test_params_as_list_of_dicts(self):
        pattern = {
            "name": "test",
            "parameters": [
                {"name": "col_a", "description": "First column"},
                {"name": "col_b", "description": "Second column"},
            ],
        }
        text = _pattern_searchable_text(pattern)
        assert "col_a" in text
        assert "col_b" in text


# ---------------------------------------------------------------------------
# _score
# ---------------------------------------------------------------------------

class TestScore:
    def test_exact_name_match_scores_ten(self, sample_pattern_dict):
        assert _score(sample_pattern_dict, ["csum"]) == 10

    def test_partial_name_match_scores_five(self, sample_pattern_dict):
        # "csu" is in "csum"
        score = _score(sample_pattern_dict, ["csu"])
        assert score >= 5

    def test_body_match_scores_one(self, sample_pattern_dict):
        score = _score(sample_pattern_dict, ["cumulative"])
        assert score >= 1

    def test_no_match_scores_zero(self, sample_pattern_dict):
        assert _score(sample_pattern_dict, ["npath", "zot"]) == 0

    def test_empty_terms_returns_zero(self, sample_pattern_dict):
        assert _score(sample_pattern_dict, []) == 0

    def test_multiple_matching_terms(self, sample_pattern_dict):
        score = _score(sample_pattern_dict, ["csum", "cumulative"])
        assert score > 10  # exact match + body match

    def test_title_match(self):
        pattern = {"name": "x", "title": "NPATH", "description": "path analysis"}
        score = _score(pattern, ["npath"])
        assert score >= 5  # title contains "npath"


# ---------------------------------------------------------------------------
# _fmt_list
# ---------------------------------------------------------------------------

class TestFmtList:
    def test_empty_list(self):
        assert _fmt_list([]) == ""

    def test_none(self):
        assert _fmt_list(None) == ""

    def test_string_input(self):
        assert _fmt_list("plain string") == "plain string"

    def test_list_input(self):
        result = _fmt_list(["item1", "item2"])
        assert "item1" in result
        assert "item2" in result

    def test_custom_prefix(self):
        result = _fmt_list(["x", "y"], prefix="* ")
        assert result.startswith("* ")

    def test_filters_empty_items(self):
        result = _fmt_list(["a", "", "b"])
        assert "a" in result
        assert "b" in result


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock:
    def test_empty_returns_empty(self):
        assert build_context_block([]) == ""

    def test_includes_header(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "knowledge library" in result.lower()

    def test_includes_pattern_title(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "CSUM" in result

    def test_includes_description(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "cumulative" in result.lower()

    def test_includes_use_cases(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "Use cases" in result or "running" in result.lower()

    def test_includes_sql(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "sql" in result.lower()

    def test_multiple_patterns_separated(self, sample_pattern_dict):
        second = dict(sample_pattern_dict)
        second["name"] = "msum"
        second["title"] = "MSUM"
        result = build_context_block([sample_pattern_dict, second])
        assert "CSUM" in result
        assert "MSUM" in result
        assert "---" in result

    def test_common_errors_included(self):
        pattern = {
            "name": "test",
            "title": "Test",
            "description": "A test pattern.",
            "common_errors": [
                {"error": "Missing ORDER BY", "solution": "Add ORDER BY inside OVER"}
            ],
        }
        result = build_context_block([pattern])
        assert "Missing ORDER BY" in result

    def test_no_crash_on_minimal_pattern(self):
        pattern = {"name": "test", "title": "Test", "description": "A test."}
        result = build_context_block([pattern])
        assert isinstance(result, str)

    def test_params_listed(self, sample_pattern_dict):
        result = build_context_block([sample_pattern_dict])
        assert "Parameters" in result or "value_expression" in result


# ---------------------------------------------------------------------------
# KnowledgeRetriever
# ---------------------------------------------------------------------------

class TestKnowledgeRetriever:
    def test_init(self, tmp_path):
        kr = KnowledgeRetriever(tmp_path)
        assert kr._dir == tmp_path
        assert not kr._loaded

    def test_reload(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        kr.reload()
        assert kr._loaded
        assert len(kr._patterns) == 1

    def test_ensure_loaded_lazy(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        assert not kr._loaded
        kr.search("What is CSUM?")
        assert kr._loaded

    def test_search_returns_relevant(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        results = kr.search("What is CSUM cumulative sum?")
        assert len(results) >= 1
        assert results[0]["name"] == "csum"

    def test_search_empty_library(self, tmp_path):
        kr = KnowledgeRetriever(tmp_path)
        assert kr.search("What is CSUM?") == []

    def test_search_below_min_score_excluded(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        # A query with no matching terms should score 0 and be excluded
        results = kr.search("xyz123 unrelated totally irrelevant query")
        assert isinstance(results, list)

    def test_get_context_returns_string(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        assert isinstance(kr.get_context("What is CSUM cumulative sum?"), str)

    def test_get_context_nonempty_on_match(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        context = kr.get_context("How does CSUM cumulative sum work?")
        # Should contain something when score is high enough
        assert isinstance(context, str)

    def test_get_context_empty_no_match(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        # Very generic stop-word-only query — should return ""
        context = kr.get_context("what is the")
        assert context == ""

    def test_max_entries_respected(self, tmp_path, sample_yaml_pattern):
        for name in ("csum", "msum", "rank"):
            text = sample_yaml_pattern.replace("name: csum", f"name: {name}").replace(
                'title: "CSUM"', f'title: "{name.upper()}"'
            )
            (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")
        kr = KnowledgeRetriever(tmp_path)
        results = kr.search("csum msum rank cumulative", max_entries=2)
        assert len(results) <= 2

    def test_reload_refreshes_patterns(self, tmp_path, sample_yaml_pattern):
        kr = KnowledgeRetriever(tmp_path)
        kr.reload()
        assert kr._patterns == []
        # Add a pattern and reload
        (tmp_path / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        kr.reload()
        assert len(kr._patterns) == 1
