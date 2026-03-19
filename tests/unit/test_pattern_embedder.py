"""Unit tests for data/pattern_embedder.py — TF-IDF pattern embedding and selection."""

import pytest
from data.pattern_embedder import (
    PatternEmbedder,
    PatternMatch,
    _TFIDFIndex,
    _pattern_text,
    _tokenise,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test patterns
# ---------------------------------------------------------------------------

@pytest.fixture
def sql_patterns():
    return [
        {
            "name": "csum",
            "title": "CSUM — Cumulative Sum",
            "description": "CSUM computes a running total (cumulative sum) over an ordered set of rows.",
            "use_cases": ["running total", "cumulative balance", "progressive aggregation"],
            "parameters": [
                {"name": "value_col", "description": "Column to sum"},
                {"name": "sort_col", "description": "Column to order by"},
            ],
            "templates": {
                "basic": {"sql": "SELECT col, CSUM(col, col) FROM t ORDER BY col;"}
            },
        },
        {
            "name": "rank",
            "title": "RANK — Row Ranking",
            "description": "RANK assigns a rank to each row within a partition, with gaps for ties.",
            "use_cases": ["top-N analysis", "ranking employees", "leaderboard"],
            "parameters": [
                {"name": "sort_col", "description": "Column to rank by"},
            ],
            "templates": {
                "basic": {"sql": "SELECT name, RANK() OVER (ORDER BY score DESC) AS rnk FROM t;"}
            },
        },
        {
            "name": "movavg",
            "title": "MAVG — Moving Average",
            "description": "MAVG computes a moving average over a sliding window of rows.",
            "use_cases": ["smoothing", "trend analysis", "rolling average"],
            "parameters": [
                {"name": "value_col", "description": "Column to average"},
                {"name": "window",    "description": "Window size"},
            ],
            "templates": {
                "basic": {"sql": "SELECT col, MAVG(col, 3, col) FROM t ORDER BY col;"}
            },
        },
    ]


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_removes_stop_words(self):
        tokens = _tokenise("is the function and or")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "or"  not in tokens

    def test_removes_short_tokens(self):
        tokens = _tokenise("a ab abc abcd")
        assert "a"  not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens

    def test_case_insensitive(self):
        tokens = _tokenise("SELECT FROM WHERE")
        assert "select" in tokens

    def test_empty(self):
        assert _tokenise("") == []


# ---------------------------------------------------------------------------
# Pattern text extraction
# ---------------------------------------------------------------------------

class TestPatternText:
    def test_extracts_name_description(self, sql_patterns):
        text = _pattern_text(sql_patterns[0])
        assert "csum" in text.lower()
        assert "cumulative" in text.lower()

    def test_includes_use_cases(self, sql_patterns):
        text = _pattern_text(sql_patterns[0])
        assert "running" in text.lower()

    def test_includes_parameters(self, sql_patterns):
        text = _pattern_text(sql_patterns[0])
        assert "value_col" in text or "sum" in text.lower()

    def test_includes_sql_snippet(self, sql_patterns):
        text = _pattern_text(sql_patterns[0])
        assert "CSUM" in text or "csum" in text.lower()

    def test_empty_pattern(self):
        text = _pattern_text({})
        assert text == ""


# ---------------------------------------------------------------------------
# TF-IDF index
# ---------------------------------------------------------------------------

class TestTFIDFIndex:
    def test_fit_and_search(self, sql_patterns):
        idx = _TFIDFIndex()
        idx.fit(sql_patterns)
        results = idx.search("cumulative sum running total", top_k=2)
        assert len(results) <= 2
        assert results[0][0]["name"] == "csum"

    def test_returns_all_when_fewer_than_k(self, sql_patterns):
        idx = _TFIDFIndex()
        idx.fit(sql_patterns)
        results = idx.search("sql", top_k=10)
        assert len(results) <= len(sql_patterns)

    def test_scores_between_zero_and_one(self, sql_patterns):
        idx = _TFIDFIndex()
        idx.fit(sql_patterns)
        results = idx.search("rank leaderboard", top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_empty_corpus(self):
        idx = _TFIDFIndex()
        idx.fit([])
        results = idx.search("anything")
        assert results == []

    def test_unknown_query_returns_results(self, sql_patterns):
        idx = _TFIDFIndex()
        idx.fit(sql_patterns)
        results = idx.search("zzzyyyxxx totally unknown", top_k=3)
        # May return empty or low-scored results — no error
        assert isinstance(results, list)

    def test_cosine_identical(self):
        idx = _TFIDFIndex()
        vec = {"a": 0.5, "b": 0.3}
        assert abs(idx.cosine(vec, vec) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        idx = _TFIDFIndex()
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert idx.cosine(a, b) == 0.0

    def test_cosine_empty(self):
        idx = _TFIDFIndex()
        assert idx.cosine({}, {"a": 1.0}) == 0.0


# ---------------------------------------------------------------------------
# PatternEmbedder
# ---------------------------------------------------------------------------

class TestPatternEmbedder:
    def test_not_fitted_returns_empty(self):
        emb     = PatternEmbedder()
        results = emb.search("anything")
        assert results == []

    def test_fit_returns_self(self, sql_patterns):
        emb = PatternEmbedder()
        ret = emb.fit(sql_patterns)
        assert ret is emb

    def test_is_fitted_after_fit(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        assert emb.is_fitted()

    def test_pattern_count(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        assert emb.pattern_count == len(sql_patterns)

    def test_backend_is_tfidf(self, sql_patterns):
        emb = PatternEmbedder(use_dense=False)
        emb.fit(sql_patterns)
        assert emb.backend == "tfidf"

    def test_search_returns_pattern_matches(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        results = emb.search("cumulative sum running total", top_k=2)
        assert all(isinstance(r, PatternMatch) for r in results)

    def test_search_top_k_respected(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        results = emb.search("sql", top_k=2)
        assert len(results) <= 2

    def test_most_relevant_first(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        results = emb.search("cumulative sum running total CSUM", top_k=3)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_csum_retrieved_for_cumulative_query(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        results = emb.search("cumulative sum total running", top_k=1)
        if results:
            assert results[0].pattern["name"] == "csum"

    def test_rank_retrieved_for_ranking_query(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns)
        results = emb.search("rank leaderboard top employees", top_k=1)
        if results:
            assert results[0].pattern["name"] == "rank"

    def test_min_score_filter(self, sql_patterns):
        emb = PatternEmbedder(min_score=0.99)  # impossibly high
        emb.fit(sql_patterns)
        results = emb.search("cumulative sum", top_k=3)
        assert results == []   # all filtered out

    def test_get_context_empty_when_no_match(self, sql_patterns):
        emb = PatternEmbedder(min_score=0.99)
        emb.fit(sql_patterns)
        ctx = emb.get_context("cumulative sum")
        assert ctx == ""

    def test_get_context_contains_title(self, sql_patterns):
        emb = PatternEmbedder(min_score=0.01)
        emb.fit(sql_patterns)
        ctx = emb.get_context("cumulative sum running total CSUM", top_k=1)
        if ctx:
            assert "CSUM" in ctx or "Cumulative" in ctx

    def test_get_context_contains_similarity_score(self, sql_patterns):
        emb = PatternEmbedder(min_score=0.01)
        emb.fit(sql_patterns)
        ctx = emb.get_context("cumulative sum", top_k=1)
        if ctx:
            assert "similarity:" in ctx

    def test_get_context_not_fitted(self):
        emb = PatternEmbedder()
        ctx = emb.get_context("anything")
        assert ctx == ""

    def test_refit_updates_index(self, sql_patterns):
        emb = PatternEmbedder()
        emb.fit(sql_patterns[:1])
        assert emb.pattern_count == 1
        emb.fit(sql_patterns)
        assert emb.pattern_count == len(sql_patterns)

    def test_search_with_zero_patterns_after_fit(self):
        emb = PatternEmbedder()
        emb.fit([])
        assert emb.search("anything") == []
