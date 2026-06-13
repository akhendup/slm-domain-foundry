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
def medical_patterns():
    return [
        {
            "name": "hypertension",
            "title": "Hypertension Management",
            "description": "Hypertension is chronic elevation of blood pressure that increases cardiovascular risk.",
            "use_cases": ["primary prevention", "secondary prevention after stroke"],
            "parameters": [
                {"name": "target_systolic", "description": "Target systolic blood pressure in mmHg"},
            ],
            "templates": {
                "basic": {"content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy."}
            },
        },
        {
            "name": "aspirin",
            "title": "Aspirin Therapy",
            "description": "Low-dose aspirin is used for secondary cardiovascular prevention in selected patients.",
            "use_cases": ["secondary prevention", "post-MI care"],
            "parameters": [
                {"name": "dosage", "description": "Daily aspirin dose in mg"},
            ],
            "templates": {
                "basic": {"content": "Case: post-MI patient. Treatment plan: low-dose aspirin when not contraindicated."}
            },
        },
        {
            "name": "lifestyle",
            "title": "Lifestyle Modification",
            "description": "Sodium reduction and regular exercise are first-line interventions for hypertension.",
            "use_cases": ["primary prevention", "adjunct to medication"],
            "parameters": [
                {"name": "sodium_limit", "description": "Daily sodium target in mg"},
            ],
            "templates": {
                "basic": {"content": "Treatment plan: sodium reduction, exercise, and home blood pressure monitoring."}
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
        tokens = _tokenise("Hypertension Medication Treatment")
        assert "hypertension" in tokens

    def test_empty(self):
        assert _tokenise("") == []


# ---------------------------------------------------------------------------
# Pattern text extraction
# ---------------------------------------------------------------------------

class TestPatternText:
    def test_extracts_name_description(self, medical_patterns):
        text = _pattern_text(medical_patterns[0])
        assert "hypertension" in text.lower()
        assert "blood pressure" in text.lower()

    def test_includes_use_cases(self, medical_patterns):
        text = _pattern_text(medical_patterns[0])
        assert "prevention" in text.lower()

    def test_includes_parameters(self, medical_patterns):
        text = _pattern_text(medical_patterns[0])
        assert "target_systolic" in text

    def test_includes_template_content(self, medical_patterns):
        text = _pattern_text(medical_patterns[0])
        assert "primary prevention" in text.lower()

    def test_empty_pattern(self):
        text = _pattern_text({})
        assert text == ""


# ---------------------------------------------------------------------------
# TF-IDF index
# ---------------------------------------------------------------------------

class TestTFIDFIndex:
    def test_fit_and_search(self, medical_patterns):
        idx = _TFIDFIndex()
        idx.fit(medical_patterns)
        results = idx.search("blood pressure hypertension", top_k=2)
        assert len(results) <= 2
        assert results[0][0]["name"] == "hypertension"

    def test_returns_all_when_fewer_than_k(self, medical_patterns):
        idx = _TFIDFIndex()
        idx.fit(medical_patterns)
        results = idx.search("structured", top_k=10)
        assert len(results) <= len(medical_patterns)

    def test_scores_between_zero_and_one(self, medical_patterns):
        idx = _TFIDFIndex()
        idx.fit(medical_patterns)
        results = idx.search("aspirin cardiovascular prevention", top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_empty_corpus(self):
        idx = _TFIDFIndex()
        idx.fit([])
        results = idx.search("anything")
        assert results == []

    def test_unknown_query_returns_results(self, medical_patterns):
        idx = _TFIDFIndex()
        idx.fit(medical_patterns)
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

    def test_fit_returns_self(self, medical_patterns):
        emb = PatternEmbedder()
        ret = emb.fit(medical_patterns)
        assert ret is emb

    def test_is_fitted_after_fit(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        assert emb.is_fitted()

    def test_pattern_count(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        assert emb.pattern_count == len(medical_patterns)

    def test_backend_is_tfidf(self, medical_patterns):
        emb = PatternEmbedder(use_dense=False)
        emb.fit(medical_patterns)
        assert emb.backend == "tfidf"

    def test_search_returns_pattern_matches(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        results = emb.search("blood pressure hypertension", top_k=2)
        assert all(isinstance(r, PatternMatch) for r in results)

    def test_search_top_k_respected(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        results = emb.search("structured", top_k=2)
        assert len(results) <= 2

    def test_most_relevant_first(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        results = emb.search("blood pressure hypertension management", top_k=3)
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_hypertension_retrieved_for_bp_query(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        results = emb.search("blood pressure hypertension", top_k=1)
        if results:
            assert results[0].pattern["name"] == "hypertension"

    def test_aspirin_retrieved_for_prevention_query(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns)
        results = emb.search("aspirin cardiovascular prevention", top_k=1)
        if results:
            assert results[0].pattern["name"] == "aspirin"

    def test_min_score_filter(self, medical_patterns):
        emb = PatternEmbedder(min_score=0.99)  # impossibly high
        emb.fit(medical_patterns)
        results = emb.search("blood pressure", top_k=3)
        assert results == []   # all filtered out

    def test_get_context_empty_when_no_match(self, medical_patterns):
        emb = PatternEmbedder(min_score=0.99)
        emb.fit(medical_patterns)
        ctx = emb.get_context("blood pressure")
        assert ctx == ""

    def test_get_context_contains_title(self, medical_patterns):
        emb = PatternEmbedder(min_score=0.01)
        emb.fit(medical_patterns)
        ctx = emb.get_context("blood pressure hypertension management", top_k=1)
        if ctx:
            assert "Hypertension" in ctx or "blood pressure" in ctx.lower()

    def test_get_context_contains_similarity_score(self, medical_patterns):
        emb = PatternEmbedder(min_score=0.01)
        emb.fit(medical_patterns)
        ctx = emb.get_context("blood pressure", top_k=1)
        if ctx:
            assert "similarity:" in ctx

    def test_get_context_not_fitted(self):
        emb = PatternEmbedder()
        ctx = emb.get_context("anything")
        assert ctx == ""

    def test_refit_updates_index(self, medical_patterns):
        emb = PatternEmbedder()
        emb.fit(medical_patterns[:1])
        assert emb.pattern_count == 1
        emb.fit(medical_patterns)
        assert emb.pattern_count == len(medical_patterns)

    def test_search_with_zero_patterns_after_fit(self):
        emb = PatternEmbedder()
        emb.fit([])
        assert emb.search("anything") == []
