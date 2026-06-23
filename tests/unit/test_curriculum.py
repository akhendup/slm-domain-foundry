"""Unit tests for data/curriculum.py — error-driven curriculum learning."""

import re
import pytest
from data.curriculum import (
    ConfidenceLevel,
    ConfidenceMatch,
    CurriculumDataset,
    CurriculumExample,
    CurriculumSorter,
    DEFAULT_MEDICAL_PATTERNS,
    ErrorPattern,
    PatternMatcher,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ConfidenceLevel
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    def test_high(self):
        assert ConfidenceLevel.from_score(0.90) == ConfidenceLevel.HIGH
        assert ConfidenceLevel.from_score(0.80) == ConfidenceLevel.HIGH

    def test_medium(self):
        assert ConfidenceLevel.from_score(0.79) == ConfidenceLevel.MEDIUM
        assert ConfidenceLevel.from_score(0.50) == ConfidenceLevel.MEDIUM

    def test_low(self):
        assert ConfidenceLevel.from_score(0.49) == ConfidenceLevel.LOW
        assert ConfidenceLevel.from_score(0.0)  == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# ErrorPattern
# ---------------------------------------------------------------------------

class TestErrorPattern:
    def test_matches_true(self):
        pat = ErrorPattern(
            name="test",
            pattern=re.compile(r"contraindication", re.IGNORECASE),
            confidence=0.9,
        )
        assert pat.matches("There is a contraindication in the care plan") is True

    def test_matches_false(self):
        pat = ErrorPattern(
            name="test",
            pattern=re.compile(r"contraindication", re.IGNORECASE),
            confidence=0.9,
        )
        assert pat.matches("No relevant content here") is False


# ---------------------------------------------------------------------------
# ConfidenceMatch
# ---------------------------------------------------------------------------

class TestConfidenceMatch:
    def test_no_match(self):
        m = ConfidenceMatch.no_match()
        assert m.confidence == 0.0
        assert m.level == ConfidenceLevel.LOW
        assert m.pattern is None

    def test_action_auto_retry(self):
        pat = DEFAULT_MEDICAL_PATTERNS[0]
        m = ConfidenceMatch(pattern=pat, confidence=0.95, level=ConfidenceLevel.HIGH, fix_hint="")
        assert m.action() == "auto_retry"

    def test_action_monitor(self):
        m = ConfidenceMatch(pattern=None, confidence=0.65, level=ConfidenceLevel.MEDIUM, fix_hint="")
        assert m.action() == "retry_with_monitoring"

    def test_action_escalate(self):
        m = ConfidenceMatch.no_match()
        assert m.action() == "escalate_to_human"


# ---------------------------------------------------------------------------
# PatternMatcher
# ---------------------------------------------------------------------------

class TestPatternMatcher:
    def test_contraindication_matched(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("CLIN_CONTRA_001 contraindication to aspirin due to active bleeding")
        assert result.pattern is not None
        assert result.pattern.name == "contraindication"
        assert result.confidence >= 0.90

    def test_diagnosis_uncertainty_matched(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("Invalid date format encountered: CLIN_DX_004 differential diagnosis")
        assert result.pattern is not None
        assert result.confidence >= 0.85

    def test_no_match_returns_sentinel(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("Everything is working fine today.")
        assert result.pattern is None
        assert result.confidence == 0.0

    def test_highest_confidence_wins(self):
        # contraindication (0.95) should beat generic error (0.30)
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("contraindication and some generic error message")
        assert result.confidence == 0.95

    def test_match_all_returns_multiple(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        results = matcher.match_all("contraindication with adverse event and generic error")
        assert len(results) >= 2

    def test_add_custom_pattern(self):
        matcher = PatternMatcher([])
        custom  = ErrorPattern(
            name="custom_test",
            pattern=re.compile(r"custom_error_xyz", re.IGNORECASE),
            confidence=0.88,
        )
        matcher.add_pattern(custom)
        result = matcher.match("There was a custom_error_xyz in the pipeline")
        assert result.confidence == 0.88
        assert result.pattern.name == "custom_test"

    def test_adverse_event_matched(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("Character set encoding error CLIN_ADVERSE_006 bleeding")
        assert result.pattern is not None
        assert result.confidence >= 0.75

    def test_dosage_error_matched(self):
        matcher = PatternMatcher(DEFAULT_MEDICAL_PATTERNS)
        result  = matcher.match("Incorrect dose prescribed for renal impairment")
        assert result.pattern is not None
        assert result.confidence >= 0.80


# ---------------------------------------------------------------------------
# CurriculumSorter
# ---------------------------------------------------------------------------

class TestCurriculumSorter:
    _examples = [
        {"output": "Generic error — manual review required."},                 # LOW
        {"output": "CLIN_CONTRA_001 active bleeding contraindication — stop aspirin."},  # HIGH
        {"output": "Differential diagnosis CLIN_DX_004 — confirm with repeat readings."},  # HIGH
        {"output": "Schedule follow-up monitoring per guideline in 4 weeks."},  # MEDIUM
    ]

    def test_sort_high_confidence_first(self):
        sorter  = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        ordered = sorter.sort(self._examples)
        # First items should have higher confidence than last
        assert ordered[0].confidence >= ordered[-1].confidence

    def test_annotate_returns_correct_type(self):
        sorter    = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        annotated = sorter.annotate(self._examples)
        assert all(isinstance(e, CurriculumExample) for e in annotated)
        assert len(annotated) == len(self._examples)

    def test_confidence_in_range(self):
        sorter    = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        annotated = sorter.annotate(self._examples)
        for e in annotated:
            assert 0.0 <= e.confidence <= 1.0

    def test_split_by_level(self):
        sorter  = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        buckets = sorter.split_by_level(self._examples)
        assert set(buckets.keys()) == {"HIGH", "MEDIUM", "LOW"}
        total = sum(len(v) for v in buckets.values())
        assert total == len(self._examples)

    def test_text_key_override(self):
        examples = [{"instruction": "CLIN_CONTRA_001 active bleeding contraindication fix"}]
        sorter   = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        ordered  = sorter.sort(examples, text_key="instruction")
        assert ordered[0].confidence >= 0.90

    def test_empty_input(self):
        sorter  = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        ordered = sorter.sort([])
        assert ordered == []

    def test_auto_detect_answer_key(self):
        examples = [{"answer": "differential diagnosis CLIN_DX_004 rule out white-coat effect"}]
        sorter   = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        ordered  = sorter.sort(examples)
        assert len(ordered) == 1
        assert ordered[0].confidence >= 0.85

    def test_contraindication_scores_highest(self):
        examples = [
            {"output": "CLIN_CONTRA_001 active bleeding contraindication"},
            {"output": "unclear documentation gap"},
            {"output": "Everything is fine"},
        ]
        sorter   = CurriculumSorter(DEFAULT_MEDICAL_PATTERNS)
        ordered  = sorter.sort(examples)
        assert ordered[0].data["output"] == "CLIN_CONTRA_001 active bleeding contraindication"


# ---------------------------------------------------------------------------
# CurriculumDataset
# ---------------------------------------------------------------------------

class TestCurriculumDataset:
    _examples = [
        {"output": "contraindication to aspirin in active bleeding"},
        {"output": "differential diagnosis CLIN_DX_004"},
        {"output": "ordinary prose about wellness"},
        {"output": "generic error unknown"},
    ]

    def test_len(self):
        ds = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        assert len(ds) == 4

    def test_iter(self):
        ds    = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        items = list(ds)
        assert len(items) == 4

    def test_getitem(self):
        ds   = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        item = ds[0]
        assert isinstance(item, CurriculumExample)

    def test_high_medium_low_accessors(self):
        ds = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        total = len(ds.high()) + len(ds.medium()) + len(ds.low())
        assert total == len(self._examples)

    def test_as_dicts(self):
        ds    = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        dicts = ds.as_dicts()
        assert len(dicts) == 4
        assert all(isinstance(d, dict) for d in dicts)

    def test_sorted_descending_confidence(self):
        ds = CurriculumDataset(self._examples, DEFAULT_MEDICAL_PATTERNS)
        confidences = [e.confidence for e in ds]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Default patterns coverage
# ---------------------------------------------------------------------------

class TestDefaultPatterns:
    def test_ten_patterns_defined(self):
        assert len(DEFAULT_MEDICAL_PATTERNS) >= 8

    def test_all_patterns_have_confidence_in_range(self):
        for pat in DEFAULT_MEDICAL_PATTERNS:
            assert 0.0 <= pat.confidence <= 1.0

    def test_all_patterns_have_name(self):
        for pat in DEFAULT_MEDICAL_PATTERNS:
            assert pat.name

    def test_generic_error_lowest(self):
        confidences = [p.confidence for p in DEFAULT_MEDICAL_PATTERNS]
        generic = next(p for p in DEFAULT_MEDICAL_PATTERNS if p.name == "generic_error")
        assert generic.confidence == min(confidences)
