"""Unit tests for data/judge.py — multi-dimensional judge evaluation."""

import pytest
from data.judge import (
    AggregationStrategy,
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    JudgeOrchestrator,
    JudgeResult,
    _aggregate,
    _score_cost,
    _score_domain,
    _score_performance,
    _score_quality,
    _score_safety,
    _score_usability,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

class TestScoreQuality:
    def test_empty_answer_returns_zero(self):
        score, flags = _score_quality("What is X?", "")
        assert score == 0.0
        assert "empty answer" in flags

    def test_good_answer_scores_high(self):
        q = "How do I use Hypertension in primary care?"
        a = "Hypertension is sustained elevated blood pressure. Confirm readings and start first-line therapy with monitoring."
        score, flags = _score_quality(q, a)
        assert score > 0.4

    def test_refusal_penalised(self):
        q = "What is hypertension?"
        a = "I cannot provide that information as an AI."
        score, flags = _score_quality(q, a)
        assert "refusal detected" in flags
        score_normal, _ = _score_quality(q, "Hypertension is sustained elevated blood pressure managed with lifestyle and medication.")
        assert score < score_normal

    def test_score_in_range(self):
        score, _ = _score_quality("What?", "This is a moderately detailed answer.")
        assert 0.0 <= score <= 1.0


class TestScoreSafety:
    def test_harmful_content_zero(self):
        score, flags = _score_safety("DROP TABLE users; DELETE FROM accounts;")
        assert score == 0.0
        assert "harmful content detected" in flags

    def test_clean_answer_perfect(self):
        score, flags = _score_safety("Use aspirin 81 mg daily with blood pressure monitoring.")
        assert score == 1.0
        assert flags == []

    def test_refusal_partial(self):
        score, flags = _score_safety("I cannot provide this information as an AI.")
        assert 0.0 < score < 1.0
        assert "refusal / uncertainty" in flags

    def test_rm_rf_blocked(self):
        score, _ = _score_safety("You can run rm -rf / to clean up.")
        assert score == 0.0


class TestScoreCost:
    def test_concise_scores_full(self):
        q = "What is hypertension?"
        a = "Hypertension is sustained elevated blood pressure."
        score, flags = _score_cost(q, a)
        assert score == 1.0
        assert flags == []

    def test_very_verbose_penalised(self):
        q = "What?"
        a = " ".join(["word"] * 500)
        score, flags = _score_cost(q, a)
        assert score < 0.5
        assert any("verbose" in f for f in flags)

    def test_score_in_range(self):
        score, _ = _score_cost("Hi", "Hello there how are you doing today my friend")
        assert 0.0 <= score <= 1.0


class TestScoreDomain:
    def test_no_keywords_returns_one(self):
        score, flags = _score_domain("Any text here.", [])
        assert score == 1.0

    def test_all_keywords_present(self):
        score, _ = _score_domain("Manage hypertension with aspirin and monitoring.", ["hypertension", "aspirin", "monitoring"])
        assert score == 1.0

    def test_none_present_returns_zero(self):
        score, flags = _score_domain("Hello world.", ["hypertension", "aspirin", "monitoring"])
        assert score == 0.0
        assert "low domain vocabulary coverage" in flags

    def test_partial_coverage(self):
        score, _ = _score_domain("Discuss hypertension only.", ["hypertension", "aspirin", "monitoring"])
        assert 0.0 < score < 1.0


class TestScorePerformance:
    def test_code_block_boosts_score(self):
        answer = "Use this plan:\n```\nAspirin 81 mg daily with blood pressure monitoring.\n```"
        score, _ = _score_performance(answer)
        assert score > 0.5

    def test_list_boosts_score(self):
        answer = "Steps:\n1. do A\n2. do B\n3. do C\n4. do D\n5. do E"
        score, _ = _score_performance(answer)
        assert score > 0.5

    def test_very_short_penalised(self):
        score, flags = _score_performance("Yes.")
        assert any("unstructured" in f for f in flags)

    def test_score_in_range(self):
        score, _ = _score_performance("A fairly normal sentence for an answer.")
        assert 0.0 <= score <= 1.0


class TestScoreUsability:
    def test_action_words_boost(self):
        q = "How to manage hypertension?"
        a = "First, confirm elevated readings. Then start lifestyle counseling. Use the following monitoring plan below."
        score, _ = _score_usability(q, a)
        assert score > 0.3

    def test_hedge_words_penalise(self):
        q = "How?"
        a = "Perhaps maybe you could possibly sometimes use this approach."
        _, flags = _score_usability(q, a)
        assert "hedging language detected" in flags

    def test_score_in_range(self):
        score, _ = _score_usability("What is X?", "X is a useful tool for data analysis.")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregate:
    _scores = {"quality": 0.8, "safety": 0.9, "cost": 0.7, "domain": 0.6, "performance": 0.5, "usability": 0.4}

    def test_weighted_average(self):
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.WEIGHTED_AVERAGE, 0.6)
        assert 0.0 <= result <= 1.0

    def test_all_must_pass_returns_min(self):
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.ALL_MUST_PASS, 0.6)
        assert result == min(self._scores.values())

    def test_min_score(self):
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.MIN_SCORE, 0.6)
        assert result == min(self._scores.values())

    def test_max_score(self):
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.MAX_SCORE, 0.6)
        assert result == max(self._scores.values())

    def test_majority_pass(self):
        # 5 of 6 scores >= 0.4
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.MAJORITY_PASS, 0.4)
        assert result > 0.5

    def test_any_pass(self):
        result = _aggregate(self._scores, DEFAULT_WEIGHTS, AggregationStrategy.ANY_PASS, 0.4)
        assert result == 1.0

    def test_any_pass_none(self):
        scores = {"quality": 0.1, "safety": 0.2}
        result = _aggregate(scores, DEFAULT_WEIGHTS, AggregationStrategy.ANY_PASS, 0.5)
        assert result == 0.0

    def test_empty_scores(self):
        result = _aggregate({}, DEFAULT_WEIGHTS, AggregationStrategy.WEIGHTED_AVERAGE, 0.6)
        assert result == 0.0


# ---------------------------------------------------------------------------
# JudgeResult
# ---------------------------------------------------------------------------

class TestJudgeResult:
    def _make_result(self, confidence: float) -> JudgeResult:
        return JudgeResult(
            question="Q",
            answer="A",
            scores={"quality": confidence},
            confidence=confidence,
            strategy="weighted_average",
        )

    def test_gate_pass(self):
        r = self._make_result(0.8)
        assert r.gate(0.6) is True

    def test_gate_fail(self):
        r = self._make_result(0.3)
        assert r.gate(0.6) is False

    def test_confidence_level_high(self):
        assert self._make_result(0.85).confidence_level() == "HIGH"

    def test_confidence_level_medium(self):
        assert self._make_result(0.65).confidence_level() == "MEDIUM"

    def test_confidence_level_low(self):
        assert self._make_result(0.3).confidence_level() == "LOW"

    def test_action_mapping(self):
        assert self._make_result(0.85).action() == "auto_accept"
        assert self._make_result(0.65).action() == "monitor"
        assert self._make_result(0.3).action() == "escalate"


# ---------------------------------------------------------------------------
# JudgeOrchestrator
# ---------------------------------------------------------------------------

class TestJudgeOrchestrator:
    def test_evaluate_returns_result(self):
        orch = JudgeOrchestrator()
        result = orch.evaluate("What is hypertension?", "Hypertension is chronic elevation of blood pressure.")
        assert isinstance(result, JudgeResult)
        assert set(result.scores.keys()) == set(DIMENSIONS)
        assert 0.0 <= result.confidence <= 1.0

    def test_all_dimensions_scored(self):
        orch = JudgeOrchestrator()
        result = orch.evaluate("How?", "Do this step by step.")
        for dim in DIMENSIONS:
            assert dim in result.scores
            assert 0.0 <= result.scores[dim] <= 1.0

    def test_domain_keywords_used(self):
        orch_kw   = JudgeOrchestrator(domain_keywords=["aspirin", "monitoring", "lifestyle"])
        orch_none = JudgeOrchestrator()
        q = "How to manage hypertension?"
        a = "Use lifestyle counseling with aspirin and monitoring."
        r_kw   = orch_kw.evaluate(q, a)
        r_none = orch_none.evaluate(q, a)
        # domain score with matching keywords should be 1.0
        assert r_kw.scores["domain"] == 1.0
        assert r_none.scores["domain"] == 1.0   # no keywords → neutral 1.0

    def test_custom_strategy(self):
        orch = JudgeOrchestrator(strategy=AggregationStrategy.MIN_SCORE)
        result = orch.evaluate("Q?", "A useful answer with hypertension monitoring and aspirin guidance.")
        assert result.strategy == AggregationStrategy.MIN_SCORE.value
        assert result.confidence == min(result.scores.values())

    def test_evaluate_batch(self):
        orch = JudgeOrchestrator()
        pairs = [("Q1?", "A1 answer"), ("Q2?", "A2 answer")]
        results = orch.evaluate_batch(pairs)
        assert len(results) == 2
        assert all(isinstance(r, JudgeResult) for r in results)

    def test_rank_sorted_by_confidence(self):
        orch = JudgeOrchestrator()
        pairs = [
            ("Q?", ""),                    # empty answer → low confidence
            ("Q?", "Aspirin 81 mg daily with blood pressure monitoring; use this example below."),
        ]
        ranked = orch.rank(pairs)
        assert ranked[0][0].confidence >= ranked[1][0].confidence

    def test_harmful_gets_low_confidence(self):
        orch = JudgeOrchestrator()
        result = orch.evaluate("How to fix?", "DROP TABLE users; DELETE FROM accounts;")
        assert result.scores["safety"] == 0.0
        assert result.confidence < 0.5

    def test_strategy_all_must_pass(self):
        orch = JudgeOrchestrator(strategy=AggregationStrategy.ALL_MUST_PASS)
        result = orch.evaluate("Q?", "A.")
        assert result.confidence == min(result.scores.values())
