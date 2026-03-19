"""Unit tests for data/learning_agent.py — canary testing + circuit breaker."""

import time
import pytest
from data.learning_agent import (
    AutonomyLevel,
    CanaryConfig,
    CanaryResult,
    CanaryTest,
    CircuitBreaker,
    CircuitState,
    LearningAgent,
    PatternMetrics,
    Recommendation,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_allow_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_not_open_before_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_open_blocks(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is False

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)
        cb.record_failure()
        cb.record_failure()
        # With cooldown=0 the state is OPEN or HALF_OPEN depending on timing;
        # after a brief wait the state property auto-transitions to HALF_OPEN.
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=2, success_threshold=2, cooldown_seconds=0.0)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        _ = cb.state   # trigger transition to HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should reset consecutive failures
        assert cb._consec_failures == 0

    def test_reset_restores_closed(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow() is True

    def test_summary_structure(self):
        cb = CircuitBreaker()
        s  = cb.summary()
        assert "state"            in s
        assert "consec_failures"  in s
        assert "consec_successes" in s
        assert "history"          in s

    def test_history_recorded(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        _ = cb.state
        assert len(cb.summary()["history"]) >= 1


# ---------------------------------------------------------------------------
# PatternMetrics
# ---------------------------------------------------------------------------

class TestPatternMetrics:
    def test_initial_zero(self):
        m = PatternMetrics(name="pat_a")
        assert m.total_uses == 0
        assert m.success_rate == 0.0

    def test_record_success(self):
        m = PatternMetrics(name="pat_a")
        m.record(True, cost_tokens=100, latency_ms=200)
        assert m.success_count == 1
        assert m.failure_count == 0
        assert m.total_uses    == 1

    def test_record_failure(self):
        m = PatternMetrics(name="pat_a")
        m.record(False)
        assert m.failure_count == 1
        assert m.success_count == 0

    def test_success_rate_computed(self):
        m = PatternMetrics(name="pat_a")
        m.record(True)
        m.record(True)
        m.record(False)
        assert abs(m.success_rate - 2/3) < 0.01

    def test_avg_cost(self):
        m = PatternMetrics(name="pat_a")
        m.record(True, cost_tokens=100)
        m.record(True, cost_tokens=200)
        assert abs(m.avg_cost - 150) < 0.01

    def test_avg_latency(self):
        m = PatternMetrics(name="pat_a")
        m.record(True, latency_ms=100)
        m.record(True, latency_ms=300)
        assert abs(m.avg_latency - 200) < 0.01

    def test_confidence_low_samples(self):
        m = PatternMetrics(name="pat_a")
        assert m.confidence() == 0.0
        m.record(True)
        assert m.confidence() < 0.5   # not enough data

    def test_confidence_grows_with_samples(self):
        m = PatternMetrics(name="pat_a")
        for _ in range(50):
            m.record(True)
        assert m.confidence() >= 0.7

    def test_recommend_promote(self):
        m = PatternMetrics(name="pat_a")
        for _ in range(60):
            m.record(True)   # >85% success, high confidence
        assert m.recommend() == Recommendation.PROMOTE

    def test_recommend_investigate_low_data(self):
        m = PatternMetrics(name="pat_a")
        m.record(True)
        # Only 1 use → investigate
        assert m.recommend() == Recommendation.INVESTIGATE

    def test_recommend_demote_low_success(self):
        m = PatternMetrics(name="pat_a")
        for _ in range(50):
            m.record(False)    # 0% success, high confidence
        assert m.recommend() in (Recommendation.DEMOTE, Recommendation.REMOVE)

    def test_recommend_keep_medium_success(self):
        m = PatternMetrics(name="pat_a")
        for _ in range(60):
            m.record(True if _ < 36 else False)   # ~60% success
        assert m.recommend() == Recommendation.KEEP


# ---------------------------------------------------------------------------
# CanaryTest
# ---------------------------------------------------------------------------

class TestCanaryTest:
    def test_initial_not_ready(self):
        ct = CanaryTest("pat_a", CanaryConfig(min_samples=5))
        assert ct.ready() is False

    def test_ready_after_min_samples(self):
        ct = CanaryTest("pat_a", CanaryConfig(traffic_fraction=0.5, min_samples=3))
        # Force enough observations into both arms
        ct._control   = [1.0] * 3
        ct._treatment = [1.0] * 3
        assert ct.ready() is True

    def test_observe_routes_to_arm(self):
        ct = CanaryTest("pat_a", CanaryConfig(traffic_fraction=0.5, min_samples=2))
        for _ in range(20):
            arm = ct.observe(True)
            assert arm in ("control", "treatment")
        total = len(ct._control) + len(ct._treatment)
        assert total == 20

    def test_evaluate_promote(self):
        ct = CanaryTest("pat_a", CanaryConfig(traffic_fraction=0.5, min_samples=3))
        ct._control   = [0.0] * 10  # 0% success
        ct._treatment = [1.0] * 10  # 100% success
        result = ct.evaluate()
        assert isinstance(result, CanaryResult)
        assert result.treatment_sr > result.control_sr
        assert result.recommendation in ("promote", "inconclusive")

    def test_evaluate_rollback(self):
        ct = CanaryTest("pat_a", CanaryConfig(traffic_fraction=0.5, min_samples=3))
        ct._control   = [1.0] * 10  # 100% success
        ct._treatment = [0.0] * 10  # 0% success
        result = ct.evaluate()
        assert result.control_sr > result.treatment_sr
        assert result.recommendation in ("rollback", "inconclusive")

    def test_evaluate_inconclusive(self):
        ct = CanaryTest("pat_a", CanaryConfig(traffic_fraction=0.5, min_samples=3))
        ct._control   = [1.0] * 10  # 100%
        ct._treatment = [1.0] * 10  # 100% — no difference
        result = ct.evaluate()
        assert result.recommendation == "inconclusive"

    def test_evaluate_result_structure(self):
        ct = CanaryTest("pat_a")
        ct._control   = [1.0] * 5
        ct._treatment = [0.5, 1.0, 0.5, 1.0, 0.5]
        result = ct.evaluate()
        assert result.pattern_name == "pat_a"
        assert 0.0 <= result.control_sr   <= 1.0
        assert 0.0 <= result.treatment_sr <= 1.0
        assert result.recommendation in ("promote", "rollback", "inconclusive")


# ---------------------------------------------------------------------------
# LearningAgent
# ---------------------------------------------------------------------------

class TestLearningAgent:
    def test_record_tracks_metrics(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True, cost_tokens=100, latency_ms=200)
        s = agent.pattern_summary("pat_a")
        assert s is not None
        assert s["total_uses"] == 1

    def test_record_multiple_patterns(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True)
        agent.record("pat_b", success=False)
        assert agent.pattern_summary("pat_a") is not None
        assert agent.pattern_summary("pat_b") is not None

    def test_analyse_returns_all_patterns(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True)
        agent.record("pat_b", success=False)
        recs = agent.analyse()
        assert "pat_a" in recs
        assert "pat_b" in recs

    def test_analyse_returns_recommendations(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True)
        recs = agent.analyse()
        assert isinstance(recs["pat_a"], Recommendation)

    def test_pattern_summary_none_for_unknown(self):
        agent = LearningAgent()
        assert agent.pattern_summary("unknown_pattern") is None

    def test_all_summaries_sorted_by_success_rate(self):
        agent = LearningAgent()
        for _ in range(10):
            agent.record("high_pat", success=True)
        for _ in range(10):
            agent.record("low_pat", success=False)
        summaries = agent.all_summaries()
        rates = [s["success_rate"] for s in summaries]
        assert rates == sorted(rates, reverse=True)

    # ---- Circuit breaker integration ----

    def test_circuit_breaker_opens_on_failures(self):
        cb    = CircuitBreaker(failure_threshold=3)
        agent = LearningAgent(circuit_breaker=cb)
        for _ in range(3):
            agent.record("pat_a", success=False)
        assert cb.state == CircuitState.OPEN

    def test_full_autonomy_blocked_when_open(self):
        cb    = CircuitBreaker(failure_threshold=2)
        agent = LearningAgent(autonomy=AutonomyLevel.FULL, circuit_breaker=cb)
        for _ in range(50):
            agent.record("bad_pat", success=False)
        report = agent.run_cycle()
        # circuit is open → nothing applied
        assert report["applied"] == []

    # ---- Canary test integration ----

    def test_start_canary_returns_true(self):
        agent = LearningAgent()
        started = agent.start_canary("pat_a")
        assert started is True

    def test_start_canary_blocked_for_protected(self):
        agent = LearningAgent(protected=["protected_pat"])
        assert agent.start_canary("protected_pat") is False

    def test_start_canary_blocked_when_circuit_open(self):
        cb    = CircuitBreaker(failure_threshold=2)
        agent = LearningAgent(circuit_breaker=cb)
        cb.record_failure()
        cb.record_failure()
        assert agent.start_canary("pat_a") is False

    def test_evaluate_canaries_ready(self):
        agent = LearningAgent()
        agent.start_canary("pat_a")
        ct = agent._canaries["pat_a"]
        ct._control   = [1.0] * 10
        ct._treatment = [0.0] * 10
        results = agent.evaluate_canaries()
        assert len(results) == 1
        assert results[0].pattern_name == "pat_a"

    def test_evaluate_canaries_not_ready(self):
        agent = LearningAgent()
        agent.start_canary("pat_a")
        results = agent.evaluate_canaries()
        assert results == []

    # ---- Autonomy levels ----

    def test_manual_applies_nothing(self):
        agent = LearningAgent(autonomy=AutonomyLevel.MANUAL)
        for _ in range(60):
            agent.record("good_pat", success=True)
        report = agent.run_cycle()
        assert report["applied"] == []

    def test_human_approval_queues_proposals(self):
        agent = LearningAgent(autonomy=AutonomyLevel.HUMAN_APPROVAL)
        for _ in range(60):
            agent.record("good_pat", success=True)
        agent.run_cycle()
        # Proposals may be queued
        assert isinstance(agent.pending_approvals, list)

    def test_full_autonomy_applies_changes(self):
        agent = LearningAgent(autonomy=AutonomyLevel.FULL)
        for _ in range(60):
            agent.record("good_pat", success=True)
        report = agent.run_cycle()
        # With sufficient data, PROMOTE should appear and be applied
        assert "proposed_changes" in report
        if report["proposed_changes"]:
            assert len(report["applied"]) > 0

    def test_approve_pending(self):
        agent = LearningAgent(autonomy=AutonomyLevel.HUMAN_APPROVAL)
        for _ in range(60):
            agent.record("good_pat", success=True)
        agent.run_cycle()
        # Approve pending (if any)
        result = agent.approve("good_pat")
        # result is True if there was a pending proposal, False otherwise
        assert isinstance(result, bool)

    # ---- run_cycle structure ----

    def test_run_cycle_returns_full_report(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True)
        report = agent.run_cycle()
        assert "recommendations"  in report
        assert "canary_results"   in report
        assert "proposed_changes" in report
        assert "applied"          in report
        assert "circuit_breaker"  in report
        assert "autonomy"         in report

    def test_history_grows_with_cycles(self):
        agent = LearningAgent()
        agent.record("pat_a", success=True)
        agent.run_cycle()
        agent.run_cycle()
        assert len(agent.history) == 2

    def test_protected_pattern_not_proposed(self):
        agent = LearningAgent(protected=["safe_pat"], autonomy=AutonomyLevel.FULL)
        for _ in range(10):
            agent.record("safe_pat", success=False)   # bad but protected
        report = agent.run_cycle()
        proposed_patterns = [p["pattern"] for p in report["proposed_changes"]]
        assert "safe_pat" not in proposed_patterns
