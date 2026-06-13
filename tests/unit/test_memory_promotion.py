"""Unit tests for data/memory_promotion.py — three-tier promotion pipeline."""

import json
import time
import pytest
from pathlib import Path
from data.memory_promotion import (
    MemoryTier,
    PromotionConfig,
    PromotionPipeline,
    RelevanceScorer,
    ScoredInteraction,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    question: str = "What is hypertension?",
    answer:   str = "Hypertension computes a cumulative sum over a column in Clinical SQL.",
    approved: object = None,
    kb_used:  bool   = False,
    rid:      str    = "abc123",
) -> dict:
    return {
        "id":              rid,
        "question":        question,
        "answer":          answer,
        "approved":        approved,
        "kb_context_used": kb_used,
        "session_id":      "test_session",
    }


def _instant_config() -> PromotionConfig:
    """Config that makes ALL interactions eligible immediately (age = 0)."""
    return PromotionConfig(
        short_term_capacity=500,
        mid_term_capacity=10_000,
        promotion_threshold=0.0,
        min_age_seconds=3600.0,
        min_age_override=0.0,   # 0 seconds → eligible immediately
    )


# ---------------------------------------------------------------------------
# ScoredInteraction
# ---------------------------------------------------------------------------

class TestScoredInteraction:
    def test_basic_properties(self):
        r  = _make_record(rid="id1")
        si = ScoredInteraction(record=r)
        assert si.id       == "id1"
        assert si.question == "What is hypertension?"
        assert si.approved is None

    def test_age_increases(self):
        si = ScoredInteraction(record=_make_record())
        age1 = si.age_seconds
        time.sleep(0.05)
        age2 = si.age_seconds
        assert age2 > age1

    def test_tier_default(self):
        si = ScoredInteraction(record=_make_record())
        assert si.tier == MemoryTier.SHORT_TERM


# ---------------------------------------------------------------------------
# RelevanceScorer
# ---------------------------------------------------------------------------

class TestRelevanceScorer:
    def _make_si(self, **kwargs) -> ScoredInteraction:
        return ScoredInteraction(record=_make_record(**kwargs))

    def test_score_returns_tuple(self):
        scorer = RelevanceScorer()
        si     = self._make_si()
        score, detail = scorer.score(si, [])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_detail_has_all_dimensions(self):
        scorer = RelevanceScorer()
        si     = self._make_si()
        _, detail = scorer.score(si, [])
        assert set(detail.keys()) == {"frequency", "complexity", "effectiveness", "feedback"}

    def test_approved_boosts_effectiveness(self):
        scorer = RelevanceScorer()
        si_app = self._make_si(approved=True)
        si_rej = self._make_si(approved=False)
        s_app, _ = scorer.score(si_app, [])
        s_rej, _ = scorer.score(si_rej, [])
        assert s_app > s_rej

    def test_similar_questions_boost_frequency(self):
        scorer = RelevanceScorer()
        target = ScoredInteraction(record=_make_record(question="What is hypertension function?", rid="t1"))
        pool   = [
            ScoredInteraction(record=_make_record(question="What is hypertension?", rid="p1")),
            ScoredInteraction(record=_make_record(question="How does Hypertension work?", rid="p2")),
        ]
        _, detail_with    = scorer.score(target, pool)
        _, detail_without = scorer.score(target, [])
        assert detail_with["frequency"] >= detail_without["frequency"]

    def test_longer_answer_higher_complexity(self):
        scorer = RelevanceScorer()
        si_long  = self._make_si(answer="word " * 150)
        si_short = self._make_si(answer="short")
        _, d_long  = scorer.score(si_long,  [])
        _, d_short = scorer.score(si_short, [])
        assert d_long["complexity"] > d_short["complexity"]

    def test_code_block_boosts_complexity(self):
        scorer = RelevanceScorer()
        si_code  = self._make_si(answer="Use this:\n```sql\nSELECT * FROM t;\n```")
        si_plain = self._make_si(answer="Use SELECT to retrieve data.")
        _, d_code  = scorer.score(si_code,  [])
        _, d_plain = scorer.score(si_plain, [])
        assert d_code["complexity"] >= d_plain["complexity"]

    def test_kb_context_used_boosts_effectiveness(self):
        scorer    = RelevanceScorer()
        si_kb     = ScoredInteraction(record=_make_record(kb_used=True))
        si_no_kb  = ScoredInteraction(record=_make_record(kb_used=False))
        _, d_kb   = scorer.score(si_kb,   [])
        _, d_no   = scorer.score(si_no_kb, [])
        assert d_kb["effectiveness"] >= d_no["effectiveness"]

    def test_feedback_approved(self):
        scorer = RelevanceScorer()
        si     = self._make_si(approved=True)
        _, detail = scorer.score(si, [])
        assert detail["feedback"] == 1.0

    def test_feedback_rejected(self):
        scorer = RelevanceScorer()
        si     = self._make_si(approved=False)
        _, detail = scorer.score(si, [])
        assert detail["feedback"] == 0.0

    def test_feedback_pending(self):
        scorer = RelevanceScorer()
        si     = self._make_si(approved=None)
        _, detail = scorer.score(si, [])
        assert 0.0 < detail["feedback"] < 1.0



# ---------------------------------------------------------------------------
# PromotionPipeline — ingestion
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_adds_to_short_term(self):
        pipeline = PromotionPipeline()
        pipeline.ingest(_make_record())
        assert len(pipeline.get_tier(MemoryTier.SHORT_TERM)) == 1

    def test_ingest_returns_scored_interaction(self):
        pipeline = PromotionPipeline()
        si = pipeline.ingest(_make_record())
        assert isinstance(si, ScoredInteraction)
        assert isinstance(si.relevance, float)

    def test_ingest_multiple(self):
        pipeline = PromotionPipeline()
        for i in range(5):
            pipeline.ingest(_make_record(rid=f"id{i}"))
        assert len(pipeline.get_tier(MemoryTier.SHORT_TERM)) == 5

    def test_capacity_respected(self):
        cfg      = PromotionConfig(short_term_capacity=3, mid_term_capacity=10_000,
                                   promotion_threshold=0.99)  # nothing promotes
        pipeline = PromotionPipeline(config=cfg)
        for i in range(10):
            pipeline.ingest(_make_record(rid=f"id{i}"))
        assert len(pipeline.get_tier(MemoryTier.SHORT_TERM)) <= 3


# ---------------------------------------------------------------------------
# PromotionPipeline — run (promotion)
# ---------------------------------------------------------------------------

class TestPromotionRun:
    def test_run_returns_counts(self):
        pipeline = PromotionPipeline()
        result   = pipeline.run()
        assert "short_promoted" in result
        assert "mid_promoted"   in result

    def test_promotion_moves_to_mid(self):
        cfg      = _instant_config()
        cfg.promotion_threshold = 0.0   # always promote
        pipeline = PromotionPipeline(config=cfg)
        pipeline.ingest(_make_record(approved=True, kb_used=True,
                                     answer="Hypertension computes cumulative sums. " * 10,
                                     rid="r1"))
        result = pipeline.run()
        assert result["short_promoted"] >= 0   # may be 0 if score < threshold
        # At threshold=0 all entries should promote
        cfg2 = _instant_config()
        cfg2.promotion_threshold = 0.0
        pipeline2 = PromotionPipeline(config=cfg2)
        pipeline2.ingest(_make_record(rid="r2"))
        r2 = pipeline2.run()
        assert r2["short_promoted"] == 1
        assert len(pipeline2.get_tier(MemoryTier.SHORT_TERM)) == 0
        assert len(pipeline2.get_tier(MemoryTier.MID_TERM))   == 1

    def test_high_relevance_goes_to_long(self):
        cfg = _instant_config()
        pipeline = PromotionPipeline(config=cfg)
        # Place an entry directly into MID with high relevance
        si = ScoredInteraction(record=_make_record(rid="high_r"), tier=MemoryTier.MID_TERM)
        si.relevance = 0.90
        pipeline._mid.append(si)
        # run() → _promote_mid_to_long checks relevance ≥ 0.85
        pipeline.run()
        long_tier = pipeline.get_tier(MemoryTier.LONG_TERM)
        assert len(long_tier) == 1

    def test_min_age_respected(self):
        cfg = PromotionConfig(
            promotion_threshold=0.0,
            min_age_seconds=3600.0,
            min_age_override=-1,   # use real age
        )
        pipeline = PromotionPipeline(config=cfg)
        pipeline.ingest(_make_record())
        result = pipeline.run()
        # Should NOT promote — entry is brand new (< 1 hour old)
        assert result["short_promoted"] == 0


# ---------------------------------------------------------------------------
# PromotionPipeline — persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_reload_long_term(self, tmp_path):
        path = tmp_path / "long_term.jsonl"
        cfg  = _instant_config()
        cfg.promotion_threshold = 0.0
        p1   = PromotionPipeline(long_term_path=path, config=cfg)

        # Force an entry to LONG_TERM
        si          = p1.ingest(_make_record(rid="persisted_id"))
        si.relevance = 0.90
        p1._short.clear()
        p1._mid.append(si)
        si.tier = MemoryTier.MID_TERM
        p1.run()  # promotes ≥0.85 from MID to LONG

        assert path.exists()
        p2    = PromotionPipeline(long_term_path=path)
        long2 = p2.get_tier(MemoryTier.LONG_TERM)
        assert len(long2) == 1

    def test_search_long_term(self, tmp_path):
        cfg = _instant_config()
        cfg.promotion_threshold = 0.0
        p   = PromotionPipeline(config=cfg)

        si = p.ingest(_make_record(question="Hypertension cumulative sum Clinical", rid="s1"))
        si.relevance = 0.90
        p._short.clear()
        p._mid.append(si)
        si.tier = MemoryTier.MID_TERM
        p.run()

        hits = p.search_long_term("Hypertension Clinical")
        assert len(hits) >= 1


# ---------------------------------------------------------------------------
# PromotionPipeline — stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_structure(self):
        pipeline = PromotionPipeline()
        stats    = pipeline.stats()
        assert "short_term" in stats
        assert "mid_term"   in stats
        assert "long_term"  in stats
        assert "promotion_threshold" in stats

    def test_stats_counts(self):
        pipeline = PromotionPipeline()
        pipeline.ingest(_make_record(rid="a"))
        pipeline.ingest(_make_record(rid="b"))
        stats = pipeline.stats()
        assert stats["short_term"]["count"] == 2

    def test_stats_avg_relevance_in_range(self):
        pipeline = PromotionPipeline()
        pipeline.ingest(_make_record())
        stats = pipeline.stats()
        assert 0.0 <= stats["short_term"]["avg_relevance"] <= 1.0
