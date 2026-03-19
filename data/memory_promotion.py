#!/usr/bin/env python3
"""
Three-tier relevance-scored memory promotion pipeline (Priority 5).

Implements the agenticwfproject memory promotion architecture:
interactions flow from SHORT_TERM → MID_TERM → LONG_TERM based on
a four-dimension relevance score.  Only high-scoring entries are
promoted; low-quality interactions age out.

Tiers
-----
SHORT_TERM : recent interactions (capacity-limited, 1-hour minimum age)
MID_TERM   : promoted patterns with stats (capacity-limited)
LONG_TERM  : archived high-value Q&A (unbounded, serialised to JSONL)

Relevance scoring (from agenticwfproject relevance_scorer.py)
--------------------------------------------------------------
frequency    (30 %) : how often a similar question appears
complexity   (25 %) : richness of the answer (length, structure)
effectiveness(25 %) : whether the answer was approved / used
feedback     (20 %) : explicit user approval signal

Promotion threshold: score ≥ 0.60 (configurable).
Minimum age before promotion: 3 600 s = 1 hour (configurable).

No external LLMs or services required.

Usage
-----
    from data.memory_promotion import PromotionPipeline

    pipeline = PromotionPipeline(long_term_path=Path("memory/long_term.jsonl"))
    pipeline.ingest(record)       # add one interaction
    promoted = pipeline.run()     # promote eligible SHORT → MID → LONG
    stats    = pipeline.stats()
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    SHORT_TERM = "short_term"
    MID_TERM   = "mid_term"
    LONG_TERM  = "long_term"


# ---------------------------------------------------------------------------
# Scored interaction
# ---------------------------------------------------------------------------

@dataclass
class ScoredInteraction:
    """Wraps a raw interaction dict with relevance scoring metadata."""
    record:       Dict[str, Any]      # raw interaction from conversation_memory
    tier:         MemoryTier          = MemoryTier.SHORT_TERM
    ingested_at:  float               = field(default_factory=time.time)
    relevance:    float               = 0.0
    score_detail: Dict[str, float]    = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.record.get("id", "")

    @property
    def question(self) -> str:
        return self.record.get("question", "")

    @property
    def answer(self) -> str:
        return self.record.get("answer", "")

    @property
    def approved(self) -> Optional[bool]:
        return self.record.get("approved")

    @property
    def age_seconds(self) -> float:
        return time.time() - self.ingested_at


# ---------------------------------------------------------------------------
# RelevanceScorer
# ---------------------------------------------------------------------------

class RelevanceScorer:
    """
    Score an interaction on four dimensions.

    frequency    (30 %) : normalised count of similar questions in pool
    complexity   (25 %) : answer richness (length + structure)
    effectiveness(25 %) : approval status + kb_context_used
    feedback     (20 %) : explicit approved=True signal (binary)
    """

    WEIGHTS = {
        "frequency":     0.30,
        "complexity":    0.25,
        "effectiveness": 0.25,
        "feedback":      0.20,
    }

    _CODE_BLOCK = re.compile(r"```[\s\S]*?```")
    _LIST_ITEM  = re.compile(r"^\s*[-*\d]+[.)]\s", re.MULTILINE)

    def score(
        self,
        interaction: ScoredInteraction,
        pool:        List[ScoredInteraction],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute relevance score in [0, 1].

        Parameters
        ----------
        interaction : the interaction to score
        pool        : current SHORT_TERM pool used for frequency counting
        """
        freq  = self._frequency(interaction, pool)
        comp  = self._complexity(interaction.answer)
        effec = self._effectiveness(interaction)
        feed  = self._feedback(interaction)

        detail = {
            "frequency":     round(freq,  4),
            "complexity":    round(comp,  4),
            "effectiveness": round(effec, 4),
            "feedback":      round(feed,  4),
        }
        total = sum(detail[k] * self.WEIGHTS[k] for k in detail)
        return round(total, 4), detail

    # -- dimension scorers --------------------------------------------------

    def _frequency(
        self,
        target: ScoredInteraction,
        pool:   List[ScoredInteraction],
    ) -> float:
        """Jaccard similarity of question terms against pool."""
        if not pool:
            return 0.0
        tgt_words = set(re.findall(r"[a-z]{3,}", target.question.lower()))
        if not tgt_words:
            return 0.0
        similarities: List[float] = []
        for other in pool:
            if other.id == target.id:
                continue
            other_words = set(re.findall(r"[a-z]{3,}", other.question.lower()))
            if not other_words:
                continue
            inter = len(tgt_words & other_words)
            union = len(tgt_words | other_words)
            similarities.append(inter / union if union > 0 else 0.0)
        if not similarities:
            return 0.0
        # Use max similarity (any near-duplicate is enough)
        return min(max(similarities) * 2, 1.0)  # scale up; cap at 1.0

    def _complexity(self, answer: str) -> float:
        """Answer richness: length + code blocks + lists."""
        word_count = len(answer.split())
        has_code   = bool(self._CODE_BLOCK.search(answer))
        has_list   = len(self._LIST_ITEM.findall(answer)) >= 2

        # Base: log-normalised word count (saturates ~200 words → 1.0)
        base = min(math.log1p(word_count) / math.log1p(200), 1.0)
        bonus = 0.15 if has_code else 0.0
        bonus += 0.10 if has_list else 0.0
        return min(base + bonus, 1.0)

    def _effectiveness(self, interaction: ScoredInteraction) -> float:
        """Approved interactions + KB context used signal."""
        score = 0.3  # baseline: any interaction has some signal
        if interaction.record.get("kb_context_used"):
            score += 0.3   # used knowledge library → likely relevant
        if interaction.approved is True:
            score += 0.4
        elif interaction.approved is False:
            score = max(0.0, score - 0.3)
        return min(score, 1.0)

    def _feedback(self, interaction: ScoredInteraction) -> float:
        """Binary: approved=True → 1.0, rejected → 0.0, pending → 0.3."""
        if interaction.approved is True:
            return 1.0
        if interaction.approved is False:
            return 0.0
        return 0.3  # pending


# ---------------------------------------------------------------------------
# PromotionConfig
# ---------------------------------------------------------------------------

@dataclass
class PromotionConfig:
    short_term_capacity:  int   = 500
    mid_term_capacity:    int   = 10_000
    promotion_threshold:  float = 0.60
    min_age_seconds:      float = 3_600.0    # 1 hour
    # For testing, allow override to 0
    min_age_override:     float = -1.0       # -1 = disabled


# ---------------------------------------------------------------------------
# PromotionPipeline
# ---------------------------------------------------------------------------

class PromotionPipeline:
    """
    Manages the three-tier memory promotion pipeline.

    Interactions enter at SHORT_TERM.  On each `run()`, eligible
    SHORT_TERM entries are scored and promoted to MID_TERM (if score ≥
    threshold and age ≥ min_age).  MID_TERM entries are promoted to
    LONG_TERM when MID_TERM is at capacity (oldest-first eviction).
    LONG_TERM is serialised to a JSONL file for persistence.

    Parameters
    ----------
    long_term_path : JSONL file path for LONG_TERM persistence.
                     If None, LONG_TERM is in-memory only.
    config         : PromotionConfig with capacity and threshold settings.
    """

    def __init__(
        self,
        long_term_path: Optional[Path] = None,
        config:         Optional[PromotionConfig] = None,
    ) -> None:
        self._cfg     = config or PromotionConfig()
        self._scorer  = RelevanceScorer()
        self._lt_path = long_term_path

        self._short:  List[ScoredInteraction] = []
        self._mid:    List[ScoredInteraction] = []
        self._long:   List[ScoredInteraction] = []

        if long_term_path and long_term_path.exists():
            self._load_long_term()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, record: Dict[str, Any]) -> ScoredInteraction:
        """
        Add a new raw interaction record to SHORT_TERM.

        If SHORT_TERM is at capacity, the lowest-scoring entry is
        dropped (or promoted to MID_TERM if it qualifies).
        """
        si = ScoredInteraction(record=record)
        # Score immediately
        score, detail = self._scorer.score(si, self._short)
        si.relevance    = score
        si.score_detail = detail

        # Evict if over capacity before adding
        if len(self._short) >= self._cfg.short_term_capacity:
            self._evict_short()

        self._short.append(si)
        return si

    def _evict_short(self) -> None:
        """Remove the lowest-relevance entry from SHORT_TERM."""
        if not self._short:
            return
        self._short.sort(key=lambda x: x.relevance)
        self._short.pop(0)

    # ------------------------------------------------------------------
    # Promotion run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, int]:
        """
        Perform one promotion pass.

        Returns
        -------
        dict with counts: short_promoted, mid_promoted
        """
        short_promoted = self._promote_short_to_mid()
        mid_promoted   = self._promote_mid_to_long()
        return {"short_promoted": short_promoted, "mid_promoted": mid_promoted}

    def _min_age(self) -> float:
        if self._cfg.min_age_override >= 0:
            return self._cfg.min_age_override
        return self._cfg.min_age_seconds

    def _promote_short_to_mid(self) -> int:
        """Promote qualifying SHORT_TERM → MID_TERM."""
        min_age = self._min_age()
        promote: List[ScoredInteraction] = []
        keep:    List[ScoredInteraction] = []

        for si in self._short:
            if (si.age_seconds >= min_age and
                    si.relevance >= self._cfg.promotion_threshold):
                si.tier = MemoryTier.MID_TERM
                promote.append(si)
            else:
                keep.append(si)

        self._short = keep

        # Re-score promoted entries against the larger MID context
        for si in promote:
            score, detail = self._scorer.score(si, self._mid)
            si.relevance    = score
            si.score_detail = detail

        self._mid.extend(promote)

        # Trim MID_TERM if over capacity (oldest first)
        evicted_count = 0
        if len(self._mid) > self._cfg.mid_term_capacity:
            self._mid.sort(key=lambda x: x.ingested_at)
            overflow = len(self._mid) - self._cfg.mid_term_capacity
            evicted  = self._mid[:overflow]
            self._mid = self._mid[overflow:]
            # Evicted MID entries go to LONG_TERM
            for si in evicted:
                si.tier = MemoryTier.LONG_TERM
                self._long.append(si)
            evicted_count += len(evicted)

        return len(promote)

    def _promote_mid_to_long(self) -> int:
        """
        Force-promote high-relevance MID entries to LONG_TERM.
        (Entries below threshold stay in MID until evicted by capacity.)
        """
        promote: List[ScoredInteraction] = []
        keep:    List[ScoredInteraction] = []

        for si in self._mid:
            if si.relevance >= 0.85:  # very high relevance → archive now
                si.tier = MemoryTier.LONG_TERM
                promote.append(si)
            else:
                keep.append(si)

        self._mid  = keep
        self._long.extend(promote)

        if promote and self._lt_path:
            self._append_long_term(promote)

        return len(promote)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_long_term(self, entries: List[ScoredInteraction]) -> None:
        self._lt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lt_path, "a", encoding="utf-8") as fh:
            for si in entries:
                obj = {
                    "id":         si.id,
                    "tier":       si.tier.value,
                    "relevance":  si.relevance,
                    "score_detail": si.score_detail,
                    "record":     si.record,
                }
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_long_term(self) -> None:
        for line in self._lt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                si  = ScoredInteraction(
                    record=obj.get("record", {}),
                    tier=MemoryTier.LONG_TERM,
                    relevance=obj.get("relevance", 0.0),
                    score_detail=obj.get("score_detail", {}),
                )
                self._long.append(si)
            except (json.JSONDecodeError, KeyError):
                pass

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_tier(self, tier: MemoryTier) -> List[ScoredInteraction]:
        """Return all entries in a given tier."""
        if tier == MemoryTier.SHORT_TERM:
            return list(self._short)
        if tier == MemoryTier.MID_TERM:
            return list(self._mid)
        return list(self._long)

    def search_long_term(self, query: str, top_k: int = 5) -> List[ScoredInteraction]:
        """
        Simple keyword search over LONG_TERM entries.
        Returns top-K by relevance score among those that match any query term.
        """
        terms = set(re.findall(r"[a-z]{3,}", query.lower()))
        if not terms:
            return sorted(self._long, key=lambda x: x.relevance, reverse=True)[:top_k]
        hits: List[ScoredInteraction] = []
        for si in self._long:
            text = (si.question + " " + si.answer).lower()
            if any(t in text for t in terms):
                hits.append(si)
        hits.sort(key=lambda x: x.relevance, reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return counts and average relevance per tier."""
        def _avg(lst: List[ScoredInteraction]) -> float:
            return round(sum(x.relevance for x in lst) / len(lst), 4) if lst else 0.0

        return {
            "short_term": {
                "count":       len(self._short),
                "capacity":    self._cfg.short_term_capacity,
                "avg_relevance": _avg(self._short),
            },
            "mid_term": {
                "count":       len(self._mid),
                "capacity":    self._cfg.mid_term_capacity,
                "avg_relevance": _avg(self._mid),
            },
            "long_term": {
                "count":       len(self._long),
                "avg_relevance": _avg(self._long),
            },
            "promotion_threshold": self._cfg.promotion_threshold,
        }
