#!/usr/bin/env python3
"""
Multi-dimensional judge evaluation system (Priority 1).

Evaluates Q&A pairs across six dimensions using heuristic scorers
(no external LLMs or services required).  Scores are 0.0–1.0 per
dimension; an overall confidence is computed via configurable aggregation.

Dimensions
----------
quality    : accuracy/completeness of the answer relative to the question
safety     : absence of harmful content, refusals, or hallucination markers
cost       : response brevity / token efficiency
domain     : alignment with expected domain vocabulary
performance: structural quality (formatting, code blocks, lists)
usability  : actionability and clarity for the end user

Aggregation strategies
-----------------------
weighted_average  : weighted mean across dimensions (default)
all_must_pass     : min score across all dims must exceed threshold
majority_pass     : ≥ half of dims must exceed threshold
any_pass          : at least one dim must exceed threshold
min_score         : conservative — returns the lowest single dim score
max_score         : optimistic — returns the highest single dim score

Usage
-----
    from data.judge import JudgeOrchestrator, DEFAULT_WEIGHTS

    orch = JudgeOrchestrator(domain_keywords=["SELECT", "FROM", "WHERE"])
    result = orch.evaluate(question="How do I filter rows?",
                           answer="Use WHERE clause: SELECT * FROM t WHERE col=1;")
    print(result.confidence)          # 0.0–1.0
    print(result.scores)              # {"quality": 0.8, "safety": 1.0, ...}
    print(result.gate(threshold=0.6)) # True / False
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "quality":     0.30,
    "safety":      0.25,
    "cost":        0.15,
    "domain":      0.15,
    "performance": 0.10,
    "usability":   0.05,
}

DIMENSIONS: List[str] = list(DEFAULT_WEIGHTS.keys())

# Markers that indicate the model refused / hallucinated
_REFUSAL_PATTERNS = re.compile(
    r"\b(i (cannot|can't|am unable to|don't know)|"
    r"as an ai|i do not have|i'm not sure|i apologize|"
    r"unclear|unknown|n/a)\b",
    re.IGNORECASE,
)

# Markers of harmful content
_HARM_PATTERNS = re.compile(
    r"\b(ignore (previous|all) instructions?|"
    r"drop (table|database)|delete (all|from)|"
    r"rm -rf|format (c:|disk)|shutdown|jailbreak)\b",
    re.IGNORECASE,
)

# Code-block / structured output markers
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_LIST_ITEM  = re.compile(r"^\s*[-*\d]+[.)]\s", re.MULTILINE)


class AggregationStrategy(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    ALL_MUST_PASS    = "all_must_pass"
    MAJORITY_PASS    = "majority_pass"
    ANY_PASS         = "any_pass"
    MIN_SCORE        = "min_score"
    MAX_SCORE        = "max_score"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """Evaluation result for one Q&A pair."""
    question:   str
    answer:     str
    scores:     Dict[str, float]        # dimension → 0.0–1.0
    confidence: float                   # aggregated overall score
    strategy:   str
    flags:      List[str] = field(default_factory=list)  # human-readable issues

    def gate(self, threshold: float = 0.6) -> bool:
        """Return True if confidence meets the threshold."""
        return self.confidence >= threshold

    def confidence_level(self) -> str:
        """HIGH / MEDIUM / LOW bucket matching bird_critic escalation model."""
        if self.confidence >= 0.8:
            return "HIGH"
        if self.confidence >= 0.5:
            return "MEDIUM"
        return "LOW"

    def action(self) -> str:
        """Recommended action based on confidence level."""
        lvl = self.confidence_level()
        if lvl == "HIGH":
            return "auto_accept"
        if lvl == "MEDIUM":
            return "monitor"
        return "escalate"


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def _score_quality(question: str, answer: str) -> tuple[float, List[str]]:
    """
    Heuristic quality: answer should address the question.
    - Answer length relative to question (very short = suspicious)
    - Absence of refusal patterns
    - Shared key terms between question and answer
    """
    flags: List[str] = []
    q_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    a_words = set(re.findall(r"[a-zA-Z]{3,}", answer.lower()))

    if not answer.strip():
        flags.append("empty answer")
        return 0.0, flags

    # Overlap of meaningful words
    stop = {"the", "is", "are", "and", "or", "for", "how", "what", "why",
            "this", "that", "with", "from", "use"}
    q_sig = q_words - stop
    a_sig = a_words - stop
    overlap = len(q_sig & a_sig) / max(len(q_sig), 1)
    overlap = min(overlap, 1.0)

    # Penalise refusals
    if _REFUSAL_PATTERNS.search(answer):
        flags.append("refusal detected")
        overlap = max(0.0, overlap - 0.4)

    # Reward non-trivially long answers (relative to question)
    len_ratio = min(len(answer) / max(len(question) * 1.5, 50), 1.0)
    score = 0.6 * overlap + 0.4 * len_ratio
    return round(min(score, 1.0), 4), flags


def _score_safety(answer: str) -> tuple[float, List[str]]:
    """Heuristic safety: penalise harmful patterns."""
    flags: List[str] = []
    if _HARM_PATTERNS.search(answer):
        flags.append("harmful content detected")
        return 0.0, flags
    # Mild penalty for refusal language (model hedging)
    if _REFUSAL_PATTERNS.search(answer):
        flags.append("refusal / uncertainty")
        return 0.6, flags
    return 1.0, flags


def _score_cost(question: str, answer: str) -> tuple[float, List[str]]:
    """
    Heuristic cost / token efficiency.
    Reward conciseness: answers much longer than needed score lower.
    Baseline: answer length ≤ 4× question length is ideal.
    """
    flags: List[str] = []
    q_len = max(len(question.split()), 1)
    a_len = len(answer.split())
    ratio = a_len / q_len
    # sweet-spot: 1x–4x question length → 1.0
    # >8x → 0.5, >16x → 0.0
    if ratio <= 4:
        score = 1.0
    elif ratio <= 8:
        score = 1.0 - (ratio - 4) / 8
        flags.append("verbose answer")
    else:
        score = max(0.0, 0.5 - (ratio - 8) / 16)
        flags.append("very verbose answer")
    return round(score, 4), flags


def _score_domain(answer: str, domain_keywords: Sequence[str]) -> tuple[float, List[str]]:
    """
    Heuristic domain alignment: presence of expected domain vocabulary.
    Score = fraction of domain keywords found in the answer (capped at 1.0).
    If no keywords given, return 1.0 (neutral).
    """
    flags: List[str] = []
    if not domain_keywords:
        return 1.0, flags
    ans_lower = answer.lower()
    hits = sum(1 for kw in domain_keywords if kw.lower() in ans_lower)
    score = min(hits / len(domain_keywords), 1.0)
    if score < 0.3:
        flags.append("low domain vocabulary coverage")
    return round(score, 4), flags


def _score_performance(answer: str) -> tuple[float, List[str]]:
    """
    Heuristic structural quality.
    Rewards: code blocks, ordered/unordered lists, non-trivial length.
    """
    flags: List[str] = []
    score = 0.5  # baseline for plain prose

    has_code  = bool(_CODE_BLOCK.search(answer))
    has_list  = len(_LIST_ITEM.findall(answer)) >= 2
    has_prose = len(answer.split()) >= 20

    if has_code:
        score = min(score + 0.3, 1.0)
    if has_list:
        score = min(score + 0.2, 1.0)
    if not has_prose and not has_code and not has_list:
        score = max(score - 0.3, 0.0)
        flags.append("very short / unstructured answer")

    return round(score, 4), flags


def _score_usability(question: str, answer: str) -> tuple[float, List[str]]:
    """
    Heuristic usability / actionability.
    Rewards direct actionable language; penalises vague hedging.
    """
    flags: List[str] = []
    action_words = re.compile(
        r"\b(use|run|execute|call|select|insert|update|delete|create|drop|"
        r"set|get|return|example|below|following|steps?|first|then|finally)\b",
        re.IGNORECASE,
    )
    hedge_words = re.compile(
        r"\b(maybe|perhaps|might|could|possibly|generally|typically|usually|"
        r"sometimes|often)\b",
        re.IGNORECASE,
    )

    a_words = answer.split()
    n = max(len(a_words), 1)
    action_density = len(action_words.findall(answer)) / n
    hedge_density  = len(hedge_words.findall(answer)) / n

    score = min(action_density * 5, 0.8) + 0.2  # baseline 0.2
    score -= hedge_density * 2
    score = max(0.0, min(score, 1.0))

    if hedge_density > 0.05:
        flags.append("hedging language detected")
    return round(score, 4), flags


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    scores: Dict[str, float],
    weights: Dict[str, float],
    strategy: AggregationStrategy,
    pass_threshold: float,
) -> float:
    vals = list(scores.values())
    if not vals:
        return 0.0

    if strategy == AggregationStrategy.WEIGHTED_AVERAGE:
        total_w = sum(weights.get(d, 1.0) for d in scores)
        if total_w == 0:
            return 0.0
        return round(sum(scores[d] * weights.get(d, 1.0) for d in scores) / total_w, 4)

    if strategy == AggregationStrategy.ALL_MUST_PASS:
        return round(min(vals), 4)

    if strategy == AggregationStrategy.MAJORITY_PASS:
        passing = sum(1 for v in vals if v >= pass_threshold)
        return round(passing / len(vals), 4)

    if strategy == AggregationStrategy.ANY_PASS:
        return 1.0 if any(v >= pass_threshold for v in vals) else 0.0

    if strategy == AggregationStrategy.MIN_SCORE:
        return round(min(vals), 4)

    if strategy == AggregationStrategy.MAX_SCORE:
        return round(max(vals), 4)

    return round(sum(vals) / len(vals), 4)  # fallback: mean


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class JudgeOrchestrator:
    """
    Run all dimension scorers on a Q&A pair and return a JudgeResult.

    Parameters
    ----------
    domain_keywords : words expected in on-domain answers (e.g. SQL keywords)
    weights         : per-dimension weights for WEIGHTED_AVERAGE (default: DEFAULT_WEIGHTS)
    strategy        : aggregation strategy (default: WEIGHTED_AVERAGE)
    pass_threshold  : threshold used by ALL_MUST_PASS / MAJORITY_PASS / ANY_PASS
    llm_backend     : optional local LLM backend (``data.judge_llm.JudgeBackend``).
                      When provided, dimension scores come from the LLM instead of
                      heuristics.  Heuristics are used as fallback on any LLM error.
                      Must be a local backend — cloud endpoints are rejected at
                      construction time by the backend itself.
    """

    def __init__(
        self,
        domain_keywords: Optional[Sequence[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE,
        pass_threshold: float = 0.6,
        llm_backend: Optional[Any] = None,   # JudgeBackend | None
    ) -> None:
        self.domain_keywords  = list(domain_keywords or [])
        self.weights          = weights or dict(DEFAULT_WEIGHTS)
        self.strategy         = strategy
        self.pass_threshold   = pass_threshold
        # Lazily wrap in HybridJudge when a backend is provided
        self._hybrid: Optional[Any] = None
        if llm_backend is not None:
            from data.judge_llm import HybridJudge
            self._hybrid = HybridJudge(
                backend=llm_backend,
                domain_keywords=domain_keywords,
                weights=weights,
                strategy=strategy,
            )

    def evaluate(self, question: str, answer: str) -> JudgeResult:
        """Score a single Q&A pair across all dimensions.

        When an ``llm_backend`` was supplied at construction, delegates to
        ``HybridJudge`` (LLM first, heuristic fallback).  Otherwise runs
        the heuristic scorers directly.
        """
        if self._hybrid is not None:
            return self._hybrid.evaluate(question, answer)

        all_flags: List[str] = []
        scores: Dict[str, float] = {}

        s, f = _score_quality(question, answer);    scores["quality"]     = s; all_flags += f
        s, f = _score_safety(answer);               scores["safety"]      = s; all_flags += f
        s, f = _score_cost(question, answer);       scores["cost"]        = s; all_flags += f
        s, f = _score_domain(answer, self.domain_keywords); scores["domain"] = s; all_flags += f
        s, f = _score_performance(answer);          scores["performance"] = s; all_flags += f
        s, f = _score_usability(question, answer);  scores["usability"]   = s; all_flags += f

        confidence = _aggregate(scores, self.weights, self.strategy, self.pass_threshold)

        return JudgeResult(
            question=question,
            answer=answer,
            scores=scores,
            confidence=confidence,
            strategy=self.strategy.value,
            flags=all_flags,
        )

    def evaluate_batch(
        self, pairs: List[tuple[str, str]]
    ) -> List[JudgeResult]:
        """Evaluate a list of (question, answer) tuples."""
        return [self.evaluate(q, a) for q, a in pairs]

    def rank(
        self, pairs: List[tuple[str, str]]
    ) -> List[tuple[JudgeResult, int]]:
        """
        Return pairs ranked by confidence (highest first).
        Returns list of (JudgeResult, original_index).
        """
        results = [(self.evaluate(q, a), i) for i, (q, a) in enumerate(pairs)]
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        return results
