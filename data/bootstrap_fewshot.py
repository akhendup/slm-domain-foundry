#!/usr/bin/env python3
"""
Teacher-Student BootstrapFewShot for training data quality (Priority 4).

Implements DSPy-style few-shot demonstration selection without any external
LLMs or services.  A heuristic "teacher" scorer (built on top of the judge
system in data/judge.py) selects the top-K highest-quality examples from a
candidate pool.  These demonstrations are then used as the fine-tuning
training set for the "student" model.

Workflow
--------
1. Load a pool of candidate (question, answer) pairs.
2. Score each pair with JudgeOrchestrator (heuristic — no LLM needed).
3. Select the top-K by confidence score.
4. Optionally format as ShareGPT JSONL for direct use in training.

Usage
-----
    from data.bootstrap_fewshot import BootstrapFewShot

    candidates = [
        {"question": "What is CSUM?", "answer": "CSUM computes cumulative sums..."},
        ...
    ]
    bfs = BootstrapFewShot(top_k=5)
    demos = bfs.select(candidates)
    bfs.save_jsonl(demos, Path("training_data/bootstrap_demos.jsonl"))
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.judge import JudgeOrchestrator, JudgeResult, DEFAULT_WEIGHTS, AggregationStrategy


# ---------------------------------------------------------------------------
# Demonstration record
# ---------------------------------------------------------------------------

@dataclass
class Demonstration:
    """A selected few-shot demonstration with its judge metadata."""
    question:   str
    answer:     str
    confidence: float
    scores:     Dict[str, float]
    rank:       int                    # 1 = best
    flags:      List[str] = field(default_factory=list)

    def to_sharegpt(self) -> Dict[str, Any]:
        """Format as a ShareGPT-style conversation record."""
        return {
            "conversations": [
                {"role": "user",      "content": self.question},
                {"role": "assistant", "content": self.answer},
            ]
        }

    def to_alpaca(self) -> Dict[str, Any]:
        """Format as an Alpaca-style instruction record."""
        return {
            "instruction": self.question,
            "input":       "",
            "output":      self.answer,
        }


# ---------------------------------------------------------------------------
# BootstrapFewShot
# ---------------------------------------------------------------------------

class BootstrapFewShot:
    """
    Select the top-K highest-quality demonstrations from a candidate pool.

    The "teacher" here is the JudgeOrchestrator — a multi-dimensional
    heuristic scorer that needs no external LLM.  This matches loom's
    BootstrapFewShot approach: judge-filtered demonstration selection.

    Parameters
    ----------
    top_k            : number of demonstrations to select
    min_confidence   : minimum judge confidence to include an example
                       (default 0.5 — discards LOW-confidence examples)
    domain_keywords  : forwarded to JudgeOrchestrator for domain scoring
    weights          : judge dimension weights
    strategy         : judge aggregation strategy
    question_key     : dict key for the question field in candidate dicts
    answer_key       : dict key for the answer field in candidate dicts
    """

    def __init__(
        self,
        top_k:            int                          = 5,
        min_confidence:   float                        = 0.5,
        domain_keywords:  Optional[Sequence[str]]     = None,
        weights:          Optional[Dict[str, float]]  = None,
        strategy:         AggregationStrategy         = AggregationStrategy.WEIGHTED_AVERAGE,
        question_key:     str                          = "question",
        answer_key:       str                          = "answer",
    ) -> None:
        self.top_k          = top_k
        self.min_confidence = min_confidence
        self.question_key   = question_key
        self.answer_key     = answer_key
        self._judge         = JudgeOrchestrator(
            domain_keywords=domain_keywords,
            weights=weights,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def _extract_pair(self, candidate: Dict[str, Any]) -> Tuple[str, str]:
        """Extract (question, answer) from a candidate dict."""
        q = str(candidate.get(self.question_key) or
                candidate.get("instruction") or
                candidate.get("prompt") or "")
        a = str(candidate.get(self.answer_key) or
                candidate.get("output") or
                candidate.get("response") or "")
        return q.strip(), a.strip()

    def score_candidates(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], JudgeResult]]:
        """
        Return (candidate, JudgeResult) pairs for all candidates, sorted
        by confidence descending.
        """
        scored: List[Tuple[Dict[str, Any], JudgeResult]] = []
        for c in candidates:
            q, a = self._extract_pair(c)
            if not q or not a:
                continue
            result = self._judge.evaluate(q, a)
            scored.append((c, result))
        scored.sort(key=lambda x: x[1].confidence, reverse=True)
        return scored

    def select(
        self, candidates: List[Dict[str, Any]]
    ) -> List[Demonstration]:
        """
        Score all candidates and return the top-K as Demonstration objects.

        Candidates below `min_confidence` are excluded even if they would
        otherwise rank in the top-K.
        """
        scored = self.score_candidates(candidates)
        demos: List[Demonstration] = []
        rank = 1
        for candidate, result in scored:
            if result.confidence < self.min_confidence:
                continue
            q, a = self._extract_pair(candidate)
            demos.append(Demonstration(
                question=q,
                answer=a,
                confidence=result.confidence,
                scores=result.scores,
                rank=rank,
                flags=result.flags,
            ))
            rank += 1
            if rank > self.top_k:
                break
        return demos

    # ------------------------------------------------------------------
    # Teacher-Student distillation helper
    # ------------------------------------------------------------------

    def distill(
        self,
        teacher_pool:   List[Dict[str, Any]],
        student_pool:   Optional[List[Dict[str, Any]]] = None,
        teacher_boost:  float = 0.1,
    ) -> List[Demonstration]:
        """
        Two-pool teacher-student selection.

        teacher_pool : high-quality examples (e.g. from a curated YAML library).
                       Their confidence scores receive a +teacher_boost bonus.
        student_pool : runtime-generated examples (e.g. conversation memory).
                       Selected on raw judge scores.

        The combined top-K is returned, with teacher examples preferred
        when confidence is equal.
        """
        # Score teacher pool with boost
        teacher_scored = self.score_candidates(teacher_pool)
        student_scored = self.score_candidates(student_pool or [])

        combined: List[Tuple[Dict[str, Any], JudgeResult, bool]] = []
        for c, r in teacher_scored:
            boosted = min(r.confidence + teacher_boost, 1.0)
            r2 = JudgeResult(
                question=r.question, answer=r.answer,
                scores=r.scores, confidence=boosted,
                strategy=r.strategy, flags=r.flags,
            )
            combined.append((c, r2, True))  # is_teacher=True
        for c, r in student_scored:
            combined.append((c, r, False))

        # Sort: confidence desc, teacher first on tie
        combined.sort(key=lambda x: (x[1].confidence, int(x[2])), reverse=True)

        demos: List[Demonstration] = []
        rank = 1
        for candidate, result, _ in combined:
            if result.confidence < self.min_confidence:
                continue
            q, a = self._extract_pair(candidate)
            if not q or not a:
                continue
            demos.append(Demonstration(
                question=q,
                answer=a,
                confidence=result.confidence,
                scores=result.scores,
                rank=rank,
                flags=result.flags,
            ))
            rank += 1
            if rank > self.top_k:
                break
        return demos

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_jsonl(
        self,
        demos:  List[Demonstration],
        path:   Path,
        format: str = "sharegpt",   # "sharegpt" | "alpaca"
    ) -> int:
        """
        Write demonstrations to a JSONL file.
        Returns the number of records written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for d in demos:
                rec = d.to_sharegpt() if format == "sharegpt" else d.to_alpaca()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def load_jsonl(path: Path, question_key: str = "question",
                   answer_key: str = "answer") -> List[Dict[str, Any]]:
        """
        Load a JSONL file (ShareGPT, Alpaca, or raw Q&A) into a flat list
        of dicts suitable for passing to `select()` / `score_candidates()`.
        """
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Normalise ShareGPT format
            if "conversations" in obj:
                convs   = obj["conversations"]
                user    = next((c["content"] for c in convs if c.get("role") == "user"), "")
                asst    = next((c["content"] for c in convs if c.get("role") == "assistant"), "")
                records.append({"question": user, "answer": asst})
            # Normalise Alpaca format
            elif "instruction" in obj and "output" in obj:
                records.append({"question": obj["instruction"], "answer": obj["output"]})
            else:
                records.append(obj)
        return records
