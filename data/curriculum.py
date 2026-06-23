#!/usr/bin/env python3
"""
Error-driven curriculum learning (Priority 3).

Implements confidence-scored error pattern matching and curriculum sorting
for training data.  High-confidence, well-understood patterns are placed
early in the curriculum; ambiguous or complex cases come later — exactly
using a confidence-gated escalation model.

No external LLMs or services required.

Components
----------
ErrorPattern      : a regex + metadata record describing a known error class
PatternMatcher    : matches text against a bank of ErrorPatterns and returns
                    a ConfidenceMatch with a 0.0–1.0 confidence score
CurriculumSorter  : takes a list of training examples, optionally annotates
                    each with a confidence score, and returns them ordered
                    from simple (high confidence) → complex (low confidence)
CurriculumDataset : thin wrapper that exposes a sorted iterable of examples
                    ready to feed into a training loop

Built-in clinical error patterns are provided; custom patterns can be added.

Usage
-----
    from data.curriculum import CurriculumSorter, DEFAULT_MEDICAL_PATTERNS

    sorter = CurriculumSorter(patterns=DEFAULT_MEDICAL_PATTERNS)
    examples = [
        {"instruction": "Manage elevated blood pressure", "output": "Confirm readings and start first-line therapy."},
        {"instruction": "Review contraindications", "output": "Avoid aspirin when active bleeding is present."},
    ]
    ordered = sorter.sort(examples, text_key="output")
    # High-confidence examples come first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Confidence levels for curriculum ordering
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH   = "HIGH"    # ≥ 0.80  — auto-accept, place early in curriculum
    MEDIUM = "MEDIUM"  # ≥ 0.50  — apply with monitoring, middle of curriculum
    LOW    = "LOW"     # < 0.50  — escalate / place late in curriculum

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.80:
            return cls.HIGH
        if score >= 0.50:
            return cls.MEDIUM
        return cls.LOW


# ---------------------------------------------------------------------------
# ErrorPattern
# ---------------------------------------------------------------------------

@dataclass
class ErrorPattern:
    """
    Describes a known error class for pattern matching.

    Attributes
    ----------
    name         : human-readable identifier
    pattern      : compiled regex to match in text
    confidence   : base confidence when this pattern fires (0.0–1.0)
    phase        : lifecycle phase (schema, loading, validation, …)
    fix_hint     : brief description of the canonical fix
    error_code   : optional error code (e.g. "CLIN_CONTRA_001")
    """
    name:       str
    pattern:    re.Pattern
    confidence: float
    phase:      str = "unknown"
    fix_hint:   str = ""
    error_code: str = ""

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceMatch:
    pattern:    Optional[ErrorPattern]  # None if no pattern matched
    confidence: float                   # 0.0 if unmatched
    level:      ConfidenceLevel
    fix_hint:   str

    @classmethod
    def no_match(cls) -> "ConfidenceMatch":
        return cls(
            pattern=None,
            confidence=0.0,
            level=ConfidenceLevel.LOW,
            fix_hint="",
        )

    def action(self) -> str:
        """Recommended action based on confidence level."""
        if self.level == ConfidenceLevel.HIGH:
            return "auto_retry"
        if self.level == ConfidenceLevel.MEDIUM:
            return "retry_with_monitoring"
        return "escalate_to_human"


# ---------------------------------------------------------------------------
# Built-in clinical error / complexity patterns for curriculum sorting
# ---------------------------------------------------------------------------

DEFAULT_MEDICAL_PATTERNS: List[ErrorPattern] = [
    ErrorPattern(
        name="contraindication",
        pattern=re.compile(
            r"contraindic|do\s+not\s+use|avoid\s+in|not\s+recommended\s+for",
            re.IGNORECASE,
        ),
        confidence=0.95,
        phase="safety",
        fix_hint="Review contraindications and choose an alternative therapy.",
        error_code="CLIN_CONTRA_001",
    ),
    ErrorPattern(
        name="drug_interaction",
        pattern=re.compile(
            r"interaction|potentiates|inhibits\s+metabolism|serotonin\s+syndrome",
            re.IGNORECASE,
        ),
        confidence=0.92,
        phase="pharmacology",
        fix_hint="Check medication list and adjust or monitor interacting agents.",
        error_code="CLIN_INTERACT_002",
    ),
    ErrorPattern(
        name="dosage_error",
        pattern=re.compile(
            r"overdose|subtherapeutic|incorrect\s+dose|toxic\s+dose|underdose",
            re.IGNORECASE,
        ),
        confidence=0.90,
        phase="prescribing",
        fix_hint="Verify weight, renal function, and guideline-recommended dose.",
        error_code="CLIN_DOSE_003",
    ),
    ErrorPattern(
        name="diagnosis_uncertainty",
        pattern=re.compile(
            r"differential\s+diagnosis|rule\s+out|confirm\s+with|white.?coat",
            re.IGNORECASE,
        ),
        confidence=0.85,
        phase="diagnosis",
        fix_hint="Confirm diagnosis with repeated measurements or additional workup.",
        error_code="CLIN_DX_004",
    ),
    ErrorPattern(
        name="monitoring_required",
        pattern=re.compile(
            r"monitor|follow.?up|reassess\s+in|surveillance|track\s+blood\s+pressure",
            re.IGNORECASE,
        ),
        confidence=0.82,
        phase="follow_up",
        fix_hint="Schedule follow-up and document monitoring plan.",
        error_code="CLIN_MONITOR_005",
    ),
    ErrorPattern(
        name="adverse_event",
        pattern=re.compile(
            r"adverse|side\s+effect|anaphylaxis|bleeding|hypotension|syncope",
            re.IGNORECASE,
        ),
        confidence=0.80,
        phase="safety",
        fix_hint="Assess severity, stop offending agent if needed, and treat symptoms.",
        error_code="CLIN_ADVERSE_006",
    ),
    ErrorPattern(
        name="guideline_deviation",
        pattern=re.compile(
            r"off.?label|guideline|evidence.?based|standard\s+of\s+care",
            re.IGNORECASE,
        ),
        confidence=0.75,
        phase="quality",
        fix_hint="Align plan with current clinical guidelines and document rationale.",
        error_code="CLIN_GUIDE_007",
    ),
    ErrorPattern(
        name="missing_history",
        pattern=re.compile(
            r"patient\s+history|allergy|pregnancy|comorbid|renal\s+function",
            re.IGNORECASE,
        ),
        confidence=0.72,
        phase="assessment",
        fix_hint="Collect complete history before finalizing treatment.",
        error_code="CLIN_HISTORY_008",
    ),
    ErrorPattern(
        name="documentation_gap",
        pattern=re.compile(
            r"unclear\s+indication|missing\s+documentation|incomplete\s+assessment",
            re.IGNORECASE,
        ),
        confidence=0.65,
        phase="quality",
        fix_hint="Document indication, shared decision-making, and follow-up plan.",
    ),
    ErrorPattern(
        name="generic_error",
        pattern=re.compile(r"error|exception|failed|failure|concern", re.IGNORECASE),
        confidence=0.30,
        phase="unknown",
        fix_hint="Manual clinical review required.",
    ),
]


# ---------------------------------------------------------------------------
# PatternMatcher
# ---------------------------------------------------------------------------

class PatternMatcher:
    """
    Match text against a bank of ErrorPatterns.

    Returns the *highest-confidence* match (or a no-match sentinel).
    Patterns are tried in descending confidence order so the most
    specific match wins.
    """

    def __init__(self, patterns: Optional[List[ErrorPattern]] = None) -> None:
        self._patterns = sorted(
            patterns or DEFAULT_MEDICAL_PATTERNS,
            key=lambda p: p.confidence,
            reverse=True,
        )

    def match(self, text: str) -> ConfidenceMatch:
        """Return the best-matching ErrorPattern for *text*."""
        for pat in self._patterns:
            if pat.matches(text):
                return ConfidenceMatch(
                    pattern=pat,
                    confidence=pat.confidence,
                    level=ConfidenceLevel.from_score(pat.confidence),
                    fix_hint=pat.fix_hint,
                )
        return ConfidenceMatch.no_match()

    def match_all(self, text: str) -> List[ConfidenceMatch]:
        """Return all matching patterns (sorted by confidence desc)."""
        results = []
        for pat in self._patterns:
            if pat.matches(text):
                results.append(ConfidenceMatch(
                    pattern=pat,
                    confidence=pat.confidence,
                    level=ConfidenceLevel.from_score(pat.confidence),
                    fix_hint=pat.fix_hint,
                ))
        return results

    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add a custom pattern and re-sort."""
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda p: p.confidence, reverse=True)


# ---------------------------------------------------------------------------
# Annotated example
# ---------------------------------------------------------------------------

@dataclass
class CurriculumExample:
    data:       Dict[str, Any]
    confidence: float
    level:      ConfidenceLevel
    match:      Optional[ConfidenceMatch] = None


# ---------------------------------------------------------------------------
# CurriculumSorter
# ---------------------------------------------------------------------------

class CurriculumSorter:
    """
    Sort training examples from easy (high confidence) → hard (low confidence).

    For each example, the confidence is derived by:
      1. Matching the example's text field against known error patterns.
      2. If no pattern matches, a heuristic length/complexity score is used
         so purely informational examples rank in the middle.

    Parameters
    ----------
    patterns   : error patterns to use (default: DEFAULT_MEDICAL_PATTERNS)
    text_key   : dict key to extract text from each example
                 (defaults to "output", then "answer", then full str)
    """

    def __init__(
        self,
        patterns: Optional[List[ErrorPattern]] = None,
        text_key: Optional[str] = None,
    ) -> None:
        self._matcher  = PatternMatcher(patterns)
        self._text_key = text_key

    def _get_text(self, example: Dict[str, Any]) -> str:
        if self._text_key and self._text_key in example:
            return str(example[self._text_key])
        # Auto-detect common keys
        for k in ("output", "answer", "content", "text", "instruction"):
            if k in example:
                return str(example[k])
        return str(example)

    def _heuristic_confidence(self, text: str) -> float:
        """
        Fallback confidence for non-error examples.
        Short, simple text → high confidence (easy); long/complex → lower.
        Range: 0.50–0.75 (medium band) so they sit between errors.
        """
        word_count = len(text.split())
        has_structured = bool(re.search(
            r"\b(patient|medication|treatment|diagnosis|protocol|dosage)\b", text, re.IGNORECASE
        ))
        has_code   = "```" in text or bool(re.search(r"\bdef\b|\bclass\b", text))

        base = 0.72
        if word_count > 100:
            base -= 0.10
        if has_structured or has_code:
            base -= 0.05
        return max(0.50, min(base, 0.75))

    def annotate(self, examples: List[Dict[str, Any]]) -> List[CurriculumExample]:
        """Return examples annotated with confidence and level."""
        annotated: List[CurriculumExample] = []
        for ex in examples:
            text  = self._get_text(ex)
            match = self._matcher.match(text)
            if match.pattern is not None:
                conf  = match.confidence
                level = match.level
            else:
                conf  = self._heuristic_confidence(text)
                level = ConfidenceLevel.from_score(conf)
                match = None  # type: ignore[assignment]
            annotated.append(CurriculumExample(
                data=ex,
                confidence=conf,
                level=level,
                match=match if (match and match.pattern) else None,
            ))
        return annotated

    def sort(
        self,
        examples: List[Dict[str, Any]],
        text_key: Optional[str] = None,
    ) -> List[CurriculumExample]:
        """
        Return examples sorted easy → hard (highest confidence first).

        Parameters
        ----------
        examples : list of training example dicts
        text_key : override the instance-level text_key for this call
        """
        if text_key:
            self._text_key = text_key
        annotated = self.annotate(examples)
        annotated.sort(key=lambda x: x.confidence, reverse=True)
        return annotated

    def split_by_level(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, List[CurriculumExample]]:
        """
        Partition examples into HIGH / MEDIUM / LOW buckets.
        Useful for staged training or weighted sampling.
        """
        annotated = self.annotate(examples)
        buckets: Dict[str, List[CurriculumExample]] = {
            "HIGH":   [],
            "MEDIUM": [],
            "LOW":    [],
        }
        for ex in annotated:
            buckets[ex.level.value].append(ex)
        return buckets


# ---------------------------------------------------------------------------
# CurriculumDataset
# ---------------------------------------------------------------------------

class CurriculumDataset:
    """
    Thin wrapper around CurriculumSorter that exposes a list-like interface
    for use in training loops.

    Usage
    -----
        ds = CurriculumDataset(examples, patterns=DEFAULT_MEDICAL_PATTERNS)
        for item in ds:
            train(item.data)
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        patterns: Optional[List[ErrorPattern]] = None,
        text_key: Optional[str] = None,
    ) -> None:
        sorter = CurriculumSorter(patterns=patterns, text_key=text_key)
        self._sorted: List[CurriculumExample] = sorter.sort(examples)

    def __len__(self) -> int:
        return len(self._sorted)

    def __iter__(self):
        return iter(self._sorted)

    def __getitem__(self, idx: int) -> CurriculumExample:
        return self._sorted[idx]

    def high(self) -> List[CurriculumExample]:
        return [e for e in self._sorted if e.level == ConfidenceLevel.HIGH]

    def medium(self) -> List[CurriculumExample]:
        return [e for e in self._sorted if e.level == ConfidenceLevel.MEDIUM]

    def low(self) -> List[CurriculumExample]:
        return [e for e in self._sorted if e.level == ConfidenceLevel.LOW]

    def as_dicts(self) -> List[Dict[str, Any]]:
        """Return raw example dicts in curriculum order."""
        return [e.data for e in self._sorted]
