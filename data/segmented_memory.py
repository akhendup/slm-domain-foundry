#!/usr/bin/env python3
"""
Segmented memory with hard token budgets (Priority 2).

Implements a four-segment context window manager inspired by loom's
agent memory architecture.  No external services required — token
counting uses a simple word/punctuation heuristic.

Segments (from loom architecture)
----------------------------------
ROM     (5 000 tokens, immutable) : system prompt, static patterns
Kernel  (2 000 tokens, mutable)   : session context (user prefs, goals)
L1      (10 000 tokens, FIFO)     : recent conversation turns
L2      (3 000 tokens, summary)   : extractive summary of evicted L1 turns

Total budget: ~20 000 tokens.

When L1 is full, the oldest turns are evicted and their key sentences
are appended to L2 (extractive compression — no LLM required).
When L2 is full, the oldest summary lines are trimmed.

Usage
-----
    from data.segmented_memory import SegmentedMemory

    mem = SegmentedMemory()
    mem.set_rom("You are a helpful medical assistant.")
    mem.set_kernel({"user": "Alice", "goal": "Learn hypertension management"})
    mem.add_turn("user", "What is hypertension?")
    mem.add_turn("assistant", "Hypertension is chronic elevation of blood pressure...")

    prompt_parts = mem.build_prompt()   # dict with rom/kernel/l1/l2 text
    full_context = mem.render()         # single string ready for the model
    stats        = mem.stats()          # token usage per segment
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Token counting (heuristic — avoids tiktoken dependency)
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[.,;:!?()\"'\[\]{}<>/\\|@#$%^&*+=`~_-]")


def count_tokens(text: str) -> int:
    """
    Approximate token count.

    GPT-family tokenisers produce roughly 1 token per 4 characters for
    English prose; we use word count × 1.3 as a conservative over-estimate
    that avoids under-counting.
    """
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


# ---------------------------------------------------------------------------
# Extractive summary (L2 compression — no LLM)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_key_sentences(text: str, budget_tokens: int) -> str:
    """
    Return the most informative sentences from *text* that fit in
    *budget_tokens*.

    Strategy: prefer sentences that contain structured content keywords, numbers, or
    named entities (heuristic: capitalised tokens ≥ 4 chars).
    """
    sentences = _SENTENCE_SPLIT.split(text.strip())
    if not sentences:
        return ""

    _clinical_kw = re.compile(
        r"\b(hypertension|aspirin|contraindication|dosage|dose|monitoring|"
        r"diagnosis|guideline|adverse|interaction|blood pressure|therapy|"
        r"treatment|prescrib|patient)\b",
        re.IGNORECASE,
    )
    _num = re.compile(r"\b\d+\b")
    _named = re.compile(r"\b[A-Z][a-z]{3,}\b")

    def _priority(s: str) -> int:
        return (
            len(_clinical_kw.findall(s)) * 3
            + len(_num.findall(s))
            + len(_named.findall(s))
        )

    ranked = sorted(enumerate(sentences), key=lambda x: _priority(x[1]), reverse=True)

    chosen_idx: List[int] = []
    used = 0
    for idx, sent in ranked:
        t = count_tokens(sent)
        if used + t <= budget_tokens:
            chosen_idx.append(idx)
            used += t
        if used >= budget_tokens:
            break

    # Re-emit in original order for readability
    chosen_idx.sort()
    return " ".join(sentences[i] for i in chosen_idx)


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role:    str   # "user" | "assistant" | "system"
    content: str
    tokens:  int   = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = count_tokens(self.content)


# ---------------------------------------------------------------------------
# Segment budgets
# ---------------------------------------------------------------------------

@dataclass
class SegmentBudgets:
    rom:    int = 5_000
    kernel: int = 2_000
    l1:     int = 10_000
    l2:     int = 3_000

    @property
    def total(self) -> int:
        return self.rom + self.kernel + self.l1 + self.l2


# ---------------------------------------------------------------------------
# SegmentedMemory
# ---------------------------------------------------------------------------

class SegmentedMemory:
    """
    Four-segment context window with hard per-segment token budgets.

    ROM    — set once; never evicted.
    Kernel — replaced wholesale (small session metadata).
    L1     — FIFO deque of recent turns; oldest evicted when full.
    L2     — extractive summary built from evicted L1 content.
    """

    def __init__(self, budgets: Optional[SegmentBudgets] = None) -> None:
        self._budgets = budgets or SegmentBudgets()

        self._rom:    str                = ""
        self._kernel: Dict[str, Any]     = {}
        self._l1:     Deque[Turn]        = deque()
        self._l2:     str                = ""

        self._rom_tokens:    int = 0
        self._kernel_tokens: int = 0
        self._l1_tokens:     int = 0
        self._l2_tokens:     int = 0

    # ------------------------------------------------------------------
    # ROM
    # ------------------------------------------------------------------

    def set_rom(self, text: str) -> None:
        """Set the immutable system / static prompt section."""
        t = count_tokens(text)
        if t > self._budgets.rom:
            # Truncate to budget (word boundary)
            words = text.split()
            cap   = int(self._budgets.rom / 1.3)
            text  = " ".join(words[:cap])
            t     = count_tokens(text)
        self._rom        = text
        self._rom_tokens = t

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def set_kernel(self, context: Dict[str, Any]) -> None:
        """Replace session-level context dict (user, goal, prefs, etc.)."""
        text = _dict_to_text(context)
        t    = count_tokens(text)
        if t > self._budgets.kernel:
            # Trim dict values until within budget
            keys   = list(context.keys())
            pruned = {}
            used   = 0
            for k in keys:
                val  = str(context[k])
                vt   = count_tokens(val)
                if used + vt <= self._budgets.kernel:
                    pruned[k] = val
                    used += vt
            context = pruned
            text    = _dict_to_text(context)
            t       = count_tokens(text)
        self._kernel        = context
        self._kernel_tokens = t

    def update_kernel(self, key: str, value: Any) -> None:
        """Upsert a single key in the kernel."""
        updated = dict(self._kernel)
        updated[key] = value
        self.set_kernel(updated)

    # ------------------------------------------------------------------
    # L1 (recent turns — FIFO)
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn; evict oldest to L2 if L1 is full."""
        turn   = Turn(role=role, content=content)
        budget = self._budgets.l1

        # Evict if necessary
        while self._l1_tokens + turn.tokens > budget and self._l1:
            evicted       = self._l1.popleft()
            self._l1_tokens -= evicted.tokens
            self._compress_to_l2(evicted.content)

        self._l1.append(turn)
        self._l1_tokens += turn.tokens

    def _compress_to_l2(self, text: str) -> None:
        """Extractively compress *text* and append to L2; trim L2 if full."""
        budget_per_chunk = max(self._budgets.l2 // 4, 50)
        summary          = _extract_key_sentences(text, budget_per_chunk)
        if not summary:
            return
        combined      = (self._l2 + " " + summary).strip()
        combined_tok  = count_tokens(combined)

        # Trim L2 if overflow
        if combined_tok > self._budgets.l2:
            sentences = _SENTENCE_SPLIT.split(combined)
            kept      = []
            used      = 0
            for s in reversed(sentences):          # keep newest summary
                t = count_tokens(s)
                if used + t > self._budgets.l2:
                    break
                kept.append(s)
                used += t
            combined     = " ".join(reversed(kept))
            combined_tok = count_tokens(combined)

        self._l2        = combined
        self._l2_tokens = combined_tok

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def build_prompt(self) -> Dict[str, str]:
        """Return each segment as a separate string."""
        return {
            "rom":    self._rom,
            "kernel": _dict_to_text(self._kernel),
            "l2":     self._l2,
            "l1":     _turns_to_text(list(self._l1)),
        }

    def render(self) -> str:
        """Concatenate all segments into a single context string."""
        parts = self.build_prompt()
        sections: List[str] = []
        if parts["rom"]:
            sections.append(parts["rom"])
        if parts["kernel"]:
            sections.append("[Session context]\n" + parts["kernel"])
        if parts["l2"]:
            sections.append("[Summary of earlier conversation]\n" + parts["l2"])
        if parts["l1"]:
            sections.append(parts["l1"])
        return "\n\n".join(sections)

    def get_l1_messages(self) -> List[Dict[str, str]]:
        """Return L1 turns as a list of {role, content} dicts."""
        return [{"role": t.role, "content": t.content} for t in self._l1]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Dict[str, int]]:
        """Return token usage and budgets per segment."""
        return {
            "rom":    {"used": self._rom_tokens,    "budget": self._budgets.rom},
            "kernel": {"used": self._kernel_tokens, "budget": self._budgets.kernel},
            "l1":     {"used": self._l1_tokens,     "budget": self._budgets.l1},
            "l2":     {"used": self._l2_tokens,     "budget": self._budgets.l2},
            "total":  {
                "used":   self._rom_tokens + self._kernel_tokens
                          + self._l1_tokens + self._l2_tokens,
                "budget": self._budgets.total,
            },
        }

    def clear_l1(self) -> None:
        """Flush all L1 turns (useful on session reset)."""
        for turn in self._l1:
            self._compress_to_l2(turn.content)
        self._l1.clear()
        self._l1_tokens = 0

    def reset(self) -> None:
        """Full reset — clears all segments."""
        self._rom = ""
        self._kernel = {}
        self._l1.clear()
        self._l2 = ""
        self._rom_tokens = self._kernel_tokens = self._l1_tokens = self._l2_tokens = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_text(d: Dict[str, Any]) -> str:
    if not d:
        return ""
    return "\n".join(f"{k}: {v}" for k, v in d.items())


def _turns_to_text(turns: List[Turn]) -> str:
    if not turns:
        return ""
    return "\n".join(f"{t.role.upper()}: {t.content}" for t in turns)
