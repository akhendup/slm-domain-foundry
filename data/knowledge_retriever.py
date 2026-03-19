#!/usr/bin/env python3
"""
Runtime Knowledge Library retrieval for chat inference.

At chat time, this module searches the Knowledge Library YAML files for entries
relevant to the user's question and returns structured context to inject into the
system prompt.  This keeps knowledge LIVE — new entries in the library are
immediately available without retraining the model.

Architecture:
    User question → extract_query_terms()
                 → KnowledgeRetriever.search(question)  ← scans YAML files
                 → build_context_block(entries)          ← formats for prompt
                 → inject into system message

Usage (in gradio_ui.py _chat):
    retriever = KnowledgeRetriever(LIBRARY_DIR)
    context = retriever.get_context(user_message, max_entries=2)
    if context:
        system_msg = BASE_SYSTEM_PROMPT + "\n\n" + context
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Term extraction
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "to", "of", "in", "on", "at", "by",
    "for", "with", "about", "as", "into", "through", "and", "or", "but",
    "not", "what", "when", "how", "why", "which", "who", "where",
    "show", "me", "an", "example", "give", "tell", "explain", "describe",
    "use", "used", "using", "define", "i", "my", "you", "your",
    "sql", "query", "function", "it", "this", "that", "if",
})


def extract_query_terms(question: str) -> List[str]:
    """Extract meaningful terms from a user question for library lookup."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", question.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# YAML index
# ---------------------------------------------------------------------------

def _load_all_patterns(library_dir: Path) -> List[Dict[str, Any]]:
    """Load all YAML patterns from the knowledge library directory."""
    if not _YAML_OK or not library_dir.exists():
        return []
    patterns = []
    for p in sorted(library_dir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("name"):
                data["_source_file"] = str(p)
                patterns.append(data)
        except Exception as exc:
            _log.warning("Failed to load knowledge library file %s: %s", p, exc)
    return patterns


def _pattern_searchable_text(pattern: Dict[str, Any]) -> str:
    """Return a single lowercase string of all searchable text for a pattern."""
    # Collect all top-level string values (covers any domain-specific fields generically)
    _structured_keys = frozenset({"use_cases", "parameters", "templates", "common_errors", "_source_file"})
    parts: List[str] = [
        str(v) for k, v in pattern.items()
        if k not in _structured_keys and isinstance(v, str) and v
    ]
    use_cases = pattern.get("use_cases", [])
    if isinstance(use_cases, list):
        parts.extend(str(uc) for uc in use_cases)
    params = pattern.get("parameters", [])
    if isinstance(params, list):
        for p in params:
            if isinstance(p, dict):
                parts.append(p.get("name", ""))
                parts.append(p.get("description", ""))
    return " ".join(str(x) for x in parts if x).lower()


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _score(pattern: Dict[str, Any], terms: List[str]) -> int:
    """
    Score a pattern against query terms.
    Exact name/title match scores highest; body text matches score lower.
    """
    if not terms:
        return 0
    name = (pattern.get("name") or "").lower()
    title = (pattern.get("title") or "").lower()
    body = _pattern_searchable_text(pattern)
    score = 0
    for term in terms:
        if term == name or term == title:
            score += 10          # exact name match — very strong signal
        elif term in name or term in title:
            score += 5           # partial name match
        elif term in body:
            score += 1           # body mention
    return score


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _fmt_list(items: Any, prefix: str = "- ") -> str:
    if not items:
        return ""
    if isinstance(items, str):
        return items.strip()
    return "\n".join(f"{prefix}{str(x).strip()}" for x in items if x)


def build_context_block(patterns: List[Dict[str, Any]]) -> str:
    """
    Format retrieved patterns as a compact context block for injection
    into the system prompt.  Keeps only the most useful fields to avoid
    flooding the context window.
    """
    if not patterns:
        return ""
    blocks = []
    for pat in patterns:
        title = pat.get("title") or pat.get("name", "")
        desc = (pat.get("description") or "").strip()
        use_cases = pat.get("use_cases", [])
        params = pat.get("parameters", [])
        templates = pat.get("templates", {})
        errors = pat.get("common_errors", [])

        lines = [f"### {title}"]
        if desc:
            # Keep first 3 sentences to stay concise
            sentences = re.split(r"(?<=[.!?])\s+", desc)
            lines.append("\n".join(sentences[:3]))

        if use_cases:
            lines.append("Use cases: " + "; ".join(str(u) for u in use_cases[:5]))

        if params and isinstance(params, list):
            param_names = [p.get("name", "") for p in params if isinstance(p, dict) and p.get("name")]
            if param_names:
                lines.append("Parameters: " + ", ".join(param_names))

        # First SQL template
        if isinstance(templates, dict):
            for tval in templates.values():
                if isinstance(tval, dict):
                    sql = (tval.get("sql") or "").strip()
                    if sql:
                        lines.append(f"Example SQL:\n```sql\n{sql}\n```")
                    break

        # First 2 common errors
        if errors and isinstance(errors, list):
            err_lines = []
            for err in errors[:2]:
                if isinstance(err, dict) and err.get("error"):
                    sol = err.get("solution", "")
                    err_lines.append(f"- {err['error']}: {sol}")
            if err_lines:
                lines.append("Common errors:\n" + "\n".join(err_lines))

        blocks.append("\n".join(lines))

    header = (
        "The following knowledge library entries are relevant to this question. "
        "Use them as the authoritative source for your answer:\n\n"
    )
    return header + "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """
    Loads the Knowledge Library on first use and provides fast in-process search.

    Usage:
        retriever = KnowledgeRetriever(library_dir)
        context = retriever.get_context("What is nPath?", max_entries=2)
        # Returns formatted context string, or "" if nothing relevant found.

    Call `retriever.reload()` after the user adds a new library entry so the
    in-memory index stays current without restarting the server.
    """

    def __init__(self, library_dir: Path):
        self._dir = library_dir
        self._patterns: List[Dict[str, Any]] = []
        self._loaded = False

    def reload(self) -> None:
        """Force reload of all YAML files from disk."""
        self._patterns = _load_all_patterns(self._dir)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def search(self, question: str, max_entries: int = 2, min_score: int = 3) -> List[Dict[str, Any]]:
        """
        Return up to `max_entries` patterns most relevant to `question`.
        Returns [] if nothing scores at or above `min_score`.
        """
        self._ensure_loaded()
        if not self._patterns:
            return []
        terms = extract_query_terms(question)
        if not terms:
            return []
        scored = [(pat, _score(pat, terms)) for pat in self._patterns]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pat for pat, sc in scored[:max_entries] if sc >= min_score]

    def get_context(self, question: str, max_entries: int = 2) -> str:
        """
        Return a formatted context block for `question`, ready to inject into
        the system prompt.  Returns empty string if nothing relevant is found.
        """
        relevant = self.search(question, max_entries=max_entries)
        return build_context_block(relevant)
