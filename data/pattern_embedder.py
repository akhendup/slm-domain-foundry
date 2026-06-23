#!/usr/bin/env python3
"""
TF-IDF pattern embedding and semantic selection (Priority 6).

Embeds YAML knowledge-library patterns using TF-IDF vectors and
cosine similarity so the most relevant patterns are injected into
the system prompt at inference time.  No external LLMs or services
required — uses only Python's standard library + numpy (already a
transitive dependency).

When the optional `sentence-transformers` package is installed a
richer dense embedding can be used instead, but TF-IDF is the
default and is always available.

Key idea (from agenticwfproject)
---------------------------------
Store domain knowledge inside YAML patterns so the model
*selects* the right pattern rather than *generating ad hoc answers from scratch.
This dramatically reduces the generation burden on small models.

Architecture
------------
PatternEmbedder
    .fit(patterns)       → build TF-IDF vocabulary from pattern texts
    .search(query, k)    → return top-k patterns by cosine similarity
    .get_context(query)  → formatted context string for prompt injection

Usage
-----
    from data.pattern_embedder import PatternEmbedder
    import yaml, pathlib

    patterns = [yaml.safe_load(p.read_text()) for p in pathlib.Path("library").glob("*.yaml")]
    embedder = PatternEmbedder()
    embedder.fit(patterns)
    context = embedder.get_context("How do I compute a running total?", top_k=3)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "as", "into",
    "through", "and", "or", "but", "not", "this", "that", "if",
    "it", "its", "from", "also", "use", "used", "using",
})


def _tokenise(text: str) -> List[str]:
    """Lower-case, split on non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _pattern_text(pattern: Dict[str, Any]) -> str:
    """Concatenate all human-readable fields of a pattern into one string."""
    _skip = {"_source_file", "templates"}
    parts: List[str] = []

    for k, v in pattern.items():
        if k in _skip:
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(str(x) for x in item.values() if isinstance(x, str))

    # Also index first template content from templates for domain keyword coverage
    templates = pattern.get("templates")
    if isinstance(templates, dict):
        for tval in templates.values():
            if isinstance(tval, dict) and tval.get("content"):
                parts.append(str(tval["content"])[:200])
                break

    return " ".join(parts)


# ---------------------------------------------------------------------------
# TF-IDF engine
# ---------------------------------------------------------------------------

@dataclass
class _TFIDFVector:
    pattern: Dict[str, Any]
    text:    str
    tfidf:   Dict[str, float]   # term → tf-idf weight


class _TFIDFIndex:
    """Minimal in-process TF-IDF index."""

    def __init__(self) -> None:
        self._docs:  List[_TFIDFVector] = []
        self._df:    Counter            = Counter()   # term → doc frequency
        self._n_docs: int               = 0

    def fit(self, patterns: List[Dict[str, Any]]) -> None:
        """Build the index from a list of pattern dicts."""
        self._docs   = []
        self._df     = Counter()
        self._n_docs = len(patterns)

        raw: List[Tuple[Dict[str, Any], str, Counter]] = []
        for pat in patterns:
            text   = _pattern_text(pat)
            tokens = _tokenise(text)
            tf     = Counter(tokens)
            raw.append((pat, text, tf))
            for term in tf:
                self._df[term] += 1

        for pat, text, tf in raw:
            n   = sum(tf.values()) or 1
            vec = {}
            for term, cnt in tf.items():
                tf_val  = cnt / n
                idf_val = math.log((self._n_docs + 1) / (self._df[term] + 1)) + 1
                vec[term] = tf_val * idf_val
            self._docs.append(_TFIDFVector(pattern=pat, text=text, tfidf=vec))

    def query_vector(self, query: str) -> Dict[str, float]:
        """Compute TF-IDF vector for a query string."""
        tokens  = _tokenise(query)
        tf      = Counter(tokens)
        n       = sum(tf.values()) or 1
        vec: Dict[str, float] = {}
        for term, cnt in tf.items():
            tf_val  = cnt / n
            idf_val = math.log((self._n_docs + 1) / (self._df.get(term, 0) + 1)) + 1
            vec[term] = tf_val * idf_val
        return vec

    def cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse TF-IDF vectors."""
        dot  = sum(a[t] * b[t] for t in a if t in b)
        na   = math.sqrt(sum(v * v for v in a.values()))
        nb   = math.sqrt(sum(v * v for v in b.values()))
        denom = na * nb
        return dot / denom if denom > 0 else 0.0

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Return top-k (pattern, score) pairs sorted by cosine similarity."""
        if not self._docs:
            return []
        qv     = self.query_vector(query)
        scores = [(d.pattern, self.cosine(qv, d.tfidf)) for d in self._docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# SentenceTransformer backend (optional)
# ---------------------------------------------------------------------------

def _try_load_st():
    """Return a SentenceTransformer model if the package is available."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PatternEmbedder
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    pattern:    Dict[str, Any]
    score:      float
    backend:    str          # "tfidf" | "dense"


class PatternEmbedder:
    """
    Embed YAML patterns and retrieve the most relevant ones for a query.

    By default uses TF-IDF (always available, no extra dependencies).
    Set `use_dense=True` to use sentence-transformers when installed;
    falls back to TF-IDF automatically if sentence-transformers is absent.

    Parameters
    ----------
    use_dense    : prefer dense embeddings (sentence-transformers)
    min_score    : minimum similarity score to include in results
    """

    def __init__(
        self,
        use_dense:  bool  = False,
        min_score:  float = 0.05,
    ) -> None:
        self._min_score = min_score
        self._tfidf     = _TFIDFIndex()
        self._patterns: List[Dict[str, Any]] = []
        self._fitted    = False

        # Dense backend (optional)
        self._st_model   = _try_load_st() if use_dense else None
        self._dense_vecs: Optional[Any]   = None   # numpy array when fitted
        self._backend = "dense" if self._st_model else "tfidf"

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, patterns: List[Dict[str, Any]]) -> "PatternEmbedder":
        """Index a list of pattern dicts. Call before any search."""
        self._patterns = patterns
        self._tfidf.fit(patterns)

        if self._st_model is not None:
            texts = [_pattern_text(p) for p in patterns]
            self._dense_vecs = self._st_model.encode(texts, convert_to_numpy=True)

        self._fitted = True
        return self

    def fit_from_yaml_dir(self, directory) -> "PatternEmbedder":
        """Load all *.yaml files from *directory* and fit."""
        from pathlib import Path
        try:
            import yaml as _yaml
        except ImportError:
            raise ImportError("PyYAML is required: pip install pyyaml")

        directory = Path(directory)
        patterns  = []
        for p in sorted(directory.glob("*.yaml")):
            if p.name.startswith("_"):
                continue
            try:
                data = _yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("name"):
                    data["_source_file"] = str(p)
                    patterns.append(data)
            except Exception:
                pass
        return self.fit(patterns)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[PatternMatch]:
        """Return the top-k most relevant patterns for *query*."""
        if not self._fitted or not self._patterns:
            return []

        if self._backend == "dense" and self._dense_vecs is not None:
            results = self._dense_search(query, top_k)
        else:
            results = self._tfidf_search(query, top_k)

        return [r for r in results if r.score >= self._min_score]

    def _tfidf_search(self, query: str, top_k: int) -> List[PatternMatch]:
        scored = self._tfidf.search(query, top_k=top_k)
        return [PatternMatch(pattern=p, score=round(s, 4), backend="tfidf")
                for p, s in scored]

    def _dense_search(self, query: str, top_k: int) -> List[PatternMatch]:
        import numpy as np
        qv   = self._st_model.encode([query], convert_to_numpy=True)[0]
        sims = self._dense_vecs @ qv / (
            np.linalg.norm(self._dense_vecs, axis=1) * np.linalg.norm(qv) + 1e-9
        )
        indices = np.argsort(sims)[::-1][:top_k]
        return [
            PatternMatch(pattern=self._patterns[i], score=round(float(sims[i]), 4), backend="dense")
            for i in indices
        ]

    # ------------------------------------------------------------------
    # Context builder (for prompt injection)
    # ------------------------------------------------------------------

    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Return a compact context string ready for system-prompt injection.
        Format mirrors KnowledgeRetriever.get_context() for compatibility.
        """
        matches = self.search(query, top_k=top_k)
        if not matches:
            return ""

        blocks: List[str] = []
        for m in matches:
            pat   = m.pattern
            title = pat.get("title") or pat.get("name", "")
            desc  = (pat.get("description") or "").strip()
            use_cases = pat.get("use_cases", [])

            lines = [f"### {title}  (similarity: {m.score:.2f})"]
            if desc:
                sentences = re.split(r"(?<=[.!?])\s+", desc)
                lines.append(" ".join(sentences[:2]))
            if use_cases and isinstance(use_cases, list):
                lines.append("Use cases: " + "; ".join(str(u) for u in use_cases[:3]))

            # First worked example template
            templates = pat.get("templates")
            if isinstance(templates, dict):
                for tv in templates.values():
                    if isinstance(tv, dict) and tv.get("content"):
                        content = tv["content"].strip()
                        lines.append(f"Example:\n```\n{content}\n```")
                        break

            blocks.append("\n".join(lines))

        header = (
            "Relevant knowledge patterns (ranked by semantic similarity):\n\n"
        )
        return header + "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def backend(self) -> str:
        return self._backend

    def is_fitted(self) -> bool:
        return self._fitted
