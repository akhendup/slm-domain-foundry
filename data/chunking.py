#!/usr/bin/env python3
"""
Simple text chunking for training data. Preserves paragraph boundaries.
Optional: use sentence-transformers for semantic chunking if installed.
SQL-aware chunking treats SQL paragraphs as atomic units (never split mid-query).
"""

import logging
import re
from typing import List

_log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


_SQL_KW_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|PARTITION\s+BY|OVER|"
    r"JOIN|CREATE|WITH|QUALIFY|CSUM|MSUM|MAVG|MDIFF|RANK|ROW_NUMBER)\b",
    re.IGNORECASE,
)


def is_sql_paragraph(para: str) -> bool:
    """True if paragraph contains SQL content that should not be split across chunks."""
    return len(_SQL_KW_RE.findall(para)) >= 2


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_semantic: bool = False,
) -> List[str]:
    """
    Split text into chunks. By default uses rule-based chunking.
    If use_semantic=True and sentence-transformers is installed, uses sentence boundaries.
    """
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    if use_semantic and SEMANTIC_AVAILABLE:
        try:
            return _chunk_semantic(text, chunk_size, chunk_overlap)
        except Exception as exc:
            _log.warning("Semantic chunking failed, falling back to rule-based: %s", exc)
    return _chunk_rule_based(text, chunk_size, chunk_overlap)


def chunk_text_sql_aware(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Like chunk_text but treats SQL-containing paragraphs as atomic units:
    they are never split across chunks and never included in the overlap
    carry-over (so SQL context is always complete within a chunk).
    """
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
    return _chunk_rule_based_sql_aware(text, chunk_size, chunk_overlap)


def _chunk_rule_based(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    paragraphs = re.split(r"\n\n+", text)
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        n = len(para) + 2
        if current_len + n > chunk_size and current:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            if overlap > 0 and current:
                overlap_paras: List[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current = overlap_paras + [para]
                current_len = sum(len(p) for p in current) + 2 * (len(current) - 1)
            else:
                current = [para]
                current_len = n
        else:
            current.append(para)
            current_len += n

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_rule_based_sql_aware(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Rule-based chunking that keeps SQL paragraphs whole.
    SQL paragraphs are:
    - Never split across chunks (added to current or emitted as their own chunk)
    - Never included in the overlap carry-over region
    """
    chunks: List[str] = []
    paragraphs = re.split(r"\n\n+", text)
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sql = is_sql_paragraph(para)
        n = len(para) + 2

        if current_len + n > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: carry over recent non-SQL paragraphs only
            if overlap > 0:
                overlap_paras: List[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if is_sql_paragraph(p):
                        break  # never include SQL in overlap
                    if overlap_len + len(p) <= overlap:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current = overlap_paras + [para]
                current_len = sum(len(p) for p in current) + 2 * max(0, len(current) - 1)
            else:
                current = [para]
                current_len = n
        else:
            current.append(para)
            current_len += n

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_semantic(text: str, chunk_size: int, overlap: int) -> List[str]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if not sentences:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        n = len(s) + 1
        if current_len + n > chunk_size and current:
            chunks.append(" ".join(current))
            if overlap > 0 and len(current) > 1:
                overlap_sents: List[str] = []
                overlap_len = 0
                for x in reversed(current):
                    if overlap_len + len(x) <= overlap:
                        overlap_sents.insert(0, x)
                        overlap_len += len(x)
                    else:
                        break
                current = overlap_sents + [s]
                current_len = sum(len(x) for x in current) + len(current) - 1
            else:
                current = [s]
                current_len = n
        else:
            current.append(s)
            current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks
