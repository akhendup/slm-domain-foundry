#!/usr/bin/env python3
"""
Load text or Q&A pairs from CSV for SLM training.
Supports: single text column, or question/answer columns.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_csv(
    csv_path: Path,
    text_column: Optional[str] = None,
    question_column: Optional[str] = None,
    answer_column: Optional[str] = None,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Load CSV and return (list of text chunks, list of (question, answer) pairs).

    - If question_column and answer_column are set: parse Q&A pairs.
    - If text_column is set (or only one column exists): treat as raw text lines.
    - If no columns specified: auto-detect (look for 'question'/'answer' or 'text', or use first columns).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    texts: List[str] = []
    qa_pairs: List[Tuple[str, str]] = []

    with open(csv_path, newline="", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return [], []

        first = rows[0]
        headers = list(first.keys())

        # Auto-detect columns if not specified
        if not question_column and not answer_column and not text_column:
            headers_lower = [h.lower().strip() for h in headers]
            if "question" in headers_lower and "answer" in headers_lower:
                question_column = headers[headers_lower.index("question")]
                answer_column = headers[headers_lower.index("answer")]
            elif "q" in headers_lower and "a" in headers_lower and len(headers) == 2:
                question_column = headers[headers_lower.index("q")]
                answer_column = headers[headers_lower.index("a")]
            elif "text" in headers_lower:
                text_column = headers[headers_lower.index("text")]
            elif len(headers) == 1:
                text_column = headers[0]
            else:
                # Default: first column = question, second = answer if two columns
                if len(headers) >= 2:
                    question_column, answer_column = headers[0], headers[1]
                else:
                    text_column = headers[0]

        for row in rows:
            if question_column and answer_column:
                q = (row.get(question_column) or "").strip()
                a = (row.get(answer_column) or "").strip()
                if q and a:
                    qa_pairs.append((q, a))
            elif text_column:
                t = (row.get(text_column) or "").strip()
                if t:
                    texts.append(t)
            else:
                # Fallback: concat all fields as one text
                texts.append(" ".join(str(v or "").strip() for v in row.values()).strip())

    return texts, qa_pairs
