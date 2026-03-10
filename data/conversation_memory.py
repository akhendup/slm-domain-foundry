#!/usr/bin/env python3
"""
Conversation Memory — persistent storage of every chat interaction.

Every question the user asks and every answer the model gives is logged here.
This serves three roles:

  1. IMMEDIATE RETRIEVAL  — the KnowledgeRetriever can surface past Q&A as
                            context so the running model behaves as if it
                            already knows what it was asked before.

  2. TRAINING PIPELINE   — approved interactions are exported as ShareGPT
                            JSONL and merged into the next fine-tuning run,
                            closing the human-in-the-loop cycle.

  3. PATTERN MINING      — frequent or highly-rated interactions surface
                            candidate Knowledge Library entries and patterns
                            that an agent or the user can promote.

Storage layout:
    conversation_memory/
        interactions.jsonl      ← append-only log of ALL interactions
        _stats.json             ← running counts (rebuilt on demand)

Each line of interactions.jsonl is a JSON object:
    {
        "id":         "<uuid4 hex>",
        "timestamp":  "<ISO-8601 UTC>",
        "session_id": "<session string>",
        "question":   "<user message>",
        "answer":     "<model reply>",
        "approved":   null | true | false,
        "model_name": "<active model name or empty>",
        "kb_context_used": true | false
    }
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _interactions_file(memory_dir: Path) -> Path:
    return memory_dir / "interactions.jsonl"


def _load_raw(memory_dir: Path) -> List[Dict[str, Any]]:
    """Load all interaction records from disk."""
    path = _interactions_file(memory_dir)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _write_raw(memory_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Overwrite the interactions file with all records (used for updates)."""
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = _interactions_file(memory_dir)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def log_interaction(
    memory_dir: Path,
    question: str,
    answer: str,
    session_id: str = "",
    model_name: str = "",
    kb_context_used: bool = False,
) -> str:
    """
    Append one interaction to memory.  Returns the new record's ID.
    Thread-safe via atomic append (open mode "a").
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid.uuid4().hex,
        "timestamp": _now_iso(),
        "session_id": session_id,
        "question": question.strip(),
        "answer": answer.strip(),
        "approved": None,       # pending review
        "model_name": model_name,
        "kb_context_used": kb_context_used,
    }
    with open(_interactions_file(memory_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["id"]


def load_interactions(
    memory_dir: Path,
    approved_only: bool = False,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Return interaction records, newest-first.

    Args:
        approved_only: if True, only return records where approved=True.
        session_id:    if set, filter to that session.
        limit:         cap the returned list.
    """
    records = _load_raw(memory_dir)
    records = list(reversed(records))   # newest first
    if approved_only:
        records = [r for r in records if r.get("approved") is True]
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    if limit:
        records = records[:limit]
    return records


def set_approval(memory_dir: Path, record_id: str, approved: bool) -> bool:
    """
    Approve (True) or reject (False) an interaction by ID.
    Rewrites the full file — suitable for small-to-medium memory stores.
    Returns True if the record was found and updated.
    """
    records = _load_raw(memory_dir)
    found = False
    for r in records:
        if r.get("id") == record_id:
            r["approved"] = approved
            found = True
            break
    if found:
        _write_raw(memory_dir, records)
    return found


def memory_stats(memory_dir: Path) -> Dict[str, int]:
    """Return counts: total, pending, approved, rejected."""
    records = _load_raw(memory_dir)
    approved = sum(1 for r in records if r.get("approved") is True)
    rejected = sum(1 for r in records if r.get("approved") is False)
    pending  = sum(1 for r in records if r.get("approved") is None)
    return {
        "total":    len(records),
        "approved": approved,
        "rejected": rejected,
        "pending":  pending,
    }


def export_approved_to_jsonl(
    memory_dir: Path,
    output_path: Path,
) -> int:
    """
    Write approved interactions as ShareGPT-format JSONL ready for training.

    Each record becomes:
        {"conversations": [
            {"role": "user",      "content": "<question>"},
            {"role": "assistant", "content": "<answer>"}
        ]}

    Returns the number of examples written.
    """
    approved = load_interactions(memory_dir, approved_only=True)
    if not approved:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for r in reversed(approved):  # chronological order for training
            example = {
                "conversations": [
                    {"role": "user",      "content": r["question"]},
                    {"role": "assistant", "content": r["answer"]},
                ]
            }
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Pattern mining — surface candidate patterns from memory
# ---------------------------------------------------------------------------

def mine_frequent_questions(
    memory_dir: Path,
    min_count: int = 2,
    approved_only: bool = False,
) -> List[Tuple[str, int, str]]:
    """
    Find questions asked multiple times (potential canonical Q&A candidates).

    Returns list of (question, count, latest_answer), sorted by count desc.
    This output can be promoted to the Knowledge Library as YAML patterns.
    """
    records = load_interactions(memory_dir, approved_only=approved_only)
    from collections import Counter
    # Normalize: lower-case, strip punctuation for grouping
    import re
    def _norm(q: str) -> str:
        return re.sub(r"[^\w\s]", "", q.lower()).strip()

    groups: Dict[str, List[Dict]] = {}
    for r in records:
        key = _norm(r["question"])
        groups.setdefault(key, []).append(r)

    results = []
    for key, group in groups.items():
        if len(group) >= min_count:
            # Use the most recent answer
            latest = sorted(group, key=lambda x: x.get("timestamp", ""), reverse=True)[0]
            results.append((latest["question"], len(group), latest["answer"]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
