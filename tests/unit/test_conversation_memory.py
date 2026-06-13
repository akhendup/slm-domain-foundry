"""Unit tests for data/conversation_memory.py"""
import json
from pathlib import Path

import pytest

from data.conversation_memory import (
    _now_iso,
    _interactions_file,
    _load_raw,
    _write_raw,
    export_approved_to_jsonl,
    load_interactions,
    log_interaction,
    memory_stats,
    mine_frequent_questions,
    set_approval,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _now_iso
# ---------------------------------------------------------------------------

class TestNowIso:
    def test_returns_string(self):
        ts = _now_iso()
        assert isinstance(ts, str)

    def test_ends_with_z(self):
        assert _now_iso().endswith("Z")

    def test_contains_t_separator(self):
        assert "T" in _now_iso()

    def test_reasonable_length(self):
        assert len(_now_iso()) >= 20


# ---------------------------------------------------------------------------
# _interactions_file
# ---------------------------------------------------------------------------

class TestInteractionsFile:
    def test_returns_path_in_dir(self, tmp_path):
        result = _interactions_file(tmp_path)
        assert result == tmp_path / "interactions.jsonl"

    def test_returns_path_object(self, tmp_path):
        assert isinstance(_interactions_file(tmp_path), Path)


# ---------------------------------------------------------------------------
# _load_raw
# ---------------------------------------------------------------------------

class TestLoadRaw:
    def test_empty_when_no_file(self, tmp_path):
        assert _load_raw(tmp_path) == []

    def test_loads_valid_records(self, tmp_path):
        records = [
            {"id": "abc123", "question": "Q1", "answer": "A1"},
            {"id": "def456", "question": "Q2", "answer": "A2"},
        ]
        path = _interactions_file(tmp_path)
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )
        result = _load_raw(tmp_path)
        assert len(result) == 2
        assert result[0]["id"] == "abc123"

    def test_skips_empty_lines(self, tmp_path):
        path = _interactions_file(tmp_path)
        path.write_text('{"id": "x"}\n\n\n{"id": "y"}\n', encoding="utf-8")
        result = _load_raw(tmp_path)
        assert len(result) == 2

    def test_skips_invalid_json(self, tmp_path):
        path = _interactions_file(tmp_path)
        path.write_text('{"id": "x"}\nNOT_JSON\n{"id": "y"}\n', encoding="utf-8")
        result = _load_raw(tmp_path)
        assert len(result) == 2

    def test_empty_file_returns_empty(self, tmp_path):
        _interactions_file(tmp_path).write_text("", encoding="utf-8")
        assert _load_raw(tmp_path) == []


# ---------------------------------------------------------------------------
# _write_raw
# ---------------------------------------------------------------------------

class TestWriteRaw:
    def test_writes_records(self, tmp_path):
        records = [{"id": "1", "q": "Q"}, {"id": "2", "q": "R"}]
        _write_raw(tmp_path, records)
        path = _interactions_file(tmp_path)
        assert path.exists()
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_creates_dir_if_missing(self, tmp_path):
        sub = tmp_path / "subdir" / "nested"
        _write_raw(sub, [{"id": "1"}])
        assert _interactions_file(sub).exists()

    def test_roundtrip(self, tmp_path):
        records = [{"id": "abc", "value": 42}]
        _write_raw(tmp_path, records)
        assert _load_raw(tmp_path) == records

    def test_overwrites_existing(self, tmp_path):
        _write_raw(tmp_path, [{"id": "old"}])
        _write_raw(tmp_path, [{"id": "new1"}, {"id": "new2"}])
        result = _load_raw(tmp_path)
        assert len(result) == 2
        assert result[0]["id"] == "new1"


# ---------------------------------------------------------------------------
# log_interaction
# ---------------------------------------------------------------------------

class TestLogInteraction:
    def test_returns_string_id(self, tmp_path):
        rid = log_interaction(tmp_path, "Q?", "A.")
        assert isinstance(rid, str) and rid

    def test_creates_file(self, tmp_path):
        log_interaction(tmp_path, "Q?", "A.")
        assert _interactions_file(tmp_path).exists()

    def test_record_has_expected_fields(self, tmp_path):
        rid = log_interaction(
            tmp_path, "Q?", "A.",
            session_id="sess1", model_name="model_x", kb_context_used=True,
        )
        records = _load_raw(tmp_path)
        assert len(records) == 1
        r = records[0]
        assert r["id"] == rid
        assert r["question"] == "Q?"
        assert r["answer"] == "A."
        assert r["session_id"] == "sess1"
        assert r["model_name"] == "model_x"
        assert r["kb_context_used"] is True
        assert r["approved"] is None

    def test_strips_whitespace(self, tmp_path):
        log_interaction(tmp_path, "  Q?  ", "  A.  ")
        r = _load_raw(tmp_path)[0]
        assert r["question"] == "Q?"
        assert r["answer"] == "A."

    def test_multiple_appended(self, tmp_path):
        for i in range(5):
            log_interaction(tmp_path, f"Q{i}?", f"A{i}.")
        assert len(_load_raw(tmp_path)) == 5

    def test_creates_memory_dir(self, tmp_path):
        sub = tmp_path / "memory" / "nested"
        log_interaction(sub, "Q?", "A.")
        assert sub.exists()

    def test_unique_ids(self, tmp_path):
        ids = [log_interaction(tmp_path, f"Q{i}?", f"A{i}.") for i in range(10)]
        assert len(set(ids)) == 10


# ---------------------------------------------------------------------------
# load_interactions
# ---------------------------------------------------------------------------

class TestLoadInteractions:
    def _populate(self, tmp_path, n=5):
        ids = []
        for i in range(n):
            ids.append(log_interaction(tmp_path, f"Q{i}?", f"A{i}."))
        return ids

    def test_empty_memory_dir(self, tmp_path):
        assert load_interactions(tmp_path) == []

    def test_newest_first(self, tmp_path):
        self._populate(tmp_path, 3)
        records = load_interactions(tmp_path)
        assert records[0]["question"] == "Q2?"

    def test_approved_only_filter(self, tmp_path):
        ids = self._populate(tmp_path, 4)
        set_approval(tmp_path, ids[1], True)
        set_approval(tmp_path, ids[3], True)
        approved = load_interactions(tmp_path, approved_only=True)
        assert len(approved) == 2
        assert all(r["approved"] is True for r in approved)

    def test_session_id_filter(self, tmp_path):
        log_interaction(tmp_path, "Q1?", "A1.", session_id="s1")
        log_interaction(tmp_path, "Q2?", "A2.", session_id="s2")
        log_interaction(tmp_path, "Q3?", "A3.", session_id="s1")
        filtered = load_interactions(tmp_path, session_id="s1")
        assert len(filtered) == 2
        assert all(r["session_id"] == "s1" for r in filtered)

    def test_limit(self, tmp_path):
        self._populate(tmp_path, 10)
        records = load_interactions(tmp_path, limit=3)
        assert len(records) == 3

    def test_returns_all_by_default(self, tmp_path):
        self._populate(tmp_path, 7)
        assert len(load_interactions(tmp_path)) == 7


# ---------------------------------------------------------------------------
# set_approval
# ---------------------------------------------------------------------------

class TestSetApproval:
    def test_approve(self, tmp_path):
        rid = log_interaction(tmp_path, "Q?", "A.")
        assert set_approval(tmp_path, rid, True) is True
        assert _load_raw(tmp_path)[0]["approved"] is True

    def test_reject(self, tmp_path):
        rid = log_interaction(tmp_path, "Q?", "A.")
        assert set_approval(tmp_path, rid, False) is True
        assert _load_raw(tmp_path)[0]["approved"] is False

    def test_not_found_returns_false(self, tmp_path):
        log_interaction(tmp_path, "Q?", "A.")
        assert set_approval(tmp_path, "nonexistent_id", True) is False

    def test_only_target_updated(self, tmp_path):
        r1 = log_interaction(tmp_path, "Q1?", "A1.")
        r2 = log_interaction(tmp_path, "Q2?", "A2.")
        set_approval(tmp_path, r1, True)
        by_id = {r["id"]: r["approved"] for r in _load_raw(tmp_path)}
        assert by_id[r1] is True
        assert by_id[r2] is None

    def test_can_change_approval(self, tmp_path):
        rid = log_interaction(tmp_path, "Q?", "A.")
        set_approval(tmp_path, rid, True)
        set_approval(tmp_path, rid, False)
        assert _load_raw(tmp_path)[0]["approved"] is False


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------

class TestMemoryStats:
    def test_empty(self, tmp_path):
        stats = memory_stats(tmp_path)
        assert stats == {"total": 0, "approved": 0, "rejected": 0, "pending": 0}

    def test_counts(self, tmp_path):
        r1 = log_interaction(tmp_path, "Q1?", "A1.")
        r2 = log_interaction(tmp_path, "Q2?", "A2.")
        r3 = log_interaction(tmp_path, "Q3?", "A3.")
        r4 = log_interaction(tmp_path, "Q4?", "A4.")
        set_approval(tmp_path, r1, True)
        set_approval(tmp_path, r2, False)
        # r3, r4 remain pending
        stats = memory_stats(tmp_path)
        assert stats["total"] == 4
        assert stats["approved"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 2

    def test_all_pending(self, tmp_path):
        for i in range(3):
            log_interaction(tmp_path, f"Q{i}?", f"A{i}.")
        stats = memory_stats(tmp_path)
        assert stats["pending"] == 3
        assert stats["approved"] == 0
        assert stats["rejected"] == 0


# ---------------------------------------------------------------------------
# export_approved_to_jsonl
# ---------------------------------------------------------------------------

class TestExportApprovedToJsonl:
    def test_empty_memory_returns_zero(self, tmp_path):
        out = tmp_path / "exported.jsonl"
        assert export_approved_to_jsonl(tmp_path, out) == 0
        assert not out.exists()

    def test_no_approved_returns_zero(self, tmp_path):
        log_interaction(tmp_path, "Q?", "A.")
        out = tmp_path / "exported.jsonl"
        assert export_approved_to_jsonl(tmp_path, out) == 0

    def test_exports_approved_only(self, tmp_path):
        r1 = log_interaction(tmp_path, "Q1?", "A1.")
        r2 = log_interaction(tmp_path, "Q2?", "A2.")
        log_interaction(tmp_path, "Q3?", "A3.")  # pending
        set_approval(tmp_path, r1, True)
        set_approval(tmp_path, r2, True)
        out = tmp_path / "exported.jsonl"
        count = export_approved_to_jsonl(tmp_path, out)
        assert count == 2
        lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_sharegpt_format(self, tmp_path):
        rid = log_interaction(tmp_path, "My question", "My answer")
        set_approval(tmp_path, rid, True)
        out = tmp_path / "approved.jsonl"
        export_approved_to_jsonl(tmp_path, out)
        lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
        convs = lines[0]["conversations"]
        assert convs[0] == {"role": "user", "content": "My question"}
        assert convs[1] == {"role": "assistant", "content": "My answer"}

    def test_creates_parent_dirs(self, tmp_path):
        rid = log_interaction(tmp_path, "Q?", "A.")
        set_approval(tmp_path, rid, True)
        out = tmp_path / "sub" / "dir" / "approved.jsonl"
        export_approved_to_jsonl(tmp_path, out)
        assert out.exists()

    def test_rejected_not_exported(self, tmp_path):
        r1 = log_interaction(tmp_path, "Q1?", "A1.")
        r2 = log_interaction(tmp_path, "Q2?", "A2.")
        set_approval(tmp_path, r1, True)
        set_approval(tmp_path, r2, False)
        out = tmp_path / "out.jsonl"
        count = export_approved_to_jsonl(tmp_path, out)
        assert count == 1


# ---------------------------------------------------------------------------
# mine_frequent_questions
# ---------------------------------------------------------------------------

class TestMineFrequentQuestions:
    def test_empty_memory(self, tmp_path):
        assert mine_frequent_questions(tmp_path) == []

    def test_no_repeats_below_min_count(self, tmp_path):
        log_interaction(tmp_path, "What is hypertension?", "Hypertension is cumulative sum.")
        log_interaction(tmp_path, "What is RANK?", "RANK assigns a rank.")
        assert mine_frequent_questions(tmp_path, min_count=2) == []

    def test_finds_repeated_questions(self, tmp_path):
        for _ in range(3):
            log_interaction(tmp_path, "What is hypertension?", "Hypertension is cumulative sum.")
        log_interaction(tmp_path, "What is RANK?", "RANK assigns a rank.")
        result = mine_frequent_questions(tmp_path, min_count=2)
        assert len(result) == 1
        q, count, answer = result[0]
        assert "hypertension" in q.lower() or "Hypertension" in q
        assert count == 3

    def test_sorted_by_count_desc(self, tmp_path):
        for _ in range(5):
            log_interaction(tmp_path, "Top question?", "Top answer.")
        for _ in range(2):
            log_interaction(tmp_path, "Second question?", "Second answer.")
        result = mine_frequent_questions(tmp_path, min_count=2)
        assert len(result) == 2
        assert result[0][1] >= result[1][1]

    def test_approved_only_filter(self, tmp_path):
        for _ in range(3):
            rid = log_interaction(tmp_path, "Approved Q?", "Approved A.")
            set_approval(tmp_path, rid, True)
        for _ in range(3):
            log_interaction(tmp_path, "Unapproved Q?", "Unapproved A.")
        result = mine_frequent_questions(tmp_path, min_count=2, approved_only=True)
        questions = [q for q, _, _ in result]
        assert all("Approved" in q for q in questions)

    def test_returns_tuple_structure(self, tmp_path):
        for _ in range(2):
            log_interaction(tmp_path, "What is MSUM?", "MSUM is a moving sum.")
        result = mine_frequent_questions(tmp_path, min_count=2)
        assert len(result) >= 1
        q, count, answer = result[0]
        assert isinstance(q, str)
        assert isinstance(count, int)
        assert isinstance(answer, str)

    def test_normalises_case_and_punctuation(self, tmp_path):
        log_interaction(tmp_path, "What is hypertension?", "Answer 1.")
        log_interaction(tmp_path, "what is csum", "Answer 2.")
        log_interaction(tmp_path, "What is hypertension!", "Answer 3.")
        result = mine_frequent_questions(tmp_path, min_count=2)
        # All three normalise to the same key
        assert len(result) >= 1
        assert result[0][1] >= 2
