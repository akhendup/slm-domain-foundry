"""Unit tests for data/prepare_training_data.py"""
import json
import textwrap

import pytest

from data.prepare_training_data import (
    build_alpaca_examples,
    build_sharegpt_examples,
    save_jsonl,
    text_to_qa_heuristic,
    _split_train_val,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# build_alpaca_examples
# ---------------------------------------------------------------------------

class TestBuildAlpacaExamples:
    def test_structure(self, sample_qa_pairs):
        result = build_alpaca_examples(sample_qa_pairs)
        assert len(result) == len(sample_qa_pairs)
        for ex, (q, a) in zip(result, sample_qa_pairs):
            assert ex["instruction"] == q
            assert ex["input"] == ""
            assert ex["output"] == a

    def test_empty_input(self):
        assert build_alpaca_examples([]) == []

    def test_all_keys_present(self, sample_qa_pairs):
        for ex in build_alpaca_examples(sample_qa_pairs):
            assert set(ex.keys()) == {"instruction", "input", "output"}


# ---------------------------------------------------------------------------
# build_sharegpt_examples
# ---------------------------------------------------------------------------

class TestBuildShareGptExamples:
    def test_structure(self, sample_qa_pairs):
        result = build_sharegpt_examples(sample_qa_pairs)
        assert len(result) == len(sample_qa_pairs)
        for ex, (q, a) in zip(result, sample_qa_pairs):
            convs = ex["conversations"]
            assert len(convs) == 2
            assert convs[0] == {"role": "user", "content": q}
            assert convs[1] == {"role": "assistant", "content": a}

    def test_empty_input(self):
        assert build_sharegpt_examples([]) == []

    def test_top_level_key(self, sample_qa_pairs):
        for ex in build_sharegpt_examples(sample_qa_pairs):
            assert "conversations" in ex
            assert isinstance(ex["conversations"], list)


# ---------------------------------------------------------------------------
# save_jsonl
# ---------------------------------------------------------------------------

class TestSaveJsonl:
    def test_creates_file(self, tmp_path, sample_alpaca_examples):
        out = tmp_path / "out.jsonl"
        save_jsonl(sample_alpaca_examples, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path, sample_alpaca_examples):
        out = tmp_path / "sub" / "dir" / "out.jsonl"
        save_jsonl(sample_alpaca_examples, out)
        assert out.exists()

    def test_one_json_object_per_line(self, tmp_path, sample_alpaca_examples):
        out = tmp_path / "out.jsonl"
        save_jsonl(sample_alpaca_examples, out)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(sample_alpaca_examples)
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_roundtrip_content(self, tmp_path, sample_alpaca_examples):
        out = tmp_path / "out.jsonl"
        save_jsonl(sample_alpaca_examples, out)
        loaded = [json.loads(ln) for ln in out.read_text().strip().split("\n")]
        assert loaded == sample_alpaca_examples

    def test_empty_list_creates_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        save_jsonl([], out)
        assert out.exists()
        assert out.read_text().strip() == ""


# ---------------------------------------------------------------------------
# _split_train_val
# ---------------------------------------------------------------------------

class TestSplitTrainVal:
    def test_basic_split(self):
        items = list(range(100))
        train, val = _split_train_val(items, val_ratio=0.15)
        assert len(train) + len(val) == 100
        assert len(val) == 15

    def test_minimum_one_val(self):
        items = list(range(3))
        train, val = _split_train_val(items, val_ratio=0.10)
        assert len(val) >= 1

    def test_no_overlap(self):
        items = list(range(50))
        train, val = _split_train_val(items, val_ratio=0.2)
        assert set(train) & set(val) == set()

    def test_order_preserved(self):
        items = list(range(20))
        train, val = _split_train_val(items, val_ratio=0.2)
        assert train == items[:16]
        assert val == items[16:]


# ---------------------------------------------------------------------------
# text_to_qa_heuristic
# ---------------------------------------------------------------------------

class TestTextToQaHeuristic:
    def test_heading_based_qa(self):
        # Heading must match _is_heading_line: title-case phrase or all-caps
        # "Cumulative Sum" matches ^(?:[A-Z][a-z]+\s+){1,6}[A-Z][a-z]+$
        chunk = "Cumulative Sum\n\nThe cumulative sum function returns a running total over a window."
        result = text_to_qa_heuristic([chunk], source="manual.pdf")
        assert len(result) >= 1
        qs = [q for q, _ in result]
        assert any("Cumulative" in q or "Sum" in q for q in qs)

    def test_sql_chunk_generates_qa(self):
        chunk = textwrap.dedent("""\
            SELECT id, CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) AS total
            FROM orders;
        """)
        result = text_to_qa_heuristic([chunk], source="ref.pdf")
        assert len(result) >= 1

    def test_generic_fallback_for_long_chunk(self):
        chunk = "This is a long paragraph with more than fifty characters of content here."
        result = text_to_qa_heuristic([chunk], source="doc")
        assert len(result) >= 1

    def test_skips_short_chunks(self):
        result = text_to_qa_heuristic(["short"], source="doc")
        assert result == []

    def test_empty_list_returns_empty(self):
        assert text_to_qa_heuristic([], source="doc") == []

    def test_returns_list_of_tuples(self):
        chunk = "Window Functions\n\nWindow functions compute aggregates over a set of rows."
        result = text_to_qa_heuristic([chunk])
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    def test_questions_are_nonempty(self):
        chunks = [
            "Analytic Functions\n\nAnalytic functions compute values over groups of rows.",
            "SELECT id, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) FROM emp;",
        ]
        result = text_to_qa_heuristic(chunks)
        for q, a in result:
            assert q.strip() != ""
            assert a.strip() != ""
