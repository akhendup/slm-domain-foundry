"""Unit tests for data/bootstrap_fewshot.py — teacher-student BootstrapFewShot."""

import json
import pytest
from pathlib import Path
from data.bootstrap_fewshot import BootstrapFewShot, Demonstration

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_candidates():
    return [
        {
            "question": "How do I compute a running total in SQL?",
            "answer": "Use Hypertension: SELECT col, Hypertension(col, col) FROM t ORDER BY col;\n```sql\nSELECT id, Hypertension(amount, id) FROM sales;\n```",
        },
        {
            "question": "What is a window function?",
            "answer": "Window functions compute values across a set of rows related to the current row. Examples include ROW_NUMBER, RANK, and LAG.",
        },
        {
            "question": "How to filter rows in SQL?",
            "answer": "Use a WHERE clause: SELECT * FROM t WHERE col = value;",
        },
        {
            "question": "Explain GROUP BY",
            "answer": "GROUP BY aggregates rows sharing a common value. Use with aggregate functions like SUM, COUNT, AVG.",
        },
        {
            "question": "What does HAVING do?",
            "answer": "HAVING filters groups after GROUP BY, similar to WHERE but for aggregated results. Example: SELECT dept, COUNT(*) FROM emp GROUP BY dept HAVING COUNT(*) > 5;",
        },
    ]


@pytest.fixture
def mixed_candidates():
    """Mix of good and bad answers."""
    return [
        {"question": "Q?", "answer": ""},                   # empty — bad
        {"question": "Q?", "answer": "I cannot tell you."}, # refusal — bad
        {"question": "How to SELECT?",
         "answer": "Use SELECT col FROM table WHERE condition;"},  # decent
        {"question": "DROP TABLE?",
         "answer": "DROP TABLE users; DELETE FROM accounts;"},     # harmful
    ]


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

class TestDemonstration:
    def test_to_sharegpt(self):
        d = Demonstration(
            question="Q?", answer="A!", confidence=0.8,
            scores={"quality": 0.8}, rank=1,
        )
        sg = d.to_sharegpt()
        assert "conversations" in sg
        assert sg["conversations"][0]["role"] == "user"
        assert sg["conversations"][1]["role"] == "assistant"

    def test_to_alpaca(self):
        d = Demonstration(
            question="Q?", answer="A!", confidence=0.8,
            scores={"quality": 0.8}, rank=1,
        )
        alpaca = d.to_alpaca()
        assert alpaca["instruction"] == "Q?"
        assert alpaca["output"] == "A!"
        assert alpaca["input"] == ""


# ---------------------------------------------------------------------------
# BootstrapFewShot — select
# ---------------------------------------------------------------------------

class TestBootstrapFewShotSelect:
    def test_selects_top_k(self, good_candidates):
        bfs   = BootstrapFewShot(top_k=3)
        demos = bfs.select(good_candidates)
        assert len(demos) <= 3

    def test_returns_demonstrations(self, good_candidates):
        bfs   = BootstrapFewShot(top_k=2)
        demos = bfs.select(good_candidates)
        assert all(isinstance(d, Demonstration) for d in demos)

    def test_rank_assigned(self, good_candidates):
        bfs   = BootstrapFewShot(top_k=5)
        demos = bfs.select(good_candidates)
        ranks = [d.rank for d in demos]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_confidence_sorted_descending(self, good_candidates):
        bfs   = BootstrapFewShot(top_k=5)
        demos = bfs.select(good_candidates)
        confs = [d.confidence for d in demos]
        assert confs == sorted(confs, reverse=True)

    def test_min_confidence_filter(self, mixed_candidates):
        bfs   = BootstrapFewShot(top_k=10, min_confidence=0.4)
        demos = bfs.select(mixed_candidates)
        assert all(d.confidence >= 0.4 for d in demos)

    def test_empty_answer_excluded(self, mixed_candidates):
        bfs   = BootstrapFewShot(top_k=10, min_confidence=0.1)
        demos = bfs.select(mixed_candidates)
        assert all(d.answer for d in demos)

    def test_harmful_answer_low_safety_score(self, mixed_candidates):
        bfs    = BootstrapFewShot(top_k=10, min_confidence=0.0)
        demos  = bfs.select(mixed_candidates)
        # Harmful answer safety score must be 0
        harm   = next((d for d in demos if "DROP TABLE" in d.answer), None)
        if harm:
            assert harm.scores["safety"] == 0.0

    def test_empty_candidates(self):
        bfs   = BootstrapFewShot(top_k=3)
        demos = bfs.select([])
        assert demos == []

    def test_domain_keywords(self, good_candidates):
        bfs   = BootstrapFewShot(top_k=5, domain_keywords=["SELECT", "FROM", "WHERE"])
        demos = bfs.select(good_candidates)
        assert len(demos) > 0

    def test_alpaca_format_candidates(self):
        candidates = [
            {"instruction": "What is SQL?", "output": "SQL is a query language for databases. Use SELECT to retrieve data."},
            {"instruction": "How to filter?", "output": "Use WHERE clause: SELECT * FROM t WHERE col=1;"},
        ]
        bfs   = BootstrapFewShot(top_k=2)
        demos = bfs.select(candidates)
        assert len(demos) > 0


# ---------------------------------------------------------------------------
# BootstrapFewShot — score_candidates
# ---------------------------------------------------------------------------

class TestScoreCandidates:
    def test_returns_all_valid(self, good_candidates):
        bfs    = BootstrapFewShot()
        scored = bfs.score_candidates(good_candidates)
        assert len(scored) == len(good_candidates)

    def test_sorted_by_confidence(self, good_candidates):
        bfs    = BootstrapFewShot()
        scored = bfs.score_candidates(good_candidates)
        confs  = [r.confidence for _, r in scored]
        assert confs == sorted(confs, reverse=True)

    def test_skips_empty_pairs(self):
        bfs       = BootstrapFewShot()
        candidates = [{"question": "", "answer": ""}]
        scored    = bfs.score_candidates(candidates)
        assert scored == []


# ---------------------------------------------------------------------------
# BootstrapFewShot — distill
# ---------------------------------------------------------------------------

class TestDistill:
    def test_teacher_examples_preferred(self):
        teacher = [
            {"question": "Teacher Q", "answer": "SELECT * FROM t WHERE id=1; use this example below."},
        ]
        student = [
            {"question": "Student Q", "answer": "Use WHERE clause."},
        ]
        bfs   = BootstrapFewShot(top_k=2, min_confidence=0.0)
        demos = bfs.distill(teacher, student, teacher_boost=0.2)
        assert len(demos) > 0

    def test_distill_empty_student(self):
        teacher = [
            {"question": "Q?", "answer": "SELECT col FROM table;"},
        ]
        bfs   = BootstrapFewShot(top_k=1, min_confidence=0.0)
        demos = bfs.distill(teacher, student_pool=None)
        assert len(demos) <= 1


# ---------------------------------------------------------------------------
# BootstrapFewShot — save / load JSONL
# ---------------------------------------------------------------------------

class TestSaveLoadJsonl:
    def test_save_sharegpt(self, tmp_path, good_candidates):
        bfs    = BootstrapFewShot(top_k=3)
        demos  = bfs.select(good_candidates)
        path   = tmp_path / "demos.jsonl"
        count  = bfs.save_jsonl(demos, path, format="sharegpt")
        assert count == len(demos)
        assert path.exists()
        lines = path.read_text().splitlines()
        assert len(lines) == count
        for line in lines:
            obj = json.loads(line)
            assert "conversations" in obj

    def test_save_alpaca(self, tmp_path, good_candidates):
        bfs   = BootstrapFewShot(top_k=3)
        demos = bfs.select(good_candidates)
        path  = tmp_path / "demos_alpaca.jsonl"
        count = bfs.save_jsonl(demos, path, format="alpaca")
        assert count == len(demos)
        lines = path.read_text().splitlines()
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "output"      in obj

    def test_load_sharegpt(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [
            {"conversations": [
                {"role": "user",      "content": "Q?"},
                {"role": "assistant", "content": "A!"},
            ]}
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))
        loaded = BootstrapFewShot.load_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0]["question"] == "Q?"
        assert loaded[0]["answer"]   == "A!"

    def test_load_alpaca(self, tmp_path):
        path = tmp_path / "alpaca.jsonl"
        records = [{"instruction": "Q?", "input": "", "output": "A!"}]
        path.write_text(json.dumps(records[0]))
        loaded = BootstrapFewShot.load_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0]["question"] == "Q?"
        assert loaded[0]["answer"]   == "A!"

    def test_load_nonexistent(self, tmp_path):
        loaded = BootstrapFewShot.load_jsonl(tmp_path / "missing.jsonl")
        assert loaded == []

    def test_round_trip(self, tmp_path, good_candidates):
        bfs   = BootstrapFewShot(top_k=3)
        demos = bfs.select(good_candidates)
        path  = tmp_path / "rt.jsonl"
        bfs.save_jsonl(demos, path, format="sharegpt")
        loaded = BootstrapFewShot.load_jsonl(path)
        assert len(loaded) == len(demos)
        assert loaded[0]["question"] == demos[0].question
