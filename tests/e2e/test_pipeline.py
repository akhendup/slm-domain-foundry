"""
End-to-end pipeline tests.
These tests exercise multiple modules together using real file I/O but no model weights.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from data.chunking import chunk_text, chunk_text_sql_aware
from data.csv_loader import load_csv
from data.prepare_training_data import (
    build_alpaca_examples,
    build_sharegpt_examples,
    save_jsonl,
    text_to_qa_heuristic,
    _split_train_val,
)
from data.yaml_pattern_loader import load_patterns_as_qa, load_yaml_patterns_dir


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# CSV → training data pipeline
# ---------------------------------------------------------------------------

class TestCsvToTrainingDataPipeline:
    def test_qa_csv_to_sharegpt_jsonl(self, tmp_path, tmp_csv):
        csv_path = tmp_csv([
            {"question": "What is CSUM?", "answer": "CSUM computes a cumulative sum."},
            {"question": "What is RANK?", "answer": "RANK assigns a rank to each row."},
            {"question": "What is MSUM?", "answer": "MSUM computes a moving sum."},
        ])
        texts, qa = load_csv(csv_path)
        assert len(qa) == 3

        examples = build_sharegpt_examples(qa)
        out = tmp_path / "train_sharegpt.jsonl"
        save_jsonl(examples, out)

        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().strip().split("\n")]
        assert len(lines) == 3
        assert lines[0]["conversations"][0]["content"] == "What is CSUM?"

    def test_text_csv_to_alpaca_jsonl(self, tmp_path, tmp_csv):
        csv_path = tmp_csv([
            {"text": "CSUM computes a cumulative sum over a window partition."},
            {"text": "RANK assigns an integer rank to each row within a partition."},
        ])
        texts, qa = load_csv(csv_path)
        assert len(texts) == 2

        chunks = []
        for t in texts:
            chunks.extend(chunk_text(t, chunk_size=500))
        qa_pairs = text_to_qa_heuristic(chunks, source="docs.csv")
        examples = build_alpaca_examples(qa_pairs)
        out = tmp_path / "train_alpaca.jsonl"
        save_jsonl(examples, out)

        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        assert len(lines) >= 1
        for line in lines:
            assert "instruction" in line and "output" in line

    def test_train_val_split_pipeline(self, tmp_path, tmp_csv):
        rows = [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(20)]
        csv_path = tmp_csv(rows)
        _, qa = load_csv(csv_path)
        examples = build_sharegpt_examples(qa)

        train, val = _split_train_val(examples, val_ratio=0.2)
        save_jsonl(train, tmp_path / "train.jsonl")
        save_jsonl(val, tmp_path / "val.jsonl")

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "val.jsonl").exists()
        assert len(train) == 16
        assert len(val) == 4


# ---------------------------------------------------------------------------
# Text chunking → QA → JSONL pipeline
# ---------------------------------------------------------------------------

class TestChunkingToQaPipeline:
    def test_plain_text_full_pipeline(self, tmp_path):
        text = textwrap.dedent("""\
            Window Functions Overview

            Window functions compute aggregate values over a set of rows.
            They use the OVER clause with optional PARTITION BY and ORDER BY.

            CSUM Function

            The CSUM function returns a cumulative sum.
            It accumulates values in the order specified by ORDER BY.
            Use PARTITION BY to reset the sum for each group.

            RANK Function

            The RANK function assigns an integer rank to each row.
            Rows with equal values receive the same rank.
            Gaps appear after ties: ranks go 1, 1, 3, not 1, 1, 2.
        """)
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
        assert len(chunks) >= 1

        qa = text_to_qa_heuristic(chunks, source="manual.pdf")
        assert len(qa) >= 1

        examples = build_sharegpt_examples(qa)
        out = tmp_path / "out.jsonl"
        save_jsonl(examples, out)

        lines = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        assert len(lines) >= 1
        for line in lines:
            convs = line["conversations"]
            assert convs[0]["role"] == "user"
            assert convs[1]["role"] == "assistant"

    def test_sql_text_full_pipeline(self, tmp_path, sample_sql_text):
        chunks = chunk_text_sql_aware(sample_sql_text, chunk_size=300, chunk_overlap=50)
        qa = text_to_qa_heuristic(chunks, source="sql_ref.pdf")
        examples = build_alpaca_examples(qa)
        out = tmp_path / "sql_out.jsonl"
        save_jsonl(examples, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# YAML pattern → training data pipeline
# ---------------------------------------------------------------------------

class TestYamlPatternPipeline:
    def test_single_pattern_to_qa_jsonl(self, tmp_path, sample_yaml_pattern):
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")

        qa_pairs, _ = load_patterns_as_qa(pattern_dir)
        assert len(qa_pairs) >= 5

        examples = build_sharegpt_examples(qa_pairs)
        out = tmp_path / "patterns_train.jsonl"
        save_jsonl(examples, out)

        lines = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == len(qa_pairs)

    def test_multiple_patterns_combined(self, tmp_path, sample_yaml_pattern):
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        # Second pattern
        second = sample_yaml_pattern.replace("name: csum", "name: msum").replace(
            'title: "CSUM"', 'title: "MSUM"'
        )
        (pattern_dir / "msum.yaml").write_text(second, encoding="utf-8")

        qa_pairs, _ = load_patterns_as_qa(pattern_dir)
        questions = [q for q, _ in qa_pairs]
        assert any("CSUM" in q for q in questions)
        assert any("MSUM" in q for q in questions)

    def test_real_sample_patterns_if_present(self):
        """Run against the actual sample_data/patternexamples/ if it exists."""
        root = Path(__file__).parent.parent.parent
        pattern_dir = root / "sample_data" / "patternexamples"
        if not pattern_dir.exists():
            pytest.skip("sample_data/patternexamples not found")
        qa_pairs, _ = load_patterns_as_qa(pattern_dir)
        assert len(qa_pairs) >= 10, "Expected at least 10 QA pairs from sample patterns"
        for q, a in qa_pairs:
            assert isinstance(q, str) and q.strip()
            assert isinstance(a, str) and a.strip()


# ---------------------------------------------------------------------------
# prepare_training_data CLI smoke test
# ---------------------------------------------------------------------------

class TestPrepareCLI:
    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "data.prepare_training_data", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0
        assert "pdf" in result.stdout.lower() or "csv" in result.stdout.lower()

    def test_cli_csv_produces_jsonl(self, tmp_path, tmp_csv):
        csv_path = tmp_csv([
            {"question": "What is SQL?", "answer": "A query language for databases."},
            {"question": "What is a JOIN?", "answer": "A JOIN combines rows from two tables."},
            {"question": "What is an index?", "answer": "An index speeds up data retrieval."},
        ])
        result = subprocess.run(
            [
                sys.executable, "-m", "data.prepare_training_data",
                "--csv", str(csv_path),
                "--output-dir", str(tmp_path / "out"),
                "--format", "sharegpt",
            ],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        out_dir = tmp_path / "out"
        jsonl_files = list(out_dir.glob("*.jsonl"))
        assert len(jsonl_files) >= 1

    def test_cli_yaml_produces_jsonl(self, tmp_path, sample_yaml_pattern):
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "csum.yaml").write_text(sample_yaml_pattern, encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable, "-m", "data.prepare_training_data",
                "--yaml-dir", str(pattern_dir),
                "--output-dir", str(tmp_path / "out"),
                "--format", "sharegpt",
            ],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        out_files = list((tmp_path / "out").glob("*.jsonl"))
        assert len(out_files) >= 1
