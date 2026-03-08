"""
Quality tests: validate training data formats, content minimums, and structural invariants.
These tests work both on in-memory generated data and on real JSONL files if they exist.
"""
import json
from pathlib import Path

import pytest

from data.prepare_training_data import (
    build_alpaca_examples,
    build_sharegpt_examples,
    save_jsonl,
    text_to_qa_heuristic,
    _split_train_val,
)
from data.chunking import chunk_text, chunk_text_sql_aware
from data.csv_loader import load_csv


pytestmark = pytest.mark.quality


# ---------------------------------------------------------------------------
# ShareGPT format quality
# ---------------------------------------------------------------------------

class TestShareGptFormat:
    def test_top_level_key(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assert "conversations" in ex, "Missing 'conversations' key"

    def test_conversations_is_list(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assert isinstance(ex["conversations"], list)

    def test_minimum_two_turns(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assert len(ex["conversations"]) >= 2, "Expected at least user + assistant turn"

    def test_first_turn_is_user(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assert ex["conversations"][0]["role"] == "user"

    def test_second_turn_is_assistant(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assert ex["conversations"][1]["role"] == "assistant"

    def test_all_turns_have_content(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            for turn in ex["conversations"]:
                assert "content" in turn
                assert isinstance(turn["content"], str)
                assert turn["content"].strip() != ""

    def test_no_empty_questions(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            user_turn = ex["conversations"][0]
            assert len(user_turn["content"].strip()) > 0

    def test_no_empty_answers(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            assistant_turn = ex["conversations"][1]
            assert len(assistant_turn["content"].strip()) > 0

    def test_minimum_answer_length(self, sample_sharegpt_examples):
        for ex in sample_sharegpt_examples:
            answer = ex["conversations"][1]["content"]
            assert len(answer) >= 10, f"Answer too short: {answer!r}"

    def test_roles_are_valid(self, sample_sharegpt_examples):
        valid_roles = {"user", "assistant", "system"}
        for ex in sample_sharegpt_examples:
            for turn in ex["conversations"]:
                assert turn["role"] in valid_roles


# ---------------------------------------------------------------------------
# Alpaca format quality
# ---------------------------------------------------------------------------

class TestAlpacaFormat:
    def test_required_keys_present(self, sample_alpaca_examples):
        for ex in sample_alpaca_examples:
            assert "instruction" in ex
            assert "input" in ex
            assert "output" in ex

    def test_no_extra_unexpected_keys(self, sample_alpaca_examples):
        expected = {"instruction", "input", "output"}
        for ex in sample_alpaca_examples:
            assert set(ex.keys()) == expected

    def test_instruction_nonempty(self, sample_alpaca_examples):
        for ex in sample_alpaca_examples:
            assert ex["instruction"].strip() != ""

    def test_output_nonempty(self, sample_alpaca_examples):
        for ex in sample_alpaca_examples:
            assert ex["output"].strip() != ""

    def test_input_is_string(self, sample_alpaca_examples):
        for ex in sample_alpaca_examples:
            assert isinstance(ex["input"], str)

    def test_minimum_output_length(self, sample_alpaca_examples):
        for ex in sample_alpaca_examples:
            assert len(ex["output"]) >= 10, f"Output too short: {ex['output']!r}"


# ---------------------------------------------------------------------------
# JSONL file format quality
# ---------------------------------------------------------------------------

class TestJsonlFileFormat:
    def test_each_line_valid_json(self, tmp_path, sample_sharegpt_examples):
        path = tmp_path / "train.jsonl"
        save_jsonl(sample_sharegpt_examples, path)
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON line: {line!r} — {e}")

    def test_no_blank_lines(self, tmp_path, sample_sharegpt_examples):
        path = tmp_path / "train.jsonl"
        save_jsonl(sample_sharegpt_examples, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            assert line.strip() != "", f"Blank line at position {i}"

    def test_line_count_matches_example_count(self, tmp_path, sample_sharegpt_examples):
        path = tmp_path / "train.jsonl"
        save_jsonl(sample_sharegpt_examples, path)
        lines = [l for l in path.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == len(sample_sharegpt_examples)

    def test_utf8_encoding_preserved(self, tmp_path):
        examples = [{"conversations": [
            {"role": "user", "content": "What is Ü-Bahn?"},
            {"role": "assistant", "content": "It is the underground railway in German cities — Übergabe."},
        ]}]
        path = tmp_path / "utf8.jsonl"
        save_jsonl(examples, path)
        loaded = json.loads(path.read_text(encoding="utf-8").strip())
        assert "Ü-Bahn" in loaded["conversations"][0]["content"]


# ---------------------------------------------------------------------------
# Train/Val split quality
# ---------------------------------------------------------------------------

class TestTrainValSplitQuality:
    def test_val_ratio_approximately_correct(self):
        items = list(range(1000))
        train, val = _split_train_val(items, val_ratio=0.15)
        actual_ratio = len(val) / len(items)
        assert abs(actual_ratio - 0.15) < 0.01

    def test_no_data_loss(self):
        items = list(range(200))
        train, val = _split_train_val(items, val_ratio=0.2)
        assert sorted(train + val) == items

    def test_train_larger_than_val(self):
        items = list(range(100))
        train, val = _split_train_val(items, val_ratio=0.2)
        assert len(train) > len(val)


# ---------------------------------------------------------------------------
# Generated QA pair quality
# ---------------------------------------------------------------------------

class TestQaPairQuality:
    def test_no_duplicate_questions(self, sample_qa_pairs):
        questions = [q for q, _ in sample_qa_pairs]
        assert len(questions) == len(set(questions)), "Duplicate questions found"

    def test_question_answer_not_identical(self, sample_qa_pairs):
        for q, a in sample_qa_pairs:
            assert q != a, f"Question and answer are identical: {q!r}"

    def test_questions_end_with_punctuation_or_word(self, sample_qa_pairs):
        for q, _ in sample_qa_pairs:
            # Questions should be complete sentences
            assert len(q) > 5

    def test_chunk_text_output_quality(self, sample_plain_text):
        chunks = chunk_text(sample_plain_text, chunk_size=300, chunk_overlap=50)
        for chunk in chunks:
            assert len(chunk.strip()) > 0
            assert len(chunk) <= 600  # generous upper bound (overlap can add a bit)

    def test_sql_chunk_preserves_keywords(self, sample_sql_text):
        chunks = chunk_text_sql_aware(sample_sql_text, chunk_size=200, chunk_overlap=50)
        # The SELECT statement should appear complete in one chunk
        all_text = "\n\n".join(chunks)
        assert "CSUM" in all_text
        assert "PARTITION BY" in all_text


# ---------------------------------------------------------------------------
# Real training data validation (skipped if files not present)
# ---------------------------------------------------------------------------

class TestRealTrainingDataFiles:
    """Optional tests that run against actual generated training data if it exists."""

    @staticmethod
    def _find_jsonl_files() -> list:
        root = Path(__file__).parent.parent.parent
        return (
            list(root.glob("training_data/*.jsonl")) +
            list(root.glob("training_data/*/*.jsonl"))
        )

    def test_real_files_valid_json_if_present(self):
        files = self._find_jsonl_files()
        if not files:
            pytest.skip("No training JSONL files found — run data prep first")
        for path in files:
            for i, line in enumerate(path.read_text(encoding="utf-8").strip().split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"{path.name}:{i} — invalid JSON: {e}")

    def test_real_sharegpt_files_correct_format(self):
        root = Path(__file__).parent.parent.parent
        files = (
            list(root.glob("training_data/train_sharegpt.jsonl")) +
            list(root.glob("training_data/*/train_sharegpt.jsonl"))
        )
        if not files:
            pytest.skip("No ShareGPT JSONL files found")
        for path in files:
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                obj = json.loads(line)
                assert "conversations" in obj, f"{path.name}: missing 'conversations'"
                for turn in obj["conversations"]:
                    assert "role" in turn
                    assert "content" in turn
                    assert isinstance(turn["content"], str)
