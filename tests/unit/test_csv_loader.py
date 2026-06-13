"""Unit tests for data/csv_loader.py"""
import pytest

from data.csv_loader import load_csv


pytestmark = pytest.mark.unit


class TestLoadCsv:
    # -- Auto-detection -------------------------------------------------------

    def test_auto_detect_question_answer_columns(self, tmp_csv):
        path = tmp_csv([
            {"question": "What is SQL?", "answer": "A query language."},
            {"question": "What is a table?", "answer": "A structured set of rows and columns."},
        ])
        texts, qa = load_csv(path)
        assert texts == []
        assert len(qa) == 2
        assert qa[0] == ("What is SQL?", "A query language.")

    def test_auto_detect_q_a_two_column(self, tmp_csv):
        path = tmp_csv([
            {"q": "Why use indexes?", "a": "Indexes speed up lookups."},
        ])
        texts, qa = load_csv(path)
        assert len(qa) == 1
        assert qa[0][0] == "Why use indexes?"

    def test_auto_detect_text_column(self, tmp_csv):
        path = tmp_csv([
            {"text": "First line of documentation."},
            {"text": "Second line of documentation."},
        ])
        texts, qa = load_csv(path)
        assert qa == []
        assert len(texts) == 2
        assert texts[0] == "First line of documentation."

    def test_auto_detect_single_column(self, tmp_csv):
        path = tmp_csv([
            {"content": "Row one."},
            {"content": "Row two."},
        ])
        texts, qa = load_csv(path)
        assert len(texts) == 2

    def test_auto_detect_two_columns_default_to_qa(self, tmp_csv):
        path = tmp_csv([
            {"topic": "Hypertension", "explanation": "Cumulative sum function."},
        ])
        texts, qa = load_csv(path)
        assert len(qa) == 1
        assert qa[0] == ("Hypertension", "Cumulative sum function.")

    # -- Explicit columns -----------------------------------------------------

    def test_explicit_question_answer(self, tmp_csv):
        path = tmp_csv([
            {"Q": "What is HypertensionProtocol?", "A": "A path analysis function.", "extra": "ignored"},
        ])
        texts, qa = load_csv(path, question_column="Q", answer_column="A")
        assert len(qa) == 1
        assert qa[0][0] == "What is HypertensionProtocol?"

    def test_explicit_text_column(self, tmp_csv):
        path = tmp_csv([
            {"body": "Some body text.", "meta": "ignored"},
        ])
        texts, qa = load_csv(path, text_column="body")
        assert len(texts) == 1
        assert texts[0] == "Some body text."

    # -- Edge cases -----------------------------------------------------------

    def test_empty_csv_returns_empty(self, tmp_csv):
        path = tmp_csv([])
        texts, qa = load_csv(path)
        assert texts == []
        assert qa == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")

    def test_skips_blank_qa_rows(self, tmp_csv):
        path = tmp_csv([
            {"question": "Valid question?", "answer": "Valid answer."},
            {"question": "", "answer": "Orphaned answer."},
            {"question": "Another question?", "answer": ""},
        ])
        texts, qa = load_csv(path)
        # Only the row with both q and a non-empty should be included
        assert len(qa) == 1

    def test_skips_blank_text_rows(self, tmp_csv):
        path = tmp_csv([
            {"text": "Good text."},
            {"text": ""},
            {"text": "   "},
        ])
        texts, qa = load_csv(path)
        assert len(texts) == 1

    def test_strips_whitespace(self, tmp_csv):
        path = tmp_csv([
            {"question": "  Padded question?  ", "answer": "  Padded answer.  "},
        ])
        texts, qa = load_csv(path)
        assert qa[0] == ("Padded question?", "Padded answer.")

    def test_semicolon_delimiter(self, tmp_path):
        p = tmp_path / "semi.csv"
        p.write_text("question;answer\nWhat?;Because.\n", encoding="utf-8")
        texts, qa = load_csv(p, delimiter=";")
        assert len(qa) == 1
        assert qa[0] == ("What?", "Because.")

    def test_returns_tuple_types(self, tmp_csv):
        path = tmp_csv([{"question": "Q", "answer": "A"}])
        texts, qa = load_csv(path)
        assert isinstance(texts, list)
        assert isinstance(qa, list)
        assert isinstance(qa[0], tuple)
        assert len(qa[0]) == 2
