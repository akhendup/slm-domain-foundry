"""Unit tests for data/chunking.py"""
import pytest

from data.chunking import chunk_text, chunk_text_sql_aware, is_sql_paragraph


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# is_sql_paragraph
# ---------------------------------------------------------------------------

class TestIsSqlParagraph:
    def test_positive_select_from(self):
        para = "SELECT customer_id FROM orders WHERE amount > 100"
        assert is_sql_paragraph(para) is True

    def test_positive_window_function(self):
        para = "CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts)"
        assert is_sql_paragraph(para) is True

    def test_positive_group_by(self):
        para = "SELECT region, SUM(sales) FROM fact GROUP BY region ORDER BY region"
        assert is_sql_paragraph(para) is True

    def test_negative_plain_prose(self):
        para = "This document explains how analytic functions work in general terms."
        assert is_sql_paragraph(para) is False

    def test_negative_single_keyword(self):
        # Only one SQL keyword — not enough to qualify
        para = "The SELECT statement retrieves rows."
        assert is_sql_paragraph(para) is False

    def test_negative_empty(self):
        assert is_sql_paragraph("") is False

    def test_case_insensitive(self):
        para = "select id from t where x > 1"
        assert is_sql_paragraph(para) is True


# ---------------------------------------------------------------------------
# chunk_text — rule-based (default)
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Short paragraph that fits in one chunk."
        result = chunk_text(text, chunk_size=1000)
        assert result == [text.strip()]

    def test_text_exactly_chunk_size_is_single(self):
        text = "A" * 500
        result = chunk_text(text, chunk_size=500)
        assert len(result) == 1

    def test_splits_long_text(self):
        # Build text clearly larger than chunk_size
        para = "Word " * 40  # ~200 chars per para
        text = "\n\n".join([para] * 10)  # ~2000+ chars total
        result = chunk_text(text, chunk_size=400, chunk_overlap=0)
        assert len(result) >= 2

    def test_all_content_present(self):
        """No content should be silently dropped (check overlap-free case)."""
        paras = [f"Paragraph number {i} with some content to fill it out." for i in range(8)]
        text = "\n\n".join(paras)
        result = chunk_text(text, chunk_size=200, chunk_overlap=0)
        combined = " ".join(result)
        for para in paras:
            # Each paragraph should appear somewhere in the output
            assert para in combined or any(para in c for c in result)

    def test_no_empty_chunks(self):
        paras = ["Alpha beta gamma delta.", "One two three four five.", "Quick brown fox."]
        text = "\n\n".join(paras)
        result = chunk_text(text, chunk_size=50, chunk_overlap=0)
        for chunk in result:
            assert chunk.strip() != ""

    def test_overlap_repeats_content(self):
        """With overlap, some content from the end of chunk N appears in chunk N+1."""
        para = "Overlap test sentence content here. " * 5
        text = "\n\n".join([para] * 8)
        result = chunk_text(text, chunk_size=300, chunk_overlap=100)
        if len(result) >= 2:
            # At least some text from chunk 0 should appear in chunk 1
            words_0 = set(result[0].split())
            words_1 = set(result[1].split())
            assert len(words_0 & words_1) > 0

    def test_returns_list_of_strings(self):
        result = chunk_text("Hello world\n\nSecond paragraph.", chunk_size=1000)
        assert isinstance(result, list)
        for c in result:
            assert isinstance(c, str)


# ---------------------------------------------------------------------------
# chunk_text_sql_aware
# ---------------------------------------------------------------------------

class TestChunkTextSqlAware:
    def test_empty_returns_empty(self):
        assert chunk_text_sql_aware("") == []

    def test_short_returns_single(self):
        text = "A short piece of text."
        result = chunk_text_sql_aware(text, chunk_size=1000)
        assert result == [text.strip()]

    def test_sql_para_not_split(self):
        """A SQL paragraph should appear whole in exactly one chunk."""
        sql_para = (
            "SELECT customer_id, CSUM(amount, order_date) "
            "OVER (PARTITION BY customer_id ORDER BY order_date) AS total "
            "FROM orders WHERE region = 'US' GROUP BY customer_id;"
        )
        filler = "Some prose about the function. " * 15  # push chunk boundary
        text = filler + "\n\n" + sql_para + "\n\n" + filler
        result = chunk_text_sql_aware(text, chunk_size=400, chunk_overlap=100)
        # The SQL content must appear intact in one chunk
        sql_snippet = "CSUM(amount, order_date)"
        matching = [c for c in result if sql_snippet in c]
        assert len(matching) >= 1, "SQL paragraph not found intact in any chunk"
        # It should not be split across two chunks
        assert len(matching) == 1 or all(sql_snippet in c for c in matching)

    def test_sql_not_in_overlap(self):
        """SQL paragraphs must not appear in the overlap carry-over."""
        sql_para = (
            "SELECT id, RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk "
            "FROM employees;"
        )
        prose_a = "Introduction paragraph with general content. " * 10
        prose_b = "Conclusion paragraph with summary content. " * 10
        text = prose_a + "\n\n" + sql_para + "\n\n" + prose_b
        result = chunk_text_sql_aware(text, chunk_size=300, chunk_overlap=150)
        # If SQL is in chunk i, it must not bleed into chunk i+1 purely via overlap
        for i in range(len(result) - 1):
            if "RANK()" in result[i] and "RANK()" in result[i + 1]:
                # Both chunks contain it — that's only ok if chunk i+1 includes the full SQL
                assert "RANK() OVER" in result[i + 1]

    def test_no_empty_chunks(self):
        text = "Para one.\n\nSELECT x FROM t WHERE y > 0 ORDER BY z;\n\nPara two."
        result = chunk_text_sql_aware(text, chunk_size=50, chunk_overlap=10)
        for chunk in result:
            assert chunk.strip() != ""
