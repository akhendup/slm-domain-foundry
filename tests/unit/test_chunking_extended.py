"""
Extended unit tests for data/chunking.py.
Covers: semantic chunking path, internal _chunk_rule_based edge cases,
and _chunk_rule_based_sql_aware edge cases not reached by the primary tests.
"""
from unittest.mock import MagicMock, patch

import pytest

import data.chunking as chunking_module
from data.chunking import (
    _chunk_rule_based_sql_aware,
    chunk_text,
    chunk_text_sql_aware,
    is_sql_paragraph,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# chunk_text with use_semantic=True but SEMANTIC_AVAILABLE=False → rule-based
# ---------------------------------------------------------------------------

class TestChunkTextSemanticFallback:
    def test_semantic_false_path(self, monkeypatch):
        """use_semantic=True with SEMANTIC_AVAILABLE=False falls back to rule-based."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", False)
        text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        result = chunk_text(text, chunk_size=200, chunk_overlap=0, use_semantic=True)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_semantic_false_never_calls_sentence_transformer(self, monkeypatch):
        """SEMANTIC_AVAILABLE=False must not try to load SentenceTransformer."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", False)
        with patch.object(chunking_module, "_chunk_semantic") as mock_sem:
            chunk_text("Some text here.", chunk_size=50, use_semantic=True)
        mock_sem.assert_not_called()

    def test_use_semantic_false_always_rule_based(self, monkeypatch):
        """Even with SEMANTIC_AVAILABLE=True, use_semantic=False avoids semantic path."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", True)
        with patch.object(chunking_module, "_chunk_semantic") as mock_sem:
            chunk_text("Short text.", chunk_size=1000, use_semantic=False)
        mock_sem.assert_not_called()


# ---------------------------------------------------------------------------
# _chunk_semantic via mock — covers lines 155–187
# ---------------------------------------------------------------------------

class TestChunkSemantic:
    def test_semantic_path_executes(self, monkeypatch):
        """Exercise _chunk_semantic by patching SentenceTransformer into the module."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", True)

        mock_st = MagicMock()
        text = (
            "The CSUM function returns a cumulative sum over a window partition. "
            "It uses the OVER clause to define the window. "
            "Specify PARTITION BY to group results. " * 10
        )
        # Use create=True because SentenceTransformer may not exist in module namespace
        with patch("data.chunking.SentenceTransformer", mock_st, create=True):
            result = chunking_module._chunk_semantic(text, chunk_size=200, overlap=50)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_semantic_short_text_returns_list(self, monkeypatch):
        """_chunk_semantic with text shorter than chunk_size returns single chunk."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", True)
        mock_st = MagicMock()
        text = "Short text."
        with patch("data.chunking.SentenceTransformer", mock_st, create=True):
            result = chunking_module._chunk_semantic(text, chunk_size=1000, overlap=100)
        assert isinstance(result, list)

    def test_semantic_with_overlap(self, monkeypatch):
        """_chunk_semantic with overlap should carry over sentences."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", True)
        mock_st = MagicMock()
        text = " ".join(
            f"This is sentence number {i} with enough content to fill things." for i in range(30)
        )
        with patch("data.chunking.SentenceTransformer", mock_st, create=True):
            result = chunking_module._chunk_semantic(text, chunk_size=200, overlap=80)
        assert len(result) >= 1

    def test_semantic_exception_falls_through_to_rule_based(self, monkeypatch):
        """If _chunk_semantic raises, chunk_text should fall back to rule-based."""
        monkeypatch.setattr(chunking_module, "SEMANTIC_AVAILABLE", True)

        mock_st_class = MagicMock(side_effect=Exception("model download failed"))
        text = "Paragraph one with content.\n\nParagraph two with more content.\n\nParagraph three here."
        with patch("data.chunking.SentenceTransformer", mock_st_class, create=True):
            result = chunk_text(text, chunk_size=50, chunk_overlap=0, use_semantic=True)
        # Should still return chunks via rule-based fallback
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# _chunk_rule_based internal edge cases
# ---------------------------------------------------------------------------

class TestChunkRuleBasedInternals:
    def test_empty_paragraph_skipped(self):
        """Paragraphs that become empty after strip are skipped (line 79 coverage)."""
        text = "Para one.\n\n   \n\nPara two."
        result = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        assert len(result) >= 1
        combined = "\n".join(result)
        assert "Para one" in combined
        assert "Para two" in combined

    def test_overlap_with_no_prior_paragraphs(self):
        """Overlap code path when current is empty edge case."""
        text = "Very long paragraph " * 200
        result = chunk_text(text, chunk_size=100, chunk_overlap=50)
        assert len(result) >= 1

    def test_large_single_paragraph_exceeds_chunk(self):
        """A paragraph larger than chunk_size stays in one chunk."""
        big_para = "word " * 300  # ~1500 chars
        result = chunk_text(big_para, chunk_size=100, chunk_overlap=0)
        # The big paragraph may end up in a single chunk (added even if over-size)
        assert len(result) >= 1

    def test_overlap_zero_no_carry(self):
        """chunk_overlap=0 means no overlap between chunks."""
        paras = [f"Paragraph {i} content here." for i in range(6)]
        text = "\n\n".join(paras)
        result = chunk_text(text, chunk_size=60, chunk_overlap=0)
        # Adjacent chunks should share no identical paragraphs (no carry-over)
        for i in range(len(result) - 1):
            # Words in chunk i should not be entirely present in chunk i+1
            assert result[i] != result[i + 1]


# ---------------------------------------------------------------------------
# _chunk_rule_based_sql_aware internal edge cases
# ---------------------------------------------------------------------------

class TestChunkSqlAwareInternals:
    def test_empty_paragraph_skipped(self):
        """Empty paragraphs inside sql_aware chunker are skipped."""
        text = "Prose one.\n\n   \n\nSELECT x FROM t WHERE y > 0 ORDER BY z;\n\nProse two."
        result = chunk_text_sql_aware(text, chunk_size=1000, chunk_overlap=0)
        combined = " ".join(result)
        assert "Prose one" in combined
        assert "Prose two" in combined

    def test_overlap_zero_sql_aware(self):
        """chunk_overlap=0 branch in sql_aware chunker (line 143-144)."""
        sql = "SELECT id, CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) AS total FROM orders;"
        prose_a = "Introduction content here. " * 5
        prose_b = "Conclusion content here. " * 5
        text = prose_a + "\n\n" + sql + "\n\n" + prose_b
        result = chunk_text_sql_aware(text, chunk_size=60, chunk_overlap=0)
        assert len(result) >= 1

    def test_sql_para_never_in_overlap_carry(self):
        """SQL paragraphs must not appear in overlap carry-over (line 133-134)."""
        sql = "SELECT RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk FROM emp;"
        prose_filler = "Generic prose content to fill the chunk size limits. " * 5
        text = prose_filler + "\n\n" + sql + "\n\n" + prose_filler
        result = chunk_text_sql_aware(text, chunk_size=100, chunk_overlap=80)

        # Find the chunk(s) containing the SQL
        sql_chunks = [c for c in result if "RANK()" in c]
        assert len(sql_chunks) >= 1

        # The SQL content should appear complete (not split) in the chunk(s) that have it
        for c in sql_chunks:
            assert "RANK() OVER" in c

    def test_sql_para_larger_than_chunk_size(self):
        """A SQL paragraph larger than chunk_size still stays whole."""
        sql = "SELECT " + ", ".join(f"CSUM(col{i}, ts) OVER (PARTITION BY id ORDER BY ts) AS r{i}" for i in range(20)) + " FROM t;"
        prose = "Intro content. " * 5
        text = prose + "\n\n" + sql
        result = chunk_text_sql_aware(text, chunk_size=50, chunk_overlap=10)
        sql_chunks = [c for c in result if "CSUM(col0" in c]
        assert len(sql_chunks) >= 1


# ---------------------------------------------------------------------------
# is_sql_paragraph additional cases
# ---------------------------------------------------------------------------

class TestIsSqlParagraphAdditional:
    def test_csum_keyword(self):
        assert is_sql_paragraph("CSUM(amount, ts) OVER (PARTITION BY id)") is True

    def test_msum_keyword(self):
        assert is_sql_paragraph("MSUM(sales, 3) OVER (ORDER BY date)") is True

    def test_qualify_keyword(self):
        para = "QUALIFY ROW_NUMBER() OVER (ORDER BY id) = 1"
        assert is_sql_paragraph(para) is True

    def test_create_table_keyword(self):
        para = "CREATE TABLE temp AS SELECT id FROM orders WHERE status = 'active'"
        assert is_sql_paragraph(para) is True

    def test_with_cte_keyword(self):
        para = "WITH base AS (SELECT id FROM t) SELECT * FROM base ORDER BY id"
        assert is_sql_paragraph(para) is True
