"""Unit tests for data/manual_extractor.py"""
import pytest

from data.manual_extractor import (
    detect_running_headers_footers,
    filter_pages,
    is_boilerplate_page,
    is_index_page,
    is_substantive_page,
    is_toc_page,
    strip_running_headers_footers,
    _normalize_for_hf,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# is_toc_page
# ---------------------------------------------------------------------------

class TestIsTocPage:
    def test_positive_contents_header(self):
        text = (
            "Contents\n"
            "Chapter 1. Introduction .............. 1\n"
            "Chapter 2. Overview ................. 5\n"
            "Chapter 3. Details ................. 12\n"
        )
        assert is_toc_page(text) is True

    def test_positive_dot_leaders(self):
        text = (
            "CSUM ....... 42\n"
            "MSUM ....... 56\n"
            "RANK ....... 78\n"
            "ROW_NUMBER .. 90\n"
        )
        assert is_toc_page(text) is True

    def test_negative_regular_content(self):
        text = (
            "The CSUM function computes a cumulative sum over a window partition.\n"
            "It takes two arguments: a value expression and an ordering column.\n"
            "Use PARTITION BY to define the groups.\n"
        )
        assert is_toc_page(text) is False

    def test_negative_empty(self):
        assert is_toc_page("") is False

    def test_negative_too_short(self):
        assert is_toc_page("Contents\n1") is False


# ---------------------------------------------------------------------------
# is_boilerplate_page
# ---------------------------------------------------------------------------

class TestIsBoilerplatePage:
    def test_positive_copyright(self):
        text = "Copyright 2023 Teradata. All rights reserved."
        assert is_boilerplate_page(text, page_num=1) is True

    def test_positive_nearly_empty(self):
        assert is_boilerplate_page("  ", page_num=1) is True

    def test_positive_docs_url_short(self):
        text = "docs.teradata.com — visit for more information."
        assert is_boilerplate_page(text, page_num=2) is True

    def test_negative_real_content(self):
        text = (
            "Window functions are SQL functions that perform calculations across "
            "a set of table rows related to the current row. Unlike regular aggregate "
            "functions, window functions do not collapse rows into a single output row.\n"
            "The OVER clause defines the window of rows used for each calculation.\n"
        )
        assert is_boilerplate_page(text, page_num=5) is False


# ---------------------------------------------------------------------------
# is_index_page
# ---------------------------------------------------------------------------

class TestIsIndexPage:
    def test_positive_single_letter_headers(self):
        text = (
            "A\n"
            "ACCOUNT 23\n"
            "ALIAS 45\n"
            "B\n"
            "BETWEEN 67\n"
            "C\n"
            "CSUM 89, 90, 91\n"
        )
        assert is_index_page(text) is True

    def test_positive_page_number_entries(self):
        lines = [f"TERM_{i} {i * 3}, {i * 5}" for i in range(1, 10)]
        text = "\n".join(lines)
        assert is_index_page(text) is True

    def test_negative_regular_content(self):
        text = (
            "The RANK function assigns a rank to each row within a partition.\n"
            "Rows with equal values receive the same rank, and gaps follow.\n"
            "Use ORDER BY inside OVER to specify the ranking order.\n"
        )
        assert is_index_page(text) is False

    def test_negative_empty(self):
        assert is_index_page("") is False


# ---------------------------------------------------------------------------
# is_substantive_page
# ---------------------------------------------------------------------------

class TestIsSubstantivePage:
    def test_positive(self):
        # Needs >= 2 lines with >20 chars AND sum of those line lengths > 200
        text = (
            "Window functions compute aggregate values over a set of rows defined by the OVER clause.\n"
            "They differ from regular aggregates because they do not collapse rows into a single result.\n"
            "Each row in the result retains its individual identity while also seeing window values.\n"
            "The PARTITION BY clause divides rows into logical groups for the window calculation.\n"
        )
        assert is_substantive_page(text) is True

    def test_negative_empty(self):
        assert is_substantive_page("") is False

    def test_negative_too_short(self):
        assert is_substantive_page("Short.") is False

    def test_negative_only_short_lines(self):
        text = "Hi\nOk\nYes\nNo\nMaybe\n" * 5
        assert is_substantive_page(text) is False


# ---------------------------------------------------------------------------
# filter_pages
# ---------------------------------------------------------------------------

class TestFilterPages:
    def _make_page(self, text, page_num=5):
        return {"text": text, "page": page_num}

    def _substantive(self):
        """Return text that passes is_substantive_page (>= 2 lines >20 chars, sum > 200)."""
        return (
            "Window functions compute aggregate values over a defined set of rows in SQL.\n"
            "They use the OVER clause to specify the partition and ordering of the window.\n"
            "Unlike regular aggregates, they do not collapse the result to a single row.\n"
            "The PARTITION BY clause divides rows into groups for the window function.\n"
        )

    def test_removes_toc(self):
        toc = (
            "Contents\n"
            "Chapter 1 ........... 1\n"
            "Chapter 2 ........... 5\n"
            "Chapter 3 ........... 9\n"
        )
        real = self._substantive()
        pages = [self._make_page(toc, 1), self._make_page(real, 5)]
        result = filter_pages(pages, skip_toc=True, skip_boilerplate=True, skip_index=True)
        assert len(result) == 1
        assert result[0]["text"] == real

    def test_removes_boilerplate(self):
        boiler = "Copyright 2023 Acme Corp. All rights reserved."
        real = self._substantive()
        pages = [self._make_page(boiler, 1), self._make_page(real, 3)]
        result = filter_pages(pages)
        texts = [p["text"] for p in result]
        assert real in texts
        assert boiler not in texts

    def test_keeps_substantive_pages(self):
        real = self._substantive()
        pages = [self._make_page(real, 10)]
        result = filter_pages(pages)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# detect_running_headers_footers
# ---------------------------------------------------------------------------

class TestDetectRunningHeadersFooters:
    def test_detects_repeated_header(self):
        header = "Teradata SQL Reference Guide"
        footer = "Chapter 3"
        pages = []
        for i in range(10):
            text = f"{header}\nPage content line one for page {i}.\nMore content here.\n{footer}"
            pages.append({"text": text, "page": i})
        headers, footers = detect_running_headers_footers(pages, min_freq=0.35)
        # Normalized header should be detected
        assert any("teradata sql reference guide" in h for h in headers)

    def test_no_detection_few_pages(self):
        pages = [{"text": "Content on page.", "page": i} for i in range(2)]
        headers, footers = detect_running_headers_footers(pages)
        assert headers == set()
        assert footers == set()

    def test_returns_sets(self):
        pages = [{"text": f"Header\nContent {i}.\nFooter", "page": i} for i in range(5)]
        headers, footers = detect_running_headers_footers(pages)
        assert isinstance(headers, set)
        assert isinstance(footers, set)


# ---------------------------------------------------------------------------
# strip_running_headers_footers
# ---------------------------------------------------------------------------

class TestStripRunningHeadersFooters:
    def test_strips_header_line(self):
        text = "teradata sql reference\nReal content line one.\nMore real content here."
        headers = {"teradata sql reference"}
        result = strip_running_headers_footers(text, headers, set())
        assert "teradata sql reference" not in result.lower()
        assert "Real content" in result

    def test_no_op_when_empty_sets(self):
        text = "Some text\nMore text."
        result = strip_running_headers_footers(text, set(), set())
        assert result == text

    def test_normalize_strips_page_numbers(self):
        """_normalize_for_hf should strip leading/trailing page numbers."""
        assert _normalize_for_hf("42 Teradata Reference") == "teradata reference"
        assert _normalize_for_hf("Teradata Reference 42") == "teradata reference"
