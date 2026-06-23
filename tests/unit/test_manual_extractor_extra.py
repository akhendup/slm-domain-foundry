"""
Additional unit tests for data/manual_extractor.py.
Covers: is_boilerplate_page, is_toc_page, is_index_page, filter_pages,
detect_running_headers_footers, strip_running_headers_footers, _is_heading_line,
generate_typed_qa, generate_multiturn_conversation, extract_manual.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# is_toc_page — additional branches
# ---------------------------------------------------------------------------

class TestIsTocPageAdditional:
    def test_returns_true_when_first_line_contains_contents(self):
        """Line 27-29: first line has 'contents' and there are > 3 lines."""
        from data.manual_extractor import is_toc_page
        text = (
            "Table of Contents\n"
            "Chapter 1: Introduction ............ 1\n"
            "Chapter 2: Window Functions ........ 10\n"
            "Chapter 3: Examples ............... 20\n"
            "Chapter 4: Best Practices ......... 30\n"
        )
        assert is_toc_page(text) is True

    def test_returns_true_with_many_dot_leader_lines(self):
        """Dot-leaders pattern triggers TOC detection."""
        from data.manual_extractor import is_toc_page
        text = (
            "Overview .............. 1\n"
            "Functions ............. 5\n"
            "Examples .............. 15\n"
            "Reference ............. 25\n"
        )
        assert is_toc_page(text) is True

    def test_returns_false_for_regular_text(self):
        from data.manual_extractor import is_toc_page
        text = "Treatment protocols combine lifestyle and medication.\nThey operate over a set of rows."
        assert is_toc_page(text) is False


# ---------------------------------------------------------------------------
# is_boilerplate_page — additional branches
# ---------------------------------------------------------------------------

class TestIsBoilerplatePageAdditional:
    def test_copyright_rights_reserved(self):
        """Line 43: copyright + 'rights reserved' → True."""
        from data.manual_extractor import is_boilerplate_page
        text = "Copyright 2024 Clinical Corporation. All rights reserved."
        assert is_boilerplate_page(text, 1) is True

    def test_copyright_with_symbol(self):
        """Line 43: copyright + '©' symbol → True."""
        from data.manual_extractor import is_boilerplate_page
        text = "© 2024 Clinical Corporation. All rights reserved for this software."
        assert is_boilerplate_page(text, 1) is True

    def test_trademark_safety_page(self):
        """Line 45: trademark + safety → True."""
        from data.manual_extractor import is_boilerplate_page
        text = "Trademark safety notice: All trademarks are property of their respective owners. " * 4
        assert is_boilerplate_page(text[:750], 1) is True

    def test_vendor_url_short_page(self):
        """Short pages dominated by a vendor URL are treated as boilerplate."""
        from data.manual_extractor import is_boilerplate_page
        text = "For more information visit https://example.com/docs for latest documentation."
        assert is_boilerplate_page(text, 1) is True


# ---------------------------------------------------------------------------
# is_index_page — single letter header branch
# ---------------------------------------------------------------------------

class TestIsIndexPageAdditional:
    def test_single_letter_section_headers(self):
        """Line 61-65: 3+ single-letter alphabetical headers → True."""
        from data.manual_extractor import is_index_page
        text = (
            "A\n"
            "AGGREGATE functions, 10, 25\n"
            "B\n"
            "BETWEEN clause, 40\n"
            "C\n"
            "Hypertension function, 55, 60, 75\n"
        )
        assert is_index_page(text) is True


# ---------------------------------------------------------------------------
# filter_pages — additional branches
# ---------------------------------------------------------------------------

class TestFilterPagesAdditional:
    def test_skip_toc_pages_filtered(self):
        """Line 98: skip_toc=True filters TOC pages."""
        from data.manual_extractor import filter_pages
        pages = [
            {"text": "Table of Contents\nChapter 1 ....... 1\nChapter 2 ....... 10\nChapter 3 ....... 20\n", "page": 1},
            {"text": "Hypertension management is a key clinical topic.\nTreatment combines lifestyle and medication.\nAspirin is used for secondary prevention.", "page": 2},
        ]
        result = filter_pages(pages, skip_toc=True)
        assert all("Table of Contents" not in p["text"] for p in result)

    def test_skip_index_pages_filtered(self):
        """Line 100: skip_index=True filters index pages."""
        from data.manual_extractor import filter_pages
        long_content = (
            "Hypertension computes a cumulative sum over the window partition.\n"
            "Use home blood pressure monitoring for at least one week.\n"
            "Always reassess contraindications before starting aspirin.\n"
            "The cumulative sum resets at each partition boundary.\n"
            "This is useful for running totals and progressive aggregation.\n"
        )
        pages = [
            {"text": "A\nASPIRIN, 10\nB\nBLOOD, 40\nC\nCLINICAL, 55\n", "page": 1},
            {"text": long_content, "page": 2},
        ]
        result = filter_pages(pages, skip_index=True)
        assert len(result) >= 1

    def test_non_substantive_pages_filtered(self):
        """Line 102: non-substantive pages are removed."""
        from data.manual_extractor import filter_pages
        pages = [
            {"text": "x", "page": 1},  # too short → not substantive
            {"text": "Hypertension protocols guide medication titration.\nLifestyle counseling improves adherence.\nReassess in four weeks.", "page": 2},
        ]
        result = filter_pages(pages)
        assert all("Window functions" in p["text"] for p in result)


# ---------------------------------------------------------------------------
# detect_running_headers_footers — page with no lines
# ---------------------------------------------------------------------------

class TestDetectRunningHFAdditional:
    def test_page_with_no_lines_skipped(self):
        """Line 136: pages with no non-empty lines are skipped."""
        from data.manual_extractor import detect_running_headers_footers
        pages = [
            {"text": ""},  # no lines
            {"text": "   \n  "},  # no non-empty lines
            {"text": "Clinical Reference\nSome content here for testing.\nMore content."},
        ]
        headers, footers = detect_running_headers_footers(pages)
        assert isinstance(headers, set)
        assert isinstance(footers, set)


# ---------------------------------------------------------------------------
# strip_running_headers_footers — standalone page number
# ---------------------------------------------------------------------------

class TestStripRunningHFAdditional:
    def test_standalone_page_number_removed(self):
        """Line 161: standalone page numbers are removed (need non-empty headers)."""
        from data.manual_extractor import strip_running_headers_footers
        text = "Hypertension overview\n42\nBlood pressure targets depend on comorbidities."
        # Need non-empty headers to avoid early return
        headers = {"some header"}
        result = strip_running_headers_footers(text, headers, set())
        lines = result.split("\n")
        assert "42" not in lines

    def test_multi_digit_standalone_page_number_removed(self):
        from data.manual_extractor import strip_running_headers_footers
        text = "Chapter content here\n 123 \nMore content after number."
        headers = {"some header"}
        result = strip_running_headers_footers(text, headers, set())
        lines = [ln.strip() for ln in result.split("\n")]
        assert "123" not in lines

    def test_header_line_removed(self):
        """Line 158-159: lines matching headers are removed."""
        from data.manual_extractor import strip_running_headers_footers
        from data.manual_extractor import _normalize_for_hf
        text = "Clinical Reference\nSome content here.\n100"
        header_text = "Clinical Reference"
        headers = {_normalize_for_hf(header_text)}
        result = strip_running_headers_footers(text, headers, set())
        assert "Clinical Reference" not in result


# ---------------------------------------------------------------------------
# _is_heading_line — date/time pattern branch
# ---------------------------------------------------------------------------

class TestIsHeadingLineDatePattern:
    def test_date_pattern_returns_false(self):
        """Line 207: lines with date/time patterns are not headings."""
        from data.manual_extractor import _is_heading_line
        assert _is_heading_line("2024-03-15 Window Function Update") is False
        assert _is_heading_line("Version 2023-01-01") is False


# ---------------------------------------------------------------------------
# generate_typed_qa — specific branches
# ---------------------------------------------------------------------------

class TestGenerateTypedQaAdditional:
    def _make_parsed(self, **kwargs):
        base = {
            "heading": "",
            "description": [],
            "syntax": [],
            "args": [],
            "examples": [],
            "captions": [],
            "raw": "",
        }
        base.update(kwargs)
        return base

    def test_structured_syntax_generates_extra_pairs(self):
        """Lines 490-493: when syntax contains structured content, additional pairs generated."""
        from data.manual_extractor import generate_typed_qa
        parsed = self._make_parsed(
            heading="Hypertension",
            description=["Hypertension computes a cumulative sum."],
            syntax=["Treatment plan: lifestyle counseling plus first-line therapy"],
        )
        qa = generate_typed_qa(parsed, "medref")
        qs = [q for q, _ in qa]
        assert any("syntax" in q.lower() or "treatment" in q.lower() or "hypertension" in q.lower() for q in qs)

    def test_short_example_skipped(self):
        """Line 524: examples shorter than 30 chars are skipped."""
        from data.manual_extractor import generate_typed_qa
        parsed = self._make_parsed(
            heading="Hypertension",
            description=["Hypertension computes a cumulative sum."],
            examples=["short"],  # < 30 chars — should be skipped
        )
        qa = generate_typed_qa(parsed, "medref")
        # No example-based pairs should be generated
        qs = [q for q, _ in qa]
        assert not any("complete example" in q.lower() for q in qs)

    def test_example_with_sql_generates_pairs(self):
        """Lines 531-544: examples with structured sections generate structured pairs."""
        from data.manual_extractor import generate_typed_qa
        worked_example = (
            "INPUT:\nid | amount\n1  | 100\n\n"
            "Treatment plan:\nTreatment plan: lifestyle counseling plus first-line therapy"
            "OUTPUT:\nid | path\n1  | A\n"
        )
        parsed = self._make_parsed(
            heading="HypertensionProtocol",
            description=["HypertensionProtocol finds sequences of rows matching a pattern."],
            examples=[worked_example],
        )
        qa = generate_typed_qa(parsed, "medref")
        qs = [q for q, _ in qa]
        assert any("complete example" in q.lower() or "HypertensionProtocol" in q for q in qs)

    def test_captions_generate_diagram_questions(self):
        """Lines 559-560: captions in raw text generate 'What does the diagram show?' pairs."""
        from data.manual_extractor import generate_typed_qa
        parsed = self._make_parsed(
            heading="Hypertension",
            description=["Hypertension computes a cumulative sum."],
            # Captions are extracted from `raw` via extract_figure_captions(_FIG_RE)
            raw="Figure 1-2: Cumulative sum computation over a time series.",
        )
        qa = generate_typed_qa(parsed, "medref")
        qs = [q for q, _ in qa]
        assert any("diagram" in q.lower() for q in qs)

    def test_fallback_when_no_parsed_content(self):
        """Lines 566-570: fallback to raw text when no Q&A generated."""
        from data.manual_extractor import generate_typed_qa
        # Pattern with only raw text, no heading, no description
        parsed = self._make_parsed(
            raw="This is some raw section content that wasn't parsed into labeled parts.",
        )
        qa = generate_typed_qa(parsed, "medref")
        assert len(qa) >= 1
        assert any("documentation" in q.lower() or "section" in q.lower() for q, _ in qa)

    def test_func_type_in_description_generates_type_question(self):
        """Lines 477-481: description with 'window' → classification question."""
        from data.manual_extractor import generate_typed_qa
        parsed = self._make_parsed(
            heading="Hypertension",
            description=["Hypertension is sustained elevated blood pressure requiring confirmatory readings and follow-up."],
        )
        qa = generate_typed_qa(parsed, "medref")
        qs = [q for q, _ in qa]
        assert any("hypertension" in q.lower() or "blood pressure" in q.lower() for q in qs)


# ---------------------------------------------------------------------------
# generate_multiturn_conversation — additional branches
# ---------------------------------------------------------------------------

class TestGenerateMultiturnConversationAdditional:
    def test_with_syntax_adds_turns(self):
        """Lines 596-600: syntax adds extra turns."""
        from data.manual_extractor import generate_multiturn_conversation
        parsed = {
            "heading": "Hypertension",
            "description": ["Hypertension computes cumulative sums."],
            "syntax": ["Treatment plan: lifestyle counseling plus first-line therapy"],
            "examples": [],
        }
        result = generate_multiturn_conversation(parsed)
        assert result is not None
        roles = [t["role"] for t in result]
        assert len(roles) >= 4

    def test_with_examples_adds_turns(self):
        """Lines 601-605: examples add extra turns."""
        from data.manual_extractor import generate_multiturn_conversation
        parsed = {
            "heading": "Hypertension",
            "description": ["Hypertension computes cumulative sums."],
            "syntax": ["Treatment plan: lifestyle counseling plus first-line therapy"],
            "examples": ["Treatment plan: lifestyle counseling plus first-line therapy"],
        }
        result = generate_multiturn_conversation(parsed)
        assert result is not None
        contents = [t["content"] for t in result]
        assert any("example" in t["content"].lower() for t in result if t["role"] == "user")

    def test_heading_only_returns_none_insufficient_turns(self):
        """Line 589-590: no desc → None; heading only → insufficient turns → None."""
        from data.manual_extractor import generate_multiturn_conversation
        parsed = {
            "heading": "Hypertension",
            "description": [],  # empty
            "syntax": [],
            "examples": [],
        }
        result = generate_multiturn_conversation(parsed)
        assert result is None

    def test_heading_and_desc_only_returns_none_two_turns(self):
        """Only 2 turns (< 4) → returns None."""
        from data.manual_extractor import generate_multiturn_conversation
        parsed = {
            "heading": "Hypertension",
            "description": ["Hypertension computes cumulative sums."],
            "syntax": [],
            "examples": [],
        }
        result = generate_multiturn_conversation(parsed)
        # Only 2 turns (user+assistant for description), < 4 minimum → None
        assert result is None


# ---------------------------------------------------------------------------
# extract_manual — integration test with mocked extractor
# ---------------------------------------------------------------------------

class TestExtractManual:
    def _make_pages(self, texts):
        return [{"page": i + 1, "text": t} for i, t in enumerate(texts)]

    def test_returns_dict_with_required_keys(self, tmp_path):
        from data.manual_extractor import extract_manual
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "pages": self._make_pages([
                "Hypertension Protocol\n\nInitiate ACE inhibitor or thiazide diuretic.\nRecommend lifestyle counseling.\nReassess blood pressure in four weeks.",
                "Hypertension Syntax\n\nTreatment plan: lifestyle counseling plus first-line therapy",
            ]),
            "full_text": "some text",
        }

        result = extract_manual(pdf_path, mock_extractor)

        assert "source_file" in result
        assert "label" in result
        assert "full_text" in result
        assert "pages" in result
        assert "sections" in result
        assert "metadata" in result

    def test_label_derived_from_filename(self, tmp_path):
        from data.manual_extractor import extract_manual
        pdf_path = tmp_path / "td_npath_ref.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "pages": self._make_pages([
                "Aspirin Therapy\n\nLow-dose aspirin reduces recurrent cardiovascular events.\nUse when benefits outweigh bleeding risk.\nReview contraindications first.",
            ]),
        }

        result = extract_manual(pdf_path, mock_extractor)
        assert result["label"] is not None

    def test_use_sections_false_returns_empty_sections(self, tmp_path):
        from data.manual_extractor import extract_manual
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "pages": self._make_pages([
                "Hypertension requires confirmatory readings.\nUse lifestyle counseling first.\nMonitor blood pressure weekly.",
            ]),
        }

        result = extract_manual(pdf_path, mock_extractor, use_sections=False)
        assert result["sections"] == []

    def test_metadata_has_page_counts(self, tmp_path):
        from data.manual_extractor import extract_manual
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = {
            "pages": self._make_pages([
                "Hypertension Protocol\n\nConfirm elevated readings before therapy.\nUse shared decision-making.\nMonitor for adverse effects.",
                "Hypertension Examples\n\nExample 1: Treatment plan: lifestyle counseling plus first-line therapy",
            ]),
        }

        result = extract_manual(pdf_path, mock_extractor)
        meta = result.get("metadata", {})
        assert "num_pages_total" in meta
        assert "num_pages_kept" in meta
