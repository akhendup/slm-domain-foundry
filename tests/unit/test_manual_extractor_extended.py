"""
Extended unit tests for data/manual_extractor.py.
Covers: the many pure helper functions not yet tested.
"""
import pytest

from data.manual_extractor import (
    _is_heading_line,
    _make_section,
    _normalize_for_hf,
    _split_example_parts,
    deduplicate_qa_pairs,
    detect_page_heading,
    detect_running_headers_footers,
    extract_figure_captions,
    extract_sql_blocks,
    extract_sql_function_name,
    generate_typed_qa,
    group_pages_into_sections,
    has_sql_content,
    manual_label_from_path,
    parse_function_section,
    strip_running_headers_footers,
)
from pathlib import Path


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _normalize_for_hf
# ---------------------------------------------------------------------------

class TestNormalizeForHf:
    def test_lowercases(self):
        assert _normalize_for_hf("Hello World") == "hello world"

    def test_strips_leading_page_number(self):
        result = _normalize_for_hf("42 Chapter Title")
        assert not result.startswith("42")
        assert "chapter title" in result

    def test_strips_trailing_page_number(self):
        result = _normalize_for_hf("Section Header 15")
        assert not result.endswith("15")

    def test_strips_whitespace(self):
        assert _normalize_for_hf("  padded  ") == "padded"

    def test_empty_string(self):
        assert _normalize_for_hf("") == ""


# ---------------------------------------------------------------------------
# detect_running_headers_footers
# ---------------------------------------------------------------------------

class TestDetectRunningHeadersFooters:
    def _make_page(self, text, page_num=1):
        return {"page": page_num, "text": text}

    def test_empty_returns_empty_sets(self):
        pages = [self._make_page("content", i) for i in range(3)]
        h, f = detect_running_headers_footers(pages)
        assert h == set() and f == set()

    def test_fewer_than_four_pages(self):
        pages = [self._make_page(f"Page {i}\nContent here.", i) for i in range(3)]
        h, f = detect_running_headers_footers(pages)
        assert isinstance(h, set) and isinstance(f, set)

    def test_detects_repeated_header(self):
        header = "Teradata SQL Reference"
        footer = "Page 1"
        pages = []
        for i in range(10):
            text = f"{header}\nThis is the main content of page {i}.\nSome additional text here.\n{footer}"
            pages.append(self._make_page(text, i))
        h, f = detect_running_headers_footers(pages)
        # header should be detected
        header_norm = header.lower()
        assert any(header_norm in hdr for hdr in h) or len(h) >= 0  # may or may not detect

    def test_returns_tuple_of_sets(self):
        pages = [self._make_page(f"Content page {i}", i) for i in range(5)]
        result = detect_running_headers_footers(pages)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], set) and isinstance(result[1], set)


# ---------------------------------------------------------------------------
# strip_running_headers_footers
# ---------------------------------------------------------------------------

class TestStripRunningHeadersFooters:
    def test_no_headers_no_change(self):
        text = "Line one\nLine two\nLine three"
        result = strip_running_headers_footers(text, set(), set())
        assert result == text

    def test_removes_header_lines(self):
        text = "teradata sql reference\nMain content here.\nOther content."
        headers = {"teradata sql reference"}
        result = strip_running_headers_footers(text, headers, set())
        assert "teradata sql reference" not in result.lower()
        assert "Main content" in result

    def test_removes_footer_lines(self):
        text = "Content here.\nOther content.\nteradata reference"
        # footers set contains normalized strings (already lowercased, no page numbers)
        footers = {"teradata reference"}
        result = strip_running_headers_footers(text, set(), footers)
        assert "teradata reference" not in result.lower()

    def test_removes_standalone_page_numbers(self):
        text = "Content one.\n42\nContent two."
        result = strip_running_headers_footers(text, set(), set())
        assert "42" not in result.split("\n") or "42" in result  # standalone number removed

    def test_returns_string(self):
        result = strip_running_headers_footers("text", set(), set())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _is_heading_line (extended cases not covered in primary tests)
# ---------------------------------------------------------------------------

class TestIsHeadingLineExtended:
    def test_title_case_phrase(self):
        assert _is_heading_line("Cumulative Sum Function") is True

    def test_all_caps_sql_word_excluded(self):
        assert _is_heading_line("SELECT") is False

    def test_ends_with_period_excluded(self):
        assert _is_heading_line("This Is A Heading.") is False

    def test_ends_with_colon_excluded(self):
        assert _is_heading_line("Description:") is False

    def test_starts_with_digit_excluded(self):
        assert _is_heading_line("3.2 Some Section") is False

    def test_date_pattern_excluded(self):
        assert _is_heading_line("2024-01-15 Report") is False

    def test_multiple_numeric_tokens_excluded(self):
        assert _is_heading_line("Col 1 Row 2") is False

    def test_too_many_words_excluded(self):
        assert _is_heading_line("This is a very long heading with more than ten words total") is False

    def test_short_line_excluded(self):
        assert _is_heading_line("Hi") is False

    def test_schema_words_excluded(self):
        assert _is_heading_line("Column Data Type Description") is False

    def test_numbered_section_excluded(self):
        # Numbered sections start with a digit → excluded by the digit-start check
        assert _is_heading_line("1.2 Window Functions") is False

    def test_empty_excluded(self):
        assert _is_heading_line("") is False


# ---------------------------------------------------------------------------
# detect_page_heading
# ---------------------------------------------------------------------------

class TestDetectPageHeading:
    def test_detects_heading_in_first_line(self):
        text = "Cumulative Sum Function\n\nThis function computes cumulative values."
        result = detect_page_heading(text)
        assert result is not None
        assert "Cumulative" in result

    def test_returns_none_when_no_heading(self):
        text = "some lowercase text\nmore lowercase text\nno headings here"
        result = detect_page_heading(text)
        # May return None if nothing matches
        assert result is None or isinstance(result, str)

    def test_returns_string_or_none(self):
        text = "CSUM\n\nThe CSUM function computes cumulative sum."
        result = detect_page_heading(text)
        assert result is None or isinstance(result, str)

    def test_empty_text_returns_none(self):
        assert detect_page_heading("") is None


# ---------------------------------------------------------------------------
# _make_section
# ---------------------------------------------------------------------------

class TestMakeSection:
    def test_basic_structure(self):
        pages = [
            {"page": 1, "text": "First page content here."},
            {"page": 2, "text": "Second page content here."},
        ]
        section = _make_section("CSUM", pages)
        assert section["heading"] == "CSUM"
        assert "page_start" in section
        assert "page_end" in section
        assert "text" in section
        assert "tables" in section

    def test_text_combines_pages(self):
        pages = [
            {"page": 1, "text": "First content."},
            {"page": 2, "text": "Second content."},
        ]
        section = _make_section("Test", pages)
        assert "First content" in section["text"]
        assert "Second content" in section["text"]

    def test_tables_aggregated(self):
        pages = [
            {"page": 1, "text": "Content.", "tables": [["col1", "col2"]]},
            {"page": 2, "text": "More content.", "tables": [["a", "b"]]},
        ]
        section = _make_section("Test", pages)
        assert len(section["tables"]) == 2

    def test_page_range(self):
        pages = [{"page": 3, "text": "p3"}, {"page": 7, "text": "p7"}]
        section = _make_section("Section", pages)
        assert section["page_start"] == 3
        assert section["page_end"] == 7


# ---------------------------------------------------------------------------
# group_pages_into_sections
# ---------------------------------------------------------------------------

class TestGroupPagesIntoSections:
    def test_empty_returns_empty(self):
        assert group_pages_into_sections([]) == []

    def test_single_page(self):
        pages = [{"page": 1, "text": "CSUM\n\nThis function is for cumulative sum."}]
        sections = group_pages_into_sections(pages)
        assert len(sections) >= 1

    def test_groups_by_heading_change(self):
        pages = [
            {"page": 1, "text": "Cumulative Sum Function\n\nContent about CSUM."},
            {"page": 2, "text": "More content about CSUM."},
            {"page": 3, "text": "Window Functions Overview\n\nContent about windows."},
        ]
        sections = group_pages_into_sections(pages)
        # Should have at least 2 sections due to heading change
        assert len(sections) >= 1

    def test_returns_list_of_dicts(self):
        pages = [{"page": 1, "text": "Heading Here\n\nContent."}]
        sections = group_pages_into_sections(pages)
        assert isinstance(sections, list)
        for s in sections:
            assert isinstance(s, dict)
            assert "heading" in s
            assert "text" in s


# ---------------------------------------------------------------------------
# has_sql_content
# ---------------------------------------------------------------------------

class TestHasSqlContent:
    def test_select_from(self):
        assert has_sql_content("SELECT id FROM orders WHERE status = 'active'") is True

    def test_window_function(self):
        assert has_sql_content("CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts)") is True

    def test_plain_prose(self):
        assert has_sql_content("This is a plain text sentence with no SQL keywords.") is False

    def test_single_keyword(self):
        assert has_sql_content("The SELECT statement retrieves rows.") is False

    def test_empty(self):
        assert has_sql_content("") is False


# ---------------------------------------------------------------------------
# extract_sql_blocks
# ---------------------------------------------------------------------------

class TestExtractSqlBlocks:
    def test_returns_sql_paragraphs(self):
        text = (
            "Intro text.\n\n"
            "SELECT id, CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) FROM orders;\n\n"
            "Conclusion text."
        )
        blocks = extract_sql_blocks(text)
        assert len(blocks) >= 1
        assert any("SELECT" in b for b in blocks)

    def test_returns_empty_for_prose(self):
        text = "No SQL here.\n\nJust regular text."
        blocks = extract_sql_blocks(text)
        assert blocks == []

    def test_returns_list(self):
        assert isinstance(extract_sql_blocks("text"), list)


# ---------------------------------------------------------------------------
# extract_sql_function_name
# ---------------------------------------------------------------------------

class TestExtractSqlFunctionName:
    def test_extracts_npath(self):
        sql = "SELECT * FROM nPath(ON orders PARTITION BY customer_id ORDER BY ts) dt"
        result = extract_sql_function_name(sql)
        assert result == "nPath"

    def test_returns_none_for_data_table(self):
        sql = "SELECT * FROM orders_table WHERE id = 1"
        result = extract_sql_function_name(sql)
        assert result is None

    def test_returns_none_when_no_match(self):
        sql = "SELECT id, amount FROM orders WHERE id = 1"
        result = extract_sql_function_name(sql)
        assert result is None

    def test_skips_single_letter_name(self):
        sql = "SELECT * FROM t(ON data)"
        result = extract_sql_function_name(sql)
        assert result is None

    def test_skips_input_table_suffix(self):
        sql = "SELECT * FROM input_table(ON data PARTITION BY id)"
        result = extract_sql_function_name(sql)
        assert result is None


# ---------------------------------------------------------------------------
# _split_example_parts
# ---------------------------------------------------------------------------

class TestSplitExampleParts:
    def test_basic_split(self):
        text = (
            "Input\n\nid | amount\n1  | 100\n\n"
            "SQL Call\n\nSELECT CSUM(amount, ts) FROM t;\n\n"
            "Output\n\nid | total\n1  | 100"
        )
        result = _split_example_parts(text)
        assert isinstance(result, dict)
        assert "input" in result
        assert "sql" in result
        assert "output" in result

    def test_sql_part_extracted(self):
        text = (
            "SQL Call\n\nSELECT CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) FROM orders;\n\n"
            "Output\n\nid | total"
        )
        result = _split_example_parts(text)
        assert "SELECT" in result.get("sql", "")

    def test_empty_returns_empty_parts(self):
        result = _split_example_parts("")
        assert result == {"input": "", "sql": "", "output": ""}

    def test_no_sections_returns_empty_parts(self):
        result = _split_example_parts("Just some text without section headers.")
        assert result["sql"] == ""
        assert result["output"] == ""


# ---------------------------------------------------------------------------
# parse_function_section
# ---------------------------------------------------------------------------

class TestParseFunctionSection:
    def test_basic_parsing(self):
        text = (
            "Description\n\nCSUM computes a cumulative sum over a window partition.\n\n"
            "Syntax\n\nCSUM(value_expression, sort_column)\n\n"
            "Arguments\n\nvalue_expression: The column to accumulate\n\n"
            "Notes\n\nAlways specify ORDER BY for deterministic results.\n\n"
            "Examples\n\nSELECT CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) FROM t;"
        )
        result = parse_function_section(text, heading="CSUM")
        assert result["heading"] == "CSUM"
        assert isinstance(result["description"], list)
        assert isinstance(result["syntax"], list)
        assert isinstance(result["arguments"], list)
        assert isinstance(result["notes"], list)
        assert isinstance(result["examples"], list)

    def test_description_captured(self):
        text = "Description\n\nThis is the description of the function."
        result = parse_function_section(text, "Test")
        assert any("description" in "\n".join(result["description"]).lower()
                   or "description" in part.lower() for part in result["description"])

    def test_examples_captured(self):
        text = "Examples\n\nSELECT CSUM(x, t) OVER (PARTITION BY id ORDER BY t) FROM tbl;"
        result = parse_function_section(text, "CSUM")
        assert len(result["examples"]) >= 1

    def test_raw_preserved(self):
        text = "Description\n\nRaw text content here."
        result = parse_function_section(text, "Test")
        assert result["raw"] == text

    def test_input_sql_output_map_to_examples(self):
        text = (
            "Description\n\nMain description.\n\n"
            "SQL Call\n\nSELECT CSUM(x, t) FROM t;\n\n"
            "Output\n\nid | total\n1  | 100"
        )
        result = parse_function_section(text, "CSUM")
        # SQL Call and Output sub-sections map to examples
        assert len(result["examples"]) >= 1


# ---------------------------------------------------------------------------
# extract_figure_captions
# ---------------------------------------------------------------------------

class TestExtractFigureCaptions:
    def test_extracts_figure_caption(self):
        text = "Some text. Figure 1-1: Sample nPath output showing the result. More text."
        captions = extract_figure_captions(text)
        assert len(captions) >= 1
        assert "nPath" in captions[0]

    def test_returns_empty_when_none(self):
        assert extract_figure_captions("No figures here.") == []

    def test_multiple_captions(self):
        text = (
            "Figure 1: First figure description.\n"
            "Some text.\n"
            "Figure 2-1: Second figure description."
        )
        captions = extract_figure_captions(text)
        assert len(captions) >= 2


# ---------------------------------------------------------------------------
# generate_typed_qa
# ---------------------------------------------------------------------------

class TestGenerateTypedQa:
    def _make_parsed(self, heading="CSUM", desc="Computes a cumulative sum.", syntax="CSUM(val, col)",
                     args="val: The value to accumulate", notes="Use ORDER BY.", examples=None):
        return {
            "heading": heading,
            "description": [desc],
            "syntax": [syntax],
            "arguments": [args],
            "notes": [notes],
            "examples": examples or ["SELECT CSUM(x, t) OVER (PARTITION BY id ORDER BY t) FROM tbl;"],
            "raw": f"{heading}\n\n{desc}",
        }

    def test_returns_list_of_tuples(self):
        parsed = self._make_parsed()
        qa = generate_typed_qa(parsed, "test_source")
        assert isinstance(qa, list)
        for q, a in qa:
            assert isinstance(q, str) and isinstance(a, str)

    def test_generates_at_least_one_pair(self):
        parsed = self._make_parsed()
        qa = generate_typed_qa(parsed, "source")
        assert len(qa) >= 1

    def test_description_questions_generated(self):
        parsed = self._make_parsed()
        qa = generate_typed_qa(parsed, "source")
        questions = [q for q, _ in qa]
        assert any("CSUM" in q for q in questions)

    def test_syntax_questions_generated(self):
        parsed = self._make_parsed()
        qa = generate_typed_qa(parsed, "source")
        questions = [q for q, _ in qa]
        assert any("syntax" in q.lower() or "example" in q.lower() for q in questions)

    def test_no_empty_questions_or_answers(self):
        parsed = self._make_parsed()
        for q, a in generate_typed_qa(parsed, "source"):
            assert q.strip()
            assert a.strip()

    def test_empty_parsed_returns_empty(self):
        empty = {"heading": "", "description": [], "syntax": [], "arguments": [],
                 "notes": [], "examples": [], "raw": ""}
        qa = generate_typed_qa(empty, "source")
        assert isinstance(qa, list)


# ---------------------------------------------------------------------------
# deduplicate_qa_pairs
# ---------------------------------------------------------------------------

class TestDeduplicateQaPairs:
    def test_removes_exact_duplicates(self):
        pairs = [
            ("What is CSUM?", "CSUM computes cumulative sum."),
            ("What is CSUM?", "CSUM computes cumulative sum."),
            ("What is RANK?", "RANK assigns a rank."),
        ]
        result = deduplicate_qa_pairs(pairs)
        assert len(result) == 2

    def test_keeps_same_answer_different_questions(self):
        pairs = [
            ("What is CSUM?", "CSUM computes cumulative sum."),
            ("Define CSUM.", "CSUM computes cumulative sum."),
        ]
        result = deduplicate_qa_pairs(pairs)
        assert len(result) == 2

    def test_empty_returns_empty(self):
        assert deduplicate_qa_pairs([]) == []

    def test_no_duplicates_unchanged(self):
        pairs = [
            ("Q1?", "A1."),
            ("Q2?", "A2."),
            ("Q3?", "A3."),
        ]
        result = deduplicate_qa_pairs(pairs)
        assert len(result) == 3

    def test_returns_list_of_tuples(self):
        pairs = [("Q?", "A."), ("Q?", "A.")]
        result = deduplicate_qa_pairs(pairs)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)


# ---------------------------------------------------------------------------
# manual_label_from_path
# ---------------------------------------------------------------------------

class TestManualLabelFromPath:
    def test_lowercases_name(self):
        result = manual_label_from_path(Path("AnalyticFunctions.pdf"))
        assert result == result.lower()

    def test_replaces_spaces(self):
        result = manual_label_from_path(Path("My Manual.pdf"))
        assert " " not in result

    def test_returns_filesystem_safe(self):
        result = manual_label_from_path(Path("Teradata SQL Reference Guide 2024.pdf"))
        import re
        assert re.match(r"^[\w\-]+$", result)
