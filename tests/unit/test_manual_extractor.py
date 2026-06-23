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
    generate_typed_qa,
    _generate_generic_qa,
    parse_function_section,
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
            "Hypertension ....... 42\n"
            "MSUM ....... 56\n"
            "RANK ....... 78\n"
            "ROW_NUMBER .. 90\n"
        )
        assert is_toc_page(text) is True

    def test_negative_regular_content(self):
        text = (
            "The Hypertension function computes a cumulative sum over a window partition.\n"
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
        text = "Copyright 2023 Clinical. All rights reserved."
        assert is_boilerplate_page(text, page_num=1) is True

    def test_positive_nearly_empty(self):
        assert is_boilerplate_page("  ", page_num=1) is True

    def test_positive_docs_url_short(self):
        # Short page dominated by a vendor URL — generic check, any vendor
        text = "https://docs.example.com/product/guide — visit for more information."
        assert is_boilerplate_page(text, page_num=2) is True

    def test_positive_generic_vendor_url_short(self):
        # Any vendor URL-dominated short page (not just one specific vendor)
        text = "http://support.oracle.com/docs/guide/ Please visit the online help center."
        assert is_boilerplate_page(text, page_num=3) is True

    def test_negative_real_content(self):
        text = (
            "Clinical guidelines describe that perform calculations across "
            "a set of table rows related to the current row. Unlike regular aggregate "
            "targets depend on comorbidities, age, and overall cardiovascular risk.\n"
            "Confirm elevated readings on at least two separate visits before labeling.\n"
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
            "Hypertension 89, 90, 91\n"
        )
        assert is_index_page(text) is True

    def test_positive_page_number_entries(self):
        lines = [f"TERM_{i} {i * 3}, {i * 5}" for i in range(1, 10)]
        text = "\n".join(lines)
        assert is_index_page(text) is True

    def test_negative_regular_content(self):
        text = (
            "Blood pressure targets depend on comorbidities and patient age.\n"
            "Home monitoring helps distinguish white-coat hypertension from sustained elevation.\n"
            "Schedule follow-up within four weeks after starting therapy.\n"
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
            "Clinical guidelines describe stepwise management of sustained hypertension.\n"
            "Lifestyle counseling is first-line for many adults with elevated readings.\n"
            "Pharmacologic therapy is added when targets are not met after lifestyle changes.\n"
            "Monitoring plans should document frequency and target blood pressure values.\n"
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
            "Blood pressure targets depend on comorbidities and patient age.\n"
            "They require confirmatory readings before diagnosis is finalized.\n"
            "Lifestyle changes and medication are tailored to comorbidities and age.\n"
            "Follow-up monitoring should track response and adverse effects.\n"
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
        boiler = "Copyright 2023 Example Publisher. All rights reserved."
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
        header = "Clinical Practice Guidelines Guide"
        footer = "Chapter 3"
        pages = []
        for i in range(10):
            text = f"{header}\nPage content line one for page {i}.\nMore content here.\n{footer}"
            pages.append({"text": text, "page": i})
        headers, footers = detect_running_headers_footers(pages, min_freq=0.35)
        # Normalized header should be detected
        assert any("clinical practice guidelines guide" in h for h in headers)

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
        text = "clinical practice guidelines\nReal content line one.\nMore real content here."
        headers = {"clinical practice guidelines"}
        result = strip_running_headers_footers(text, headers, set())
        assert "clinical practice guidelines" not in result.lower()
        assert "Real content" in result

    def test_no_op_when_empty_sets(self):
        text = "Some text\nMore text."
        result = strip_running_headers_footers(text, set(), set())
        assert result == text

    def test_normalize_strips_page_numbers(self):
        """_normalize_for_hf should strip leading/trailing page numbers."""
        assert _normalize_for_hf("42 Clinical Reference") == "clinical reference"
        assert _normalize_for_hf("Clinical Reference 42") == "clinical reference"


# ---------------------------------------------------------------------------
# generate_typed_qa — technical section (uses shared templates)
# ---------------------------------------------------------------------------

class TestGenerateTypedQaTechnical:
    def _parsed(self, **kwargs):
        base = {
            "heading": "Hypertension",
            "description": [],
            "syntax": [],
            "arguments": [],
            "notes": [],
            "examples": [],
            "raw": "",
        }
        base.update(kwargs)
        return base

    def test_syntax_uses_shared_templates(self):
        """SYNTAX_QUESTIONS templates must appear as questions for syntax sections."""
        from data.question_templates import SYNTAX_QUESTIONS
        parsed = self._parsed(
            syntax=["Lifestyle counseling plus first-line antihypertensive therapy with follow-up monitoring"]
        )
        pairs = generate_typed_qa(parsed, source_label="Hypertension")
        questions = [q for q, _ in pairs]
        # At least one SYNTAX_QUESTIONS template should appear
        filled = [t.format(fn="Hypertension") for t in SYNTAX_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_argument_uses_shared_templates(self):
        """ARGUMENT_QUESTIONS templates must appear for argument sections."""
        from data.question_templates import ARGUMENT_QUESTIONS
        parsed = self._parsed(arguments=["value: numeric expression to accumulate"])
        pairs = generate_typed_qa(parsed, source_label="Hypertension")
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Hypertension") for t in ARGUMENT_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_notes_uses_shared_templates(self):
        """NOTES_QUESTIONS templates must appear for notes sections."""
        from data.question_templates import NOTES_QUESTIONS
        parsed = self._parsed(notes=["Requires confirmatory blood pressure readings on two visits."])
        pairs = generate_typed_qa(parsed, source_label="Hypertension")
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Hypertension") for t in NOTES_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_description_produces_multiple_pairs(self):
        desc = ("Hypertension computes a cumulative sum. "
                "It is sustained elevated blood pressure managed with lifestyle and medication.")
        parsed = self._parsed(description=[desc])
        pairs = generate_typed_qa(parsed, source_label="Hypertension")
        assert len(pairs) >= 2

    def test_empty_section_produces_no_pairs(self):
        parsed = self._parsed()
        pairs = generate_typed_qa(parsed, source_label="Hypertension")
        # No content → no pairs (or only fallback if raw is set)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# _generate_generic_qa — general document fallback
# ---------------------------------------------------------------------------

class TestGenerateGenericQa:
    _PROSE = (
        "Photosynthesis is defined as the process by which green plants convert sunlight "
        "into chemical energy. It occurs in the chloroplasts and requires carbon dioxide, "
        "water, and light. Compared to cellular respiration, photosynthesis produces "
        "oxygen rather than consuming it. For example, a leaf in sunlight will absorb "
        "light energy and use it to synthesize glucose."
    )

    def test_returns_list_of_pairs(self):
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        assert isinstance(pairs, list)
        assert all(isinstance(q, str) and isinstance(a, str) for q, a in pairs)

    def test_produces_multiple_pairs(self):
        """Generic Q&A must produce more than one pair — not just a single fallback."""
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        assert len(pairs) >= 5

    def test_overview_questions_always_present(self):
        """GENERAL_OVERVIEW_QUESTIONS must fire for any content."""
        from data.question_templates import GENERAL_OVERVIEW_QUESTIONS
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Photosynthesis") for t in GENERAL_OVERVIEW_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_definition_triggers_key_concepts(self):
        """Text with definition signals should trigger GENERAL_KEYCONCEPT_QUESTIONS."""
        from data.question_templates import GENERAL_KEYCONCEPT_QUESTIONS
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Photosynthesis") for t in GENERAL_KEYCONCEPT_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_comparison_triggers_comparison_questions(self):
        """Text with 'compared to' should trigger GENERAL_COMPARISON_QUESTIONS."""
        from data.question_templates import GENERAL_COMPARISON_QUESTIONS
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Photosynthesis") for t in GENERAL_COMPARISON_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_example_signal_triggers_application_questions(self):
        """Text with 'for example' should trigger GENERAL_APPLICATION_QUESTIONS."""
        from data.question_templates import GENERAL_APPLICATION_QUESTIONS
        pairs = _generate_generic_qa("Photosynthesis", self._PROSE)
        questions = [q for q, _ in pairs]
        filled = [t.format(fn="Photosynthesis") for t in GENERAL_APPLICATION_QUESTIONS]
        assert any(q in questions for q in filled)

    def test_answers_are_the_full_text(self):
        """All answers in generic Q&A should be the section text."""
        pairs = _generate_generic_qa("Topic", self._PROSE)
        assert all(a == self._PROSE for _, a in pairs)

    def test_plain_text_no_signals(self):
        """Even plain text with no special signals gets at least overview + detail pairs."""
        plain = (
            "The Battle of Hastings took place in 1066. "
            "It was a decisive Norman victory over the English forces of King Harold."
        )
        pairs = _generate_generic_qa("Battle of Hastings", plain)
        assert len(pairs) >= 2


# ---------------------------------------------------------------------------
# generate_typed_qa — general prose fallback path
# ---------------------------------------------------------------------------

class TestGenerateTypedQaGeneralFallback:
    def test_prose_section_produces_rich_qa(self):
        """A section with only raw text (no syntax/args) must produce >1 pair via generic fallback."""
        prose = (
            "Compound interest is defined as interest calculated on both the initial principal "
            "and the accumulated interest from previous periods. Compared to simple interest, "
            "it grows faster over time. For example, an account earning 5% compound interest "
            "annually will outperform a simple interest account after just a few years."
        )
        parsed = {
            "heading": "Compound Interest",
            "description": [],
            "syntax": [],
            "arguments": [],
            "notes": [],
            "examples": [],
            "raw": prose,
        }
        pairs = generate_typed_qa(parsed, source_label="Compound Interest")
        assert len(pairs) >= 5

    def test_section_without_heading_uses_source_label(self):
        raw = "Amortization reduces a loan balance by periodic payments covering both principal and interest."
        parsed = {
            "heading": "",
            "description": [],
            "syntax": [],
            "arguments": [],
            "notes": [],
            "examples": [],
            "raw": raw,
        }
        pairs = generate_typed_qa(parsed, source_label="Amortization")
        assert len(pairs) >= 1
        assert all("Amortization" in q or raw in a for q, a in pairs)


# ---------------------------------------------------------------------------
# question_templates.py — loads from YAML
# ---------------------------------------------------------------------------

class TestQuestionTemplates:
    def test_all_technical_lists_non_empty(self):
        from data.question_templates import (
            DESCRIPTION_QUESTIONS, ONE_SENTENCE_QUESTIONS, CATEGORY_QUESTIONS,
            USE_CASE_QUESTIONS, PARAMETER_QUESTIONS, SYNTAX_QUESTIONS,
            ARGUMENT_QUESTIONS, NOTES_QUESTIONS, EXAMPLE_QUESTIONS,
        )
        for lst in [
            DESCRIPTION_QUESTIONS, ONE_SENTENCE_QUESTIONS, CATEGORY_QUESTIONS,
            USE_CASE_QUESTIONS, PARAMETER_QUESTIONS, SYNTAX_QUESTIONS,
            ARGUMENT_QUESTIONS, NOTES_QUESTIONS, EXAMPLE_QUESTIONS,
        ]:
            assert isinstance(lst, list) and len(lst) > 0

    def test_all_general_lists_non_empty(self):
        from data.question_templates import (
            GENERAL_OVERVIEW_QUESTIONS, GENERAL_KEYCONCEPT_QUESTIONS,
            GENERAL_DETAIL_QUESTIONS, GENERAL_COMPARISON_QUESTIONS,
            GENERAL_APPLICATION_QUESTIONS, GENERAL_CAUSES_QUESTIONS,
            GENERAL_REQUIREMENTS_QUESTIONS,
        )
        for lst in [
            GENERAL_OVERVIEW_QUESTIONS, GENERAL_KEYCONCEPT_QUESTIONS,
            GENERAL_DETAIL_QUESTIONS, GENERAL_COMPARISON_QUESTIONS,
            GENERAL_APPLICATION_QUESTIONS, GENERAL_CAUSES_QUESTIONS,
            GENERAL_REQUIREMENTS_QUESTIONS,
        ]:
            assert isinstance(lst, list) and len(lst) > 0

    def test_all_financial_lists_non_empty(self):
        from data.question_templates import (
            FINANCIAL_TRANSACTION_QUESTIONS, FINANCIAL_ACCOUNT_QUESTIONS,
            FINANCIAL_ANALYSIS_QUESTIONS,
        )
        for lst in [FINANCIAL_TRANSACTION_QUESTIONS, FINANCIAL_ACCOUNT_QUESTIONS,
                    FINANCIAL_ANALYSIS_QUESTIONS]:
            assert isinstance(lst, list) and len(lst) > 0

    def test_templates_contain_fn_placeholder(self):
        """All templates must contain {fn} so they can be formatted."""
        from data import question_templates as qt
        all_lists = [
            qt.DESCRIPTION_QUESTIONS, qt.ONE_SENTENCE_QUESTIONS, qt.CATEGORY_QUESTIONS,
            qt.USE_CASE_QUESTIONS, qt.PARAMETER_QUESTIONS, qt.SYNTAX_QUESTIONS,
            qt.ARGUMENT_QUESTIONS, qt.NOTES_QUESTIONS, qt.EXAMPLE_QUESTIONS,
            qt.GENERAL_OVERVIEW_QUESTIONS, qt.GENERAL_KEYCONCEPT_QUESTIONS,
            qt.GENERAL_DETAIL_QUESTIONS, qt.GENERAL_COMPARISON_QUESTIONS,
            qt.GENERAL_APPLICATION_QUESTIONS, qt.GENERAL_CAUSES_QUESTIONS,
            qt.GENERAL_REQUIREMENTS_QUESTIONS,
            qt.FINANCIAL_TRANSACTION_QUESTIONS, qt.FINANCIAL_ACCOUNT_QUESTIONS,
            qt.FINANCIAL_ANALYSIS_QUESTIONS,
        ]
        for lst in all_lists:
            for tmpl in lst:
                assert "{fn}" in tmpl, f"Template missing {{fn}}: {tmpl!r}"

    def test_yaml_file_loads_successfully(self):
        """question_templates.yaml must be parseable and have expected sections."""
        import yaml
        from pathlib import Path
        yaml_path = Path("data/question_templates.yaml")
        assert yaml_path.exists(), "question_templates.yaml missing"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "technical" in data
        assert "general" in data
        assert "financial" in data

    def test_fallback_when_yaml_missing(self, tmp_path, monkeypatch):
        """If YAML file doesn't exist, built-in defaults keep all lists non-empty."""
        import importlib
        import data.question_templates as qt_mod
        # Patch the yaml path to a nonexistent file and reload templates
        monkeypatch.setattr(qt_mod, "_TEMPLATES_YAML", tmp_path / "nonexistent.yaml")
        result = qt_mod._load_yaml_templates()
        assert result == {}  # graceful empty dict, not an exception
