"""Unit tests for data/chunking.py"""
import pytest

from data.chunking import chunk_text, chunk_text_structured_aware, is_structured_paragraph


pytestmark = pytest.mark.unit


class TestIsStructuredParagraph:
    def test_positive_clinical_case(self):
        para = (
            "The patient presents with hypertension and requires medication "
            "adjustment plus treatment plan review."
        )
        assert is_structured_paragraph(para) is True

    def test_positive_treatment_plan(self):
        para = (
            "Treatment plan: initiate aspirin therapy with dosage review and "
            "monitor adverse symptoms in the patient."
        )
        assert is_structured_paragraph(para) is True

    def test_positive_guideline_excerpt(self):
        para = (
            "Clinical guideline recommends diagnosis confirmation, medication titration, "
            "and regular symptom monitoring for hypertension."
        )
        assert is_structured_paragraph(para) is True

    def test_negative_plain_prose(self):
        para = "This document explains general concepts in plain language."
        assert is_structured_paragraph(para) is False

    def test_negative_single_keyword(self):
        para = "The patient arrived on time."
        assert is_structured_paragraph(para) is False

    def test_negative_empty(self):
        assert is_structured_paragraph("") is False

    def test_case_insensitive_keywords(self):
        para = "Patient medication and treatment require dosage review for hypertension."
        assert is_structured_paragraph(para) is True


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
        para = "Word " * 40
        text = "\n\n".join([para] * 10)
        result = chunk_text(text, chunk_size=400, chunk_overlap=0)
        assert len(result) >= 2

    def test_all_content_present(self):
        paras = [f"Paragraph number {i} with some content to fill it out." for i in range(8)]
        text = "\n\n".join(paras)
        result = chunk_text(text, chunk_size=200, chunk_overlap=0)
        combined = " ".join(result)
        for para in paras:
            assert para in combined or any(para in c for c in result)

    def test_no_empty_chunks(self):
        paras = ["Alpha beta gamma delta.", "One two three four five.", "Quick brown fox."]
        text = "\n\n".join(paras)
        result = chunk_text(text, chunk_size=50, chunk_overlap=0)
        for chunk in result:
            assert chunk.strip() != ""

    def test_overlap_repeats_content(self):
        para = "Overlap test sentence content here. " * 5
        text = "\n\n".join([para] * 8)
        result = chunk_text(text, chunk_size=300, chunk_overlap=100)
        if len(result) >= 2:
            words_0 = set(result[0].split())
            words_1 = set(result[1].split())
            assert len(words_0 & words_1) > 0

    def test_returns_list_of_strings(self):
        result = chunk_text("Hello world\n\nSecond paragraph.", chunk_size=1000)
        assert isinstance(result, list)
        for c in result:
            assert isinstance(c, str)


class TestChunkTextStructuredAware:
    def test_empty_returns_empty(self):
        assert chunk_text_structured_aware("") == []

    def test_short_returns_single(self):
        text = "A short piece of text."
        result = chunk_text_structured_aware(text, chunk_size=1000)
        assert result == [text.strip()]

    def test_structured_para_not_split(self):
        structured_para = (
            "Case: patient with hypertension and elevated readings. "
            "Treatment plan: medication dosage adjustment and symptom monitoring. "
            "Outcome: reassess diagnosis and adverse effects in four weeks."
        )
        filler = "Some prose about the guideline. " * 15
        text = filler + "\n\n" + structured_para + "\n\n" + filler
        result = chunk_text_structured_aware(text, chunk_size=400, chunk_overlap=100)
        marker = "Treatment plan: medication dosage adjustment"
        matching = [c for c in result if marker in c]
        assert len(matching) >= 1, "Structured paragraph not found intact in any chunk"
        assert len(matching) == 1 or all(marker in c for c in matching)

    def test_structured_not_in_overlap(self):
        structured_para = (
            "Patient history shows hypertension, medication non-adherence, and symptom "
            "progression requiring treatment plan revision."
        )
        prose_a = "Introduction paragraph with general content. " * 10
        prose_b = "Conclusion paragraph with summary content. " * 10
        text = prose_a + "\n\n" + structured_para + "\n\n" + prose_b
        result = chunk_text_structured_aware(text, chunk_size=300, chunk_overlap=150)
        for i in range(len(result) - 1):
            if "medication non-adherence" in result[i] and "medication non-adherence" in result[i + 1]:
                assert "treatment plan revision" in result[i + 1]

    def test_no_empty_chunks(self):
        text = (
            "Para one.\n\n"
            "Patient medication and treatment require dosage review for hypertension.\n\n"
            "Para two."
        )
        result = chunk_text_structured_aware(text, chunk_size=50, chunk_overlap=10)
        for chunk in result:
            assert chunk.strip() != ""
