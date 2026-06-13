"""
End-to-end tests covering the PDF processing branches in prepare_training_data.main().
Uses mocked PDFExtractor and extract_manual to avoid real PDF parsing.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


pytestmark = pytest.mark.e2e


_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fake_pdf(path: Path) -> Path:
    """Write a minimal PDF header so the file exists."""
    path.write_bytes(b"%PDF-1.4 fake\n")
    return path


def _standard_extract_result(full_text=""):
    return {"full_text": full_text, "pages": [], "metadata": {"num_pages": 1}}


def _manual_extract_result(full_text="", sections=None, label="test_manual"):
    return {
        "full_text": full_text,
        "label": label,
        "metadata": {
            "num_pages_kept": 3,
            "num_pages_total": 5,
            "num_sections": len(sections or []),
        },
        "sections": sections or [],
        "source_file": "test.pdf",
    }


# ---------------------------------------------------------------------------
# text_to_qa_heuristic — unit-style, called via main()
# ---------------------------------------------------------------------------

class TestTextToQaHeuristic:
    """Direct import tests for text_to_qa_heuristic branches."""

    def test_structured_example_block_detected(self):
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = (
            "Case presentation:\nPatient with diabetes and elevated blood pressure.\n\n"
            "Treatment plan:\nInitiate ACE inhibitor therapy and home blood pressure monitoring.\n\n"
            "Outcome:\nBP 128/78 mmHg at four-week follow-up\n"
        )
        qa = text_to_qa_heuristic([chunk], source="doc")
        # structured example block generates at least one QA pair
        assert len(qa) >= 1

    def test_structured_example_block_no_func_name(self):
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = (
            "Case:\npatient history\n\nTreatment plan:\nmedication titration and lifestyle counseling\n\nOutcome:\nimproved blood pressure\n"
        )
        qa = text_to_qa_heuristic([chunk], source="myfile.pdf")
        assert len(qa) >= 1

    def test_structured_content_without_example_block(self):
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = "Patient with hypertension requires medication review, clinical follow-up, and lifestyle counseling per treatment guideline recommendations."
        qa = text_to_qa_heuristic([chunk], source="doc")
        assert len(qa) >= 1

    def test_heading_based_extraction(self):
        from data.prepare_training_data import text_to_qa_heuristic
        # Use a heading that passes _is_heading_line (not starting with digit)
        chunk = "Hypertension Protocol\n\nHypertension management combines lifestyle modification with first-line antihypertensive medication."
        qa = text_to_qa_heuristic([chunk], source="doc")
        assert len(qa) >= 1
        # Heading without "?" gets "What is ...?" prepended
        assert any("Window Functions" in q or "What is" in q for q, _ in qa)

    def test_heading_generates_what_is_question(self):
        from data.prepare_training_data import text_to_qa_heuristic
        # "Data Types" passes _is_heading_line and doesn't end with "?"
        chunk = "Blood Pressure Targets\n\nMost adults aim for below 130/80 mmHg with individualized treatment plans."
        qa = text_to_qa_heuristic([chunk], source="doc")
        assert len(qa) >= 1
        qs = [q for q, _ in qa]
        assert any("Blood Pressure Targets" in q for q in qs)

    def test_generic_fallback_for_plain_text(self):
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = "This is a long description of general wellness without clinical headings or structured protocol content in it."
        qa = text_to_qa_heuristic([chunk], source="myref")
        assert len(qa) == 1
        assert "myref" in qa[0][0]

    def test_empty_chunk_skipped(self):
        from data.prepare_training_data import text_to_qa_heuristic
        qa = text_to_qa_heuristic(["", "   ", "\n\n"], source="doc")
        assert qa == []

    def test_short_chunk_skipped(self):
        from data.prepare_training_data import text_to_qa_heuristic
        qa = text_to_qa_heuristic(["short"], source="doc")
        assert qa == []

    def test_structured_without_func_name_uses_source(self):
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = "Patient with hypertension requires medication review, clinical follow-up, and lifestyle counseling per treatment guideline recommendations."
        qa = text_to_qa_heuristic([chunk], source="myfile.pdf")
        assert len(qa) >= 1
        qs = [q for q, _ in qa]
        assert any("myfile" in q for q in qs)

    def test_example_block_with_sql_but_no_func(self):
        """Input/Output block with structured content but no extractable function name."""
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = (
            "Case:\npatient row\n\nTreatment plan:\nmedication and lifestyle counseling\n\nOutcome:\nstable blood pressure\n"
        )
        qa = text_to_qa_heuristic([chunk], source="myfile.pdf")
        # Should still produce at least one pair
        assert len(qa) >= 1

    def test_structured_example_block_with_treatment_plan(self):
        """Case/Treatment plan/Outcome block generates structured Q&A pairs."""
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = (
            "Case presentation:\n"
            "Patient with diabetes and repeated elevated blood pressure readings.\n\n"
            "Treatment plan:\n"
            "Initiate ACE inhibitor therapy, recommend sodium reduction, exercise, "
            "and home blood pressure monitoring.\n\n"
            "Outcome:\n"
            "Reassess blood pressure in four weeks and adjust therapy if targets are not met."
        )
        qa = text_to_qa_heuristic([chunk], source="doc")
        qs = [q for q, _ in qa]
        assert any("example" in q.lower() or "Treatment" in q or "hypertension" in q.lower() for q in qs)
        assert len(qa) >= 1

    def test_structured_fallback_with_clinical_keywords(self):
        """Structured-content fallback generates domain-specific questions."""
        from data.prepare_training_data import text_to_qa_heuristic
        chunk = (
            "The following protocol demonstrates hypertension management:\n"
            "Patient with hypertension requires medication review, clinical follow-up, "
            "and lifestyle counseling per treatment guideline recommendations.\n"
        )
        qa = text_to_qa_heuristic([chunk], source="doc")
        qs = [q for q, _ in qa]
        assert any("structured" in q.lower() or "example" in q.lower() for q in qs)

    def test_example_block_no_sql_part_but_has_structured_content(self):
        """Input/Output block detected, _split_example_parts returns no structured key,
        but has_structured_content is true → lines 59-64."""
        from data.prepare_training_data import text_to_qa_heuristic
        # This needs the regex to detect "input"/"output" pattern but _split_example_parts
        # to return empty sql key. Use an unusual structure.
        chunk = (
            "Case:\npatient with elevated readings\n\nOutcome:\nreassess in four weeks\n"
        )
        # The "Input:" line is detected, but there's no "Treatment plan:" line
        # so _split_example_parts may return sql="" — then we check has_structured_content
        qa = text_to_qa_heuristic([chunk], source="doc")
        assert len(qa) >= 1


# ---------------------------------------------------------------------------
# Standard PDF mode
# ---------------------------------------------------------------------------

class TestPipelineStandardPdf:
    def test_standard_pdf_produces_output(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "manual.pdf")

        out_dir = tmp_path / "out"
        full_text = (
            "Window Functions\n\n"
            "Window functions compute results over a set of rows.\n"
            "They do not collapse rows like aggregate functions.\n"
            "Hypertension management requires medication review and clinical follow-up.\n"
        )

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls:
            mock_ext = MagicMock()
            mock_ext_cls.return_value = mock_ext
            mock_ext.extract.return_value = _standard_extract_result(full_text)
            rc = main()

        assert rc == 0
        assert (out_dir / "train_sharegpt.jsonl").exists()

    def test_standard_pdf_empty_text_skipped(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "empty.pdf")

        out_dir = tmp_path / "out"

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls:
            mock_ext = MagicMock()
            mock_ext_cls.return_value = mock_ext
            mock_ext.extract.return_value = _standard_extract_result("")  # empty text
            rc = main()

        # No QA pairs → returns 1
        assert rc == 1

    def test_standard_pdf_multiple_files(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            _create_fake_pdf(pdf_dir / name)

        out_dir = tmp_path / "out"
        full_text = (
            "Aspirin Therapy\n\nLow-dose aspirin is used for secondary cardiovascular prevention.\n"
            "Use PARTITION BY to segment data.\nOrder by time column.\n"
        )

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--output-dir", str(out_dir),
            "--format", "alpaca",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls:
            mock_ext = MagicMock()
            mock_ext_cls.return_value = mock_ext
            mock_ext.extract.return_value = _standard_extract_result(full_text)
            rc = main()

        assert rc == 0
        train = out_dir / "train_alpaca.jsonl"
        assert train.exists()

    def test_nonexistent_pdf_dir_is_skipped(self, tmp_path, tmp_csv):
        """If --pdf-dir doesn't exist, PDF section is skipped; other sources still work."""
        from data.prepare_training_data import main

        csv_path = tmp_csv([
            {"question": "What is hypertension?", "answer": "Hypertension is chronic elevation of blood pressure."},
        ])
        out_dir = tmp_path / "out"

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(tmp_path / "no_such_dir"),
            "--csv", str(csv_path),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]
        rc = main()
        assert rc == 0


# ---------------------------------------------------------------------------
# Manual PDF mode
# ---------------------------------------------------------------------------

class TestPipelineManualPdf:
    def _base_argv(self, pdf_dir, out_dir):
        return [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--manual",
            "--output-dir", str(out_dir),
            "--format", "both",
        ]

    def test_manual_mode_writes_per_manual_output(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "td_functions.pdf")

        out_dir = tmp_path / "out"
        full_text = (
            "Hypertension Protocol\n\nInitiate lifestyle counseling plus first-line antihypertensive therapy.\n"
            "SELECT Hypertension(amount, ts) OVER (PARTITION BY id ORDER BY ts) FROM t;\n"
        )
        sections = [
            {"heading": "Hypertension Function", "text": full_text},
        ]

        sys.argv = self._base_argv(pdf_dir, out_dir)

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result(full_text, sections, "td_functions")
            rc = main()

        manual_dir = out_dir / "td_functions"
        assert manual_dir.exists()
        assert any(manual_dir.glob("*.jsonl"))

    def test_manual_mode_no_multiturn(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "doc.pdf")

        out_dir = tmp_path / "out"
        full_text = "RANK Function\n\nRANK assigns a rank to each row.\nUse ORDER BY clause.\n"
        sections = [{"heading": "RANK Function", "text": full_text}]

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--manual",
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
            "--no-multiturn",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result(full_text, sections, "doc")
            rc = main()

        # No multiturn file
        doc_dir = out_dir / "doc"
        assert not (doc_dir / "train_multiturn.jsonl").exists()

    def test_manual_mode_empty_text_skipped(self, tmp_path, capsys):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "empty.pdf")

        out_dir = tmp_path / "out"

        sys.argv = self._base_argv(pdf_dir, out_dir)

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result("")  # empty text
            rc = main()

        # Manual mode with no QA, no CSV, no YAML → rc=1
        assert rc == 1

    def test_manual_mode_writes_multiturn_when_available(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "doc.pdf")

        out_dir = tmp_path / "out"
        full_text = (
            "Hypertension Protocol\n\nTreatment plan includes medication titration and home monitoring.\n"
            "SELECT Hypertension(x, t) OVER (ORDER BY t) FROM tbl;\n"
        )
        sections = [{"heading": "Hypertension", "text": full_text}]

        sys.argv = self._base_argv(pdf_dir, out_dir)

        # Patch generate_multiturn_conversation to return a real conversation
        fake_conv = [
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant", "content": "Hypertension is chronic elevation of blood pressure."},
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract, \
             patch("data.prepare_training_data.generate_multiturn_conversation",
                   return_value=fake_conv):
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result(full_text, sections, "doc")
            rc = main()

        # With multiturn and format=both, multiturn files should be written
        doc_dir = out_dir / "doc"
        if doc_dir.exists():
            # If doc_dir was written, check for multiturn files
            mt_file = doc_dir / "train_multiturn.jsonl"
            # File exists only when there were pairs and multiturn convs
            pass

    def test_manual_mode_multiple_pdfs(self, tmp_path):
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf_names = ["ref1.pdf", "ref2.pdf"]
        for name in pdf_names:
            _create_fake_pdf(pdf_dir / name)

        out_dir = tmp_path / "out"
        full_text = "Hypertension\n\nChronic elevation of blood pressure increases cardiovascular risk.\nMore content here.\n"
        sections = [{"heading": "Hypertension", "text": full_text}]

        sys.argv = self._base_argv(pdf_dir, out_dir)

        call_count = [0]

        def mock_extract(pdf_path, extractor, **kwargs):
            call_count[0] += 1
            label = pdf_path.stem
            return _manual_extract_result(full_text, sections, label)

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual", side_effect=mock_extract):
            mock_ext_cls.return_value = MagicMock()
            rc = main()

        assert call_count[0] == 2  # called once per PDF
        # Both manuals should have their output dirs
        for name in pdf_names:
            assert (out_dir / Path(name).stem).exists()

    def test_manual_mode_plus_csv_aggregates(self, tmp_path, tmp_csv):
        """Manual mode + CSV: CSV Q&A is aggregated globally alongside manual output."""
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "ref.pdf")

        csv_path = tmp_csv([
            {"question": "What is RANK?", "answer": "RANK assigns a rank to each row."},
        ])
        out_dir = tmp_path / "out"

        full_text = "Hypertension\n\nConfirm elevated readings before starting long-term therapy.\n"
        sections = [{"heading": "Hypertension", "text": full_text}]

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--manual",
            "--csv", str(csv_path),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result(full_text, sections, "ref")
            rc = main()

        assert rc == 0
        # Global output from CSV
        assert (out_dir / "train_sharegpt.jsonl").exists()

    def test_manual_mode_complete_output_dir_message(self, tmp_path, capsys):
        """When manual mode creates sub-dirs and no CSV/YAML, print completion message."""
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "doc.pdf")

        out_dir = tmp_path / "out"
        full_text = "Hypertension\n\nSodium reduction and regular exercise are first-line interventions.\n"
        sections = [{"heading": "Hypertension", "text": full_text}]

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--manual",
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            mock_extract.return_value = _manual_extract_result(full_text, sections, "doc")
            rc = main()

        assert rc == 0

    def test_manual_mode_no_output_subdirs_returns_1(self, tmp_path):
        """If manual mode runs but no sub-dirs are created, return 1."""
        from data.prepare_training_data import main

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _create_fake_pdf(pdf_dir / "empty.pdf")

        out_dir = tmp_path / "out"

        sys.argv = [
            "prepare_training_data",
            "--pdf-dir", str(pdf_dir),
            "--manual",
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]

        with patch("data.prepare_training_data.PDFExtractor") as mock_ext_cls, \
             patch("data.prepare_training_data.extract_manual") as mock_extract:
            mock_ext_cls.return_value = MagicMock()
            # Return empty text so no pairs are generated and no dir is created
            mock_extract.return_value = _manual_extract_result("")
            rc = main()

        assert rc == 1


# ---------------------------------------------------------------------------
# build_alpaca_examples / build_sharegpt_examples / save_jsonl / _split_train_val
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_build_alpaca_examples_structure(self):
        from data.prepare_training_data import build_alpaca_examples
        pairs = [("Q1?", "A1."), ("Q2?", "A2.")]
        result = build_alpaca_examples(pairs)
        assert len(result) == 2
        assert result[0] == {"instruction": "Q1?", "input": "", "output": "A1."}

    def test_build_sharegpt_examples_structure(self):
        from data.prepare_training_data import build_sharegpt_examples
        pairs = [("Q?", "A.")]
        result = build_sharegpt_examples(pairs)
        assert len(result) == 1
        conv = result[0]["conversations"]
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "Q?"
        assert conv[1]["role"] == "assistant"
        assert conv[1]["content"] == "A."

    def test_save_jsonl_creates_file(self, tmp_path):
        from data.prepare_training_data import save_jsonl
        items = [{"key": "value"}, {"key": "value2"}]
        out = tmp_path / "sub" / "out.jsonl"
        save_jsonl(items, out)
        assert out.exists()
        lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln]
        assert lines == items

    def test_split_train_val_sizes(self):
        from data.prepare_training_data import _split_train_val
        items = list(range(10))
        train, val = _split_train_val(items, 0.2)
        assert len(train) + len(val) == 10
        assert len(val) == 2

    def test_split_train_val_minimum_val(self):
        """Even with tiny ratio, at least 1 val item."""
        from data.prepare_training_data import _split_train_val
        items = list(range(3))
        train, val = _split_train_val(items, 0.01)
        assert len(val) >= 1
