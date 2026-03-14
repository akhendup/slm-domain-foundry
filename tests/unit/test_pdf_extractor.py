"""Unit tests for data/pdf_extractor.py"""
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

import data.pdf_extractor as pe
from data.pdf_extractor import PDFExtractor


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PDFExtractor.__init__
# ---------------------------------------------------------------------------

class TestPDFExtractorInit:
    def test_creates_instance(self):
        assert PDFExtractor(use_ocr=False) is not None

    def test_use_ocr_false_when_ocr_unavailable(self, monkeypatch):
        monkeypatch.setattr(pe, "OCR_AVAILABLE", False)
        extractor = PDFExtractor(use_ocr=True)
        assert extractor.use_ocr is False

    def test_use_ocr_respects_ocr_available(self, monkeypatch):
        monkeypatch.setattr(pe, "OCR_AVAILABLE", True)
        extractor = PDFExtractor(use_ocr=True)
        assert extractor.use_ocr is True

    def test_ocr_language_stored(self):
        assert PDFExtractor(use_ocr=False, ocr_language="fra").ocr_language == "fra"

    def test_method_pdfplumber_when_available(self, monkeypatch):
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        assert PDFExtractor(use_ocr=False)._method == "pdfplumber"

    def test_method_pypdf2_when_pdfplumber_unavailable(self, monkeypatch):
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", False)
        monkeypatch.setattr(pe, "PYPDF2_AVAILABLE", True)
        assert PDFExtractor(use_ocr=False)._method == "pypdf2"

    def test_raises_when_no_pdf_library(self, monkeypatch):
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", False)
        monkeypatch.setattr(pe, "PYPDF2_AVAILABLE", False)
        with pytest.raises(ImportError):
            PDFExtractor(use_ocr=False)


# ---------------------------------------------------------------------------
# PDFExtractor._is_valid_text
# ---------------------------------------------------------------------------

class TestIsValidText:
    def setup_method(self):
        self.extractor = PDFExtractor(use_ocr=False)

    def test_valid_text(self):
        text = (
            "This is a valid text with plenty of alphabetic characters.\n"
            "Second line here with more words.\n"
            "Third line for good measure."
        )
        assert self.extractor._is_valid_text(text) is True

    def test_none_returns_false(self):
        assert self.extractor._is_valid_text(None) is False

    def test_empty_returns_false(self):
        assert self.extractor._is_valid_text("") is False

    def test_too_short_returns_false(self):
        assert self.extractor._is_valid_text("short") is False

    def test_too_few_lines_returns_false(self):
        # Only one non-empty line
        assert self.extractor._is_valid_text("One single line of text here.") is False

    def test_mostly_numeric_returns_false(self):
        text = "123 456 789\n111 222 333\n444 555 666\n777 888 999"
        assert self.extractor._is_valid_text(text) is False

    def test_fewer_than_ten_words_returns_false(self):
        text = "abc\ndef\nghi"
        assert self.extractor._is_valid_text(text) is False


# ---------------------------------------------------------------------------
# PDFExtractor._extract_pdfplumber
# ---------------------------------------------------------------------------

class TestExtractPdfplumber:
    def _make_extractor(self, monkeypatch):
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        return PDFExtractor(use_ocr=False)

    def test_basic_extraction(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Some page text here with words."
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor._extract_pdfplumber(pdf_path)

        assert "pages" in result
        assert "full_text" in result
        assert "Some page text" in result["full_text"]
        assert result["pages"][0]["page"] == 1

    def test_empty_pages(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_page.extract_tables.return_value = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor._extract_pdfplumber(pdf_path)

        assert result["full_text"] == ""

    def test_tables_extracted(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page with table"
        mock_page.extract_tables.return_value = [[["col1", "col2"], ["v1", "v2"]]]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        pdf_path = tmp_path / "table.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor._extract_pdfplumber(pdf_path)

        assert result["pages"][0]["tables"] == [[["col1", "col2"], ["v1", "v2"]]]

    def test_table_extraction_exception_handled(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page text"
        mock_page.extract_tables.side_effect = Exception("table parse error")

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor._extract_pdfplumber(pdf_path)

        assert result["pages"][0]["tables"] == []


# ---------------------------------------------------------------------------
# PDFExtractor._extract_pypdf2
# ---------------------------------------------------------------------------

class TestExtractPyPDF2:
    def _make_extractor(self, monkeypatch):
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", False)
        monkeypatch.setattr(pe, "PYPDF2_AVAILABLE", True)
        return PDFExtractor(use_ocr=False)

    def test_basic_extraction(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "PyPDF2 page text content here"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {}

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 test")

        with patch("PyPDF2.PdfReader", return_value=mock_reader):
            with patch("builtins.open", mock_open(read_data=b"%PDF-1.4")):
                result = extractor._extract_pypdf2(pdf_path)

        assert "PyPDF2 page text content here" in result["full_text"]
        assert result["metadata"]["num_pages"] == 1

    def test_empty_text_page(self, tmp_path, monkeypatch):
        extractor = self._make_extractor(monkeypatch)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {}

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("PyPDF2.PdfReader", return_value=mock_reader):
            with patch("builtins.open", mock_open(read_data=b"%PDF-1.4")):
                result = extractor._extract_pypdf2(pdf_path)

        assert result["full_text"] == ""


# ---------------------------------------------------------------------------
# PDFExtractor.extract
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_result_dict(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "Valid extracted text with enough content to pass validation.\n"
            "Second line here.\nThird line for validation threshold."
        )
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(pdf_path)

        assert isinstance(result, dict)
        assert "full_text" in result
        assert "pages" in result

    def test_returns_failed_when_exception_and_no_ocr(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"not a real pdf")

        with patch("pdfplumber.open", side_effect=Exception("parse error")):
            result = extractor.extract(pdf_path)

        assert isinstance(result, dict)
        assert "full_text" in result

    def test_force_ocr_skips_standard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pe, "OCR_AVAILABLE", False)
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        # When force_ocr=True but OCR is not available, should return failed result
        result = extractor.extract(pdf_path, force_ocr=True)
        assert isinstance(result, dict)
        assert result.get("extraction_method") == "failed"

    def test_extraction_method_set(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "Sufficient text for validation here.\n"
            "Second line of content.\nThird line present."
        )
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(pdf_path)

        assert "extraction_method" in result


# ---------------------------------------------------------------------------
# PDFExtractor.extract_directory
# ---------------------------------------------------------------------------

class TestExtractDirectory:
    def test_empty_dir_returns_empty(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        assert extractor.extract_directory(tmp_path) == []

    def test_skips_files_with_no_text(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.object(extractor, "extract", return_value={"full_text": ""}):
            result = extractor.extract_directory(tmp_path)

        assert result == []

    def test_includes_files_with_text(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.object(extractor, "extract", return_value={"full_text": "Some content here."}):
            result = extractor.extract_directory(tmp_path)

        assert len(result) == 1
        assert result[0]["source_file"] == "test.pdf"

    def test_processes_multiple_pdfs(self, tmp_path):
        extractor = PDFExtractor(use_ocr=False)
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            (tmp_path / name).write_bytes(b"%PDF-1.4")

        with patch.object(extractor, "extract", return_value={"full_text": "Content here."}):
            result = extractor.extract_directory(tmp_path)

        assert len(result) == 3
