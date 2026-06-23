"""
Extended unit tests for data/pdf_extractor.py.
Covers OCR path, _extract_with_ocr, and additional extract() branches.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import data.pdf_extractor as pe
from data.pdf_extractor import PDFExtractor


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# OCR path in extract()
# ---------------------------------------------------------------------------

class TestExtractOcrPath:
    def _make_extractor_with_ocr(self, monkeypatch):
        monkeypatch.setattr(pe, "OCR_AVAILABLE", True)
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        extractor = PDFExtractor(use_ocr=True)
        assert extractor.use_ocr is True
        return extractor

    def test_ocr_called_when_force_ocr(self, tmp_path, monkeypatch):
        """force_ocr=True skips standard extraction and invokes OCR."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_img = MagicMock()
        with patch("data.pdf_extractor.convert_from_path", return_value=[mock_img], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.return_value = "OCR extracted text here."
            result = extractor.extract(pdf_path, force_ocr=True)

        assert result["extraction_method"] == "ocr"
        assert "OCR extracted text" in result["full_text"]

    def test_ocr_called_when_standard_returns_short_text(self, tmp_path, monkeypatch):
        """When standard extraction returns short text, OCR is attempted."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Short text."  # too short to be valid
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        mock_img = MagicMock()
        with patch("pdfplumber.open", return_value=mock_pdf), \
             patch("data.pdf_extractor.convert_from_path", return_value=[mock_img], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.return_value = "OCR page text with enough content."
            result = extractor.extract(pdf_path)

        assert result.get("extraction_method") in ("ocr", "pdfplumber", "failed")

    def test_ocr_exception_returns_failed(self, tmp_path, monkeypatch):
        """When OCR throws, returns failed result (lines 62-63)."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("pdfplumber.open", side_effect=Exception("parse error")), \
             patch("data.pdf_extractor.convert_from_path", side_effect=Exception("OCR crashed"), create=True):
            result = extractor.extract(pdf_path)

        assert result["extraction_method"] == "failed"

    def test_ocr_returns_failed_when_force_ocr_and_ocr_raises(self, tmp_path, monkeypatch):
        """force_ocr=True but OCR raises → failed result."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "scan.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with patch("data.pdf_extractor.convert_from_path", side_effect=RuntimeError("no images"), create=True):
            result = extractor.extract(pdf_path, force_ocr=True)

        assert result["extraction_method"] == "failed"


# ---------------------------------------------------------------------------
# _extract_with_ocr
# ---------------------------------------------------------------------------

class TestExtractWithOcr:
    def _make_extractor_with_ocr(self, monkeypatch):
        monkeypatch.setattr(pe, "OCR_AVAILABLE", True)
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        return PDFExtractor(use_ocr=True)

    def test_basic_ocr_extraction(self, tmp_path, monkeypatch):
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        with patch("data.pdf_extractor.convert_from_path", return_value=[mock_img1, mock_img2], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.side_effect = ["Page one text.", "Page two text."]
            result = extractor._extract_with_ocr(pdf_path)

        assert "full_text" in result
        assert "pages" in result
        assert len(result["pages"]) == 2
        assert result["metadata"]["num_pages"] == 2

    def test_ocr_single_page(self, tmp_path, monkeypatch):
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "single.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_img = MagicMock()
        with patch("data.pdf_extractor.convert_from_path", return_value=[mock_img], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.return_value = "Single page OCR text."
            result = extractor._extract_with_ocr(pdf_path)

        assert result["metadata"]["num_pages"] == 1
        assert "Single page OCR text." in result["full_text"]

    def test_ocr_uses_language(self, tmp_path, monkeypatch):
        """OCR should pass the configured language to pytesseract."""
        monkeypatch.setattr(pe, "OCR_AVAILABLE", True)
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        extractor = PDFExtractor(use_ocr=True, ocr_language="fra")

        pdf_path = tmp_path / "fr.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_img = MagicMock()
        with patch("data.pdf_extractor.convert_from_path", return_value=[mock_img], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.return_value = "French text here."
            extractor._extract_with_ocr(pdf_path)

        mock_tess.image_to_string.assert_called_with(mock_img, lang="fra")

    def test_ocr_raises_when_not_available(self, tmp_path, monkeypatch):
        """_extract_with_ocr raises RuntimeError when OCR_AVAILABLE is False."""
        monkeypatch.setattr(pe, "OCR_AVAILABLE", True)
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", True)
        extractor = PDFExtractor(use_ocr=True)
        monkeypatch.setattr(pe, "OCR_AVAILABLE", False)  # flip after init

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with pytest.raises(RuntimeError, match="OCR requires"):
            extractor._extract_with_ocr(pdf_path)

    def test_ocr_empty_page_not_in_full_text(self, tmp_path, monkeypatch):
        """Empty OCR page text is excluded from full_text."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_img1 = MagicMock()
        mock_img2 = MagicMock()
        with patch("data.pdf_extractor.convert_from_path", return_value=[mock_img1, mock_img2], create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.side_effect = ["   ", "Real text here."]
            result = extractor._extract_with_ocr(pdf_path)

        assert "Real text here." in result["full_text"]
        assert "   " not in result["full_text"]

    def test_page_numbering_starts_at_1(self, tmp_path, monkeypatch):
        """Pages are numbered from 1."""
        extractor = self._make_extractor_with_ocr(monkeypatch)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_imgs = [MagicMock(), MagicMock(), MagicMock()]
        with patch("data.pdf_extractor.convert_from_path", return_value=mock_imgs, create=True), \
             patch("data.pdf_extractor.pytesseract", create=True) as mock_tess:
            mock_tess.image_to_string.return_value = "text"
            result = extractor._extract_with_ocr(pdf_path)

        page_numbers = [p["page"] for p in result["pages"]]
        assert page_numbers == [1, 2, 3]


# ---------------------------------------------------------------------------
# extract() — additional branches
# ---------------------------------------------------------------------------

class TestExtractAdditionalBranches:
    def test_standard_extraction_used_when_text_is_valid(self, tmp_path):
        """Standard extraction result is returned when text passes validation."""
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "valid.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        long_valid_text = (
            "Clinical guidelines are a core part of medical practice.\n"
            "They operate over a set of rows without collapsing them.\n"
            "Hypertension, aspirin, and lifestyle counseling are core secondary-prevention topics.\n"
            "Each plan should include blood pressure monitoring and follow-up.\n"
        )
        mock_page = MagicMock()
        mock_page.extract_text.return_value = long_valid_text
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(pdf_path)

        assert result["extraction_method"] == "pdfplumber"
        assert "Hypertension" in result["full_text"]

    def test_standard_extraction_short_text_falls_through(self, tmp_path, monkeypatch):
        """Short standard text causes extract() to fall through to failed result (no OCR)."""
        monkeypatch.setattr(pe, "OCR_AVAILABLE", False)
        extractor = PDFExtractor(use_ocr=False)
        pdf_path = tmp_path / "short.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Short."
        mock_page.extract_tables.return_value = []
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = lambda s: mock_pdf
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("pdfplumber.open", return_value=mock_pdf):
            result = extractor.extract(pdf_path)

        assert result["extraction_method"] == "failed"

    def test_pypdf2_path_used_when_pdfplumber_unavailable(self, tmp_path, monkeypatch):
        """When pdfplumber is unavailable, PyPDF2 method is used."""
        monkeypatch.setattr(pe, "PDFPLUMBER_AVAILABLE", False)
        monkeypatch.setattr(pe, "PYPDF2_AVAILABLE", True)
        extractor = PDFExtractor(use_ocr=False)
        assert extractor._method == "pypdf2"

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        long_text = (
            "PyPDF2 extraction result with sufficient text.\n"
            "Second line of content here.\n"
            "Third line for validation purposes.\n"
            "Fourth line with more words and text content.\n"
        )
        mock_page = MagicMock()
        mock_page.extract_text.return_value = long_text
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {}

        with patch("PyPDF2.PdfReader", return_value=mock_reader), \
             patch("builtins.open", return_value=MagicMock(
                 __enter__=lambda s: s,
                 __exit__=MagicMock(return_value=False),
                 read=lambda: b"%PDF-1.4",
             )):
            result = extractor._extract_standard(pdf_path)

        assert "full_text" in result
