#!/usr/bin/env python3
"""
PDF text extraction for SLM training data.
Extracts text from PDFs with optional OCR for scanned documents.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files for training data."""

    def __init__(self, use_ocr: bool = True, ocr_language: str = "eng"):
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        if PDFPLUMBER_AVAILABLE:
            self._method = "pdfplumber"
        elif PYPDF2_AVAILABLE:
            self._method = "pypdf2"
        else:
            raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber")

    def extract(self, pdf_path: Path, force_ocr: bool = False) -> Dict:
        """Extract text from a PDF. Uses OCR if needed for scanned PDFs."""
        if not force_ocr:
            try:
                result = self._extract_standard(pdf_path)
                full_text = result.get("full_text", "")
                if full_text and len(full_text.strip()) > 100 and self._is_valid_text(full_text):
                    result["extraction_method"] = self._method
                    return result
            except Exception as e:
                logger.warning(f"Standard extraction failed for {pdf_path.name}: {e}")
        if self.use_ocr:
            try:
                result = self._extract_with_ocr(pdf_path)
                result["extraction_method"] = "ocr"
                return result
            except Exception as e:
                logger.error(f"OCR failed for {pdf_path.name}: {e}")
        return {
            "metadata": {"num_pages": 0},
            "pages": [],
            "full_text": "",
            "extraction_method": "failed",
        }

    def _extract_standard(self, pdf_path: Path) -> Dict:
        if self._method == "pdfplumber":
            return self._extract_pdfplumber(pdf_path)
        return self._extract_pypdf2(pdf_path)

    def _extract_pdfplumber(self, pdf_path: Path) -> Dict:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                tables = []
                try:
                    raw_tables = page.extract_tables() or []
                    tables = [t for t in raw_tables if t]
                except Exception:
                    tables = []
                pages.append({"page": i, "text": text.strip(), "tables": tables})
            meta = {"num_pages": len(pdf.pages), "metadata": pdf.metadata or {}}
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return {"metadata": meta, "pages": pages, "full_text": full_text}

    def _extract_pypdf2(self, pdf_path: Path) -> Dict:
        pages = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = {"num_pages": len(reader.pages), "metadata": reader.metadata or {}}
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                pages.append({"page": i, "text": text.strip()})
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return {"metadata": meta, "pages": pages, "full_text": full_text}

    def _extract_with_ocr(self, pdf_path: Path) -> Dict:
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR requires: pip install pytesseract pdf2image")
        images = convert_from_path(str(pdf_path))
        pages = []
        for i, img in enumerate(images, 1):
            text = pytesseract.image_to_string(img, lang=self.ocr_language)
            pages.append({"page": i, "text": text.strip()})
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        return {
            "metadata": {"num_pages": len(images)},
            "pages": pages,
            "full_text": full_text,
        }

    def _is_valid_text(self, text: str) -> bool:
        if not text or len(text.strip()) < 50:
            return False
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) < 3 or len(text.split()) < 10:
            return False
        alpha = sum(1 for c in text if c.isalpha())
        total = len(text.replace(" ", "").replace("\n", ""))
        return total > 0 and (alpha / total) >= 0.4

    def extract_directory(self, dir_path: Path) -> List[Dict]:
        """Extract text from all PDFs in a directory."""
        results = []
        for path in Path(dir_path).glob("*.pdf"):
            r = self.extract(path)
            if r.get("full_text"):
                r["source_file"] = path.name
                results.append(r)
        return results
