#!/usr/bin/env python3
"""
Extract training-relevant content from manual PDFs.
Section-aware extraction, header/footer stripping, index detection,
typed Q&A generation, multi-turn conversations, near-duplicate removal.
Domain-specific keyword/pattern detection is configured in domain_config.yaml.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.domain_config import (
    example_section_label_regexes,
    extract_named_pattern as extract_sql_function_name,
    has_structured_content as has_sql_content,
)
from data.pdf_extractor import PDFExtractor

try:
    from data.question_templates import (
        SYNTAX_QUESTIONS,
        ARGUMENT_QUESTIONS,
        NOTES_QUESTIONS,
        EXAMPLE_QUESTIONS,
        GENERAL_OVERVIEW_QUESTIONS,
        GENERAL_KEYCONCEPT_QUESTIONS,
        GENERAL_DETAIL_QUESTIONS,
        GENERAL_COMPARISON_QUESTIONS,
        GENERAL_APPLICATION_QUESTIONS,
        GENERAL_CAUSES_QUESTIONS,
        GENERAL_REQUIREMENTS_QUESTIONS,
        FINANCIAL_TRANSACTION_QUESTIONS,
        FINANCIAL_ACCOUNT_QUESTIONS,
        FINANCIAL_ANALYSIS_QUESTIONS,
    )
    _TEMPLATES_LOADED = True
except ImportError:
    _TEMPLATES_LOADED = False
    SYNTAX_QUESTIONS = ["What is the syntax for {fn}?"]
    ARGUMENT_QUESTIONS = ["What are the arguments to {fn}?"]
    NOTES_QUESTIONS = ["What are the usage notes for {fn}?"]
    EXAMPLE_QUESTIONS = ["Show me a complete example of {fn} with input and output."]
    GENERAL_OVERVIEW_QUESTIONS = ["What is {fn}?", "Summarize {fn}."]
    GENERAL_KEYCONCEPT_QUESTIONS = ["What are the key concepts in {fn}?"]
    GENERAL_DETAIL_QUESTIONS = ["Explain {fn} in detail.", "How does {fn} work?"]
    GENERAL_COMPARISON_QUESTIONS = ["What makes {fn} unique?"]
    GENERAL_APPLICATION_QUESTIONS = ["Give an example of {fn}."]
    GENERAL_CAUSES_QUESTIONS = ["What causes {fn}?"]
    GENERAL_REQUIREMENTS_QUESTIONS = ["What are the requirements for {fn}?"]
    FINANCIAL_TRANSACTION_QUESTIONS = ["What transactions appear in {fn}?"]
    FINANCIAL_ACCOUNT_QUESTIONS = ["What is the opening balance in {fn}?"]
    FINANCIAL_ANALYSIS_QUESTIONS = ["What spending categories appear in {fn}?"]


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def is_toc_page(text: str) -> bool:
    """True if page looks like a table of contents (dot-leaders + page numbers)."""
    if not text or len(text.strip()) < 50:
        return False
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    if "contents" in lines[0].lower() and len(lines) > 3:
        return True
    dot_lines = sum(1 for ln in lines if re.search(r"\.\s{0,2}\d+\s*$", ln))
    if dot_lines >= 3 and dot_lines >= len(lines) // 2:
        return True
    return False


def is_boilerplate_page(text: str, page_num: int) -> bool:
    """True for cover, copyright, or nearly empty pages."""
    t = (text or "").strip()
    if len(t) < 100:
        return True
    lower = t.lower()
    if "copyright" in lower and ("rights reserved" in lower or "©" in t):
        return True
    if "trademark" in lower and "safety" in lower and len(t) < 800:
        return True
    # Short pages that are dominated by a URL (cover/redirect pages from any vendor)
    if re.search(r"https?://\S+", lower) and len(t) < 300:
        return True
    return False


def is_index_page(text: str) -> bool:
    """
    True if page looks like a back-of-book index.
    Signs: single-letter alphabetical section headers, many short lines
    with trailing page numbers or comma-separated number lists.
    """
    if not text or len(text.strip()) < 50:
        return False
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    # Alphabetical section headers like "A", "B", "C"
    single_letter_lines = sum(1 for ln in lines if re.match(r"^[A-Z]$", ln))
    if single_letter_lines >= 3:
        return True
    # Lines ending in page numbers: "CSUM 42, 56, 103"
    index_entry_lines = sum(
        1 for ln in lines
        if re.search(r"\d+(?:,\s*\d+)*\s*$", ln) and len(ln) < 80
    )
    if index_entry_lines >= 5 and index_entry_lines >= len(lines) * 0.5:
        return True
    return False


def is_substantive_page(text: str) -> bool:
    """True only if the page has enough real content (not just stray text/numbers)."""
    if not text or len(text.strip()) < 80:
        return False
    lines = [ln for ln in text.split("\n") if len(ln.strip()) > 20]
    return len(lines) >= 2 and sum(len(ln) for ln in lines) > 200


def filter_pages(
    pages: List[Dict],
    skip_toc: bool = True,
    skip_boilerplate: bool = True,
    skip_index: bool = True,
) -> List[Dict]:
    """Return only substantive pages, skipping TOC/boilerplate/index."""
    kept = []
    for p in pages:
        text = (p.get("text") or "").strip()
        page_num = p.get("page", 0)
        if skip_boilerplate and is_boilerplate_page(text, page_num):
            continue
        if skip_toc and is_toc_page(text):
            continue
        if skip_index and is_index_page(text):
            continue
        if not is_substantive_page(text):
            continue
        kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# Running header / footer detection and removal
# ---------------------------------------------------------------------------

def _normalize_for_hf(line: str) -> str:
    """Normalize a line for header/footer matching (strip page numbers, lowercase)."""
    ln = line.strip()
    ln = re.sub(r"^\d{1,4}\s+", "", ln)   # leading page number
    ln = re.sub(r"\s+\d{1,4}$", "", ln)   # trailing page number
    return ln.lower().strip()


def detect_running_headers_footers(
    pages: List[Dict], min_freq: float = 0.35
) -> Tuple[set, set]:
    """
    Detect lines that repeat as running headers/footers across >= min_freq of pages.
    Only inspects the first 2 and last 2 lines of each page.
    Returns (header_norms, footer_norms) as sets of normalized strings.
    """
    n = len(pages)
    if n < 4:
        return set(), set()
    top_counts: Counter = Counter()
    bot_counts: Counter = Counter()
    for p in pages:
        text = (p.get("text") or "").strip()
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines[:2]:
            norm = _normalize_for_hf(ln)
            if len(norm) > 4:
                top_counts[norm] += 1
        for ln in lines[-2:]:
            norm = _normalize_for_hf(ln)
            if len(norm) > 4:
                bot_counts[norm] += 1
    threshold = max(3, int(n * min_freq))
    headers = {ln for ln, cnt in top_counts.items() if cnt >= threshold}
    footers = {ln for ln, cnt in bot_counts.items() if cnt >= threshold}
    return headers, footers


def strip_running_headers_footers(text: str, headers: set, footers: set) -> str:
    """Remove detected header/footer lines from a page's text."""
    if not headers and not footers:
        return text
    out = []
    for ln in text.split("\n"):
        norm = _normalize_for_hf(ln)
        if norm in headers or norm in footers:
            continue
        if re.match(r"^\s*\d{1,4}\s*$", ln):   # standalone page number
            continue
        out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section detection and multi-page stitching
# ---------------------------------------------------------------------------

_HEADING_RE = [
    # All-caps line: function names like "CSUM", "MOVING AVERAGE"
    re.compile(r"^[A-Z][A-Z0-9 _\-/]{1,59}$"),
    # Numbered section: "3.2 Ordered Analytical Functions"
    re.compile(r"^\d+(?:\.\d+)*\s+[A-Z][^\n]{5,79}$"),
    # Title-case short phrase without trailing period: "Ordered Analytical Functions"
    re.compile(r"^(?:[A-Z][a-z]+\s+){1,6}[A-Z][a-z]+$"),
]

# SQL reserved words that should never be treated as function/section headings
_SQL_RESERVED = frozenset({
    "USING", "FROM", "SELECT", "WHERE", "ON", "WITH", "JOIN", "OVER",
    "HAVING", "GROUP", "ORDER", "UNION", "EXCEPT", "INTERSECT",
    "INSERT", "UPDATE", "DELETE", "CREATE", "TABLE", "INDEX", "VIEW",
    "RETURNS", "RETURN", "NULL", "NOT", "AND", "OR", "BY", "AS",
    "PARTITION", "QUALIFY", "CASE", "WHEN", "THEN", "ELSE", "END",
    "ROWS", "RANGE", "BETWEEN", "UNBOUNDED", "PRECEDING", "FOLLOWING",
    "CURRENT", "ROW", "DISTINCT", "ALL", "INNER", "OUTER", "LEFT",
    "RIGHT", "FULL", "CROSS", "NATURAL", "SET", "INTO", "VALUES",
    "IS", "IN", "LIKE", "EXISTS", "ANY", "SOME",
    "INPUT", "OUTPUT",  # sub-section headers, not function names
})


def _is_heading_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 3 or len(line) > 80:
        return False
    if line.endswith((".", ":", ",")):
        return False
    if len(line.split()) > 10:
        return False
    # Exclude lines starting with a digit (data rows, page numbers)
    if re.match(r"^\d", line):
        return False
    # Exclude lines containing date/time patterns (data rows)
    if re.search(r"\d{4}-\d{2}-\d{2}", line):
        return False
    # Exclude lines with 2+ numeric tokens (data rows like "1 M 100")
    if len(re.findall(r"\b\d+\b", line)) >= 2:
        return False
    # Exclude SQL reserved words used as standalone headings
    if line.upper() in _SQL_RESERVED:
        return False
    # Exclude table-schema header phrases (e.g. "Column Data Type Description",
    # "Input Table Schema", "Output Table Schema")
    _DOC_STRUCTURE_WORDS = {"Column", "Schema", "Type", "Description", "Table", "Row"}
    words = set(line.split())
    if len(words & _DOC_STRUCTURE_WORDS) >= 2:
        return False
    return any(pat.match(line) for pat in _HEADING_RE)


def detect_page_heading(text: str) -> Optional[str]:
    """Return the first plausible heading found in the top 5 lines of a page."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines[:5]:
        if _is_heading_line(ln):
            return ln
    return None


def _make_section(heading: str, pages: List[Dict]) -> Dict:
    text = "\n\n".join((p.get("text") or "").strip() for p in pages)
    tables: List = []
    for p in pages:
        tables.extend(p.get("tables", []))
    return {
        "heading": heading,
        "text": text,
        "page_start": pages[0].get("page", 0),
        "page_end": pages[-1].get("page", 0),
        "tables": tables,
    }


def group_pages_into_sections(pages: List[Dict]) -> List[Dict]:
    """
    Group consecutive pages that belong to the same logical section.
    A new section starts when a new heading is detected at the top of a page.
    Returns list of section dicts: {heading, text, page_start, page_end, tables}.
    """
    if not pages:
        return []
    sections: List[Dict] = []
    cur_heading = detect_page_heading(pages[0].get("text", "")) or "Introduction"
    cur_pages = [pages[0]]
    for p in pages[1:]:
        h = detect_page_heading(p.get("text", ""))
        if h and h != cur_heading:
            sections.append(_make_section(cur_heading, cur_pages))
            cur_heading = h
            cur_pages = [p]
        else:
            cur_pages.append(p)
    if cur_pages:
        sections.append(_make_section(cur_heading, cur_pages))
    return sections


# ---------------------------------------------------------------------------
# Structured content helpers (patterns loaded from domain_config.yaml)
# ---------------------------------------------------------------------------

def extract_sql_blocks(text: str) -> List[str]:
    """Return paragraphs that contain structured domain content."""
    return [
        para.strip()
        for para in re.split(r"\n\n+", text)
        if has_sql_content(para)
    ]


def _split_example_parts(example_text: str) -> Dict[str, str]:
    """
    Split an example block into input/sql/output sub-sections.
    Recognises standalone lines: Input, SQL Call, Output (and variants).
    Returns dict with keys 'input', 'sql', 'output' (values may be empty).
    """
    label_res = example_section_label_regexes()
    _INPUT_RE = label_res["input"]
    _SQL_RE = label_res["structured"]
    _OUTPUT_RE = label_res["output"]

    result: Dict[str, str] = {"input": "", "sql": "", "output": ""}
    current: Optional[str] = None
    buf: List[str] = []

    for para in re.split(r"\n\n+", example_text):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        rest = "\n".join(lines[1:]).strip()
        if _INPUT_RE.match(first):
            if current and buf:
                result[current] = "\n\n".join(buf).strip()
            current = "input"
            buf = [rest] if rest else []
        elif _SQL_RE.match(first):
            if current and buf:
                result[current] = "\n\n".join(buf).strip()
            current = "sql"
            buf = [rest] if rest else []
        elif _OUTPUT_RE.match(first):
            if current and buf:
                result[current] = "\n\n".join(buf).strip()
            current = "output"
            buf = [rest] if rest else []
        elif current is not None:
            buf.append(para.strip())

    if current and buf:
        result[current] = "\n\n".join(buf).strip()
    return result


# ---------------------------------------------------------------------------
# Structured section parsing
# ---------------------------------------------------------------------------

_PART_LABELS: Dict[str, re.Pattern] = {
    "description": re.compile(r"^\s*(description|overview|purpose|about)\s*:?\s*$", re.I),
    "syntax": re.compile(r"^\s*(syntax|format)\s*:?\s*$", re.I),
    "arguments": re.compile(r"^\s*(arguments?|parameters?|operands?)\s*:?\s*$", re.I),
    "notes": re.compile(
        r"^\s*(notes?|usage notes?|remarks?|restrictions?|considerations?|caveats?)\s*:?\s*$",
        re.I,
    ),
    "examples": re.compile(r"^\s*(examples?|sample)\s*:?\s*$", re.I),
    # Sub-section headers within examples — all map to the examples bucket
    "result": re.compile(r"^\s*(results?|return value)\s*:?\s*$", re.I),
    "input_data": re.compile(r"^\s*input(?:\s+(?:data|tables?))?\s*:?\s*$", re.I),
    "sql_call": re.compile(
        r"^\s*(sql\s+call|sql\s+query|sql\s+example|sql\s+statement)\s*:?\s*$", re.I
    ),
    "output": re.compile(r"^\s*output(?:\s+(?:data|tables?))?\s*:?\s*$", re.I),
}

# Labels that feed content into the examples bucket
_EXAMPLE_SUB_LABELS = frozenset({"result", "input_data", "sql_call", "output"})


def parse_function_section(section_text: str, heading: str = "") -> Dict:
    """
    Parse a function/feature section into labeled parts.
    Returns: {heading, description, syntax, arguments, notes, examples, raw}
    Sub-section headers Input/SQL Call/Output are mapped into the examples bucket.
    """
    parts: Dict = {
        "heading": heading,
        "description": [],
        "syntax": [],
        "arguments": [],
        "notes": [],
        "examples": [],
        "raw": section_text,
    }
    current = "description"
    for para in re.split(r"\n\n+", section_text):
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        label = next((k for k, p in _PART_LABELS.items() if p.match(first)), None)
        if label:
            current = "examples" if label in _EXAMPLE_SUB_LABELS else label
            rest = "\n".join(lines[1:]).strip()
            if rest:
                parts[current].append(rest)
        else:
            parts[current].append(para.strip())
    return parts


# ---------------------------------------------------------------------------
# Figure caption extraction
# ---------------------------------------------------------------------------

_FIG_RE = re.compile(r"(?:Figure|Fig\.?)\s+[\d\-\.]+\s*[:\-]\s*(.+)", re.I)


def extract_figure_captions(text: str) -> List[str]:
    """Extract 'Figure X-Y: description' captions from text."""
    return [m.group(0).strip() for m in _FIG_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Typed Q&A generation
# ---------------------------------------------------------------------------

def generate_typed_qa(parsed: Dict, source_label: str = "") -> List[Tuple[str, str]]:
    """
    Generate diverse typed Q&A pairs from a parsed section.

    For technical sections (those with structured syntax/arguments/notes), generates
    SQL-oriented Q&A using the shared template lists from question_templates.py/yaml.
    For general prose sections (textbooks, non-technical manuals), falls back to the
    GENERAL_* template lists, producing 5-10 pairs per section instead of one.

    Each section type generates many question variants to maximise training signal.
    """
    pairs: List[Tuple[str, str]] = []
    heading = (parsed.get("heading") or "").strip()
    fn = heading or source_label

    desc = "\n\n".join(parsed.get("description", [])).strip()
    syntax = "\n\n".join(parsed.get("syntax", [])).strip()
    args = "\n\n".join(parsed.get("arguments", [])).strip()
    notes = "\n\n".join(parsed.get("notes", [])).strip()
    examples = [e.strip() for e in parsed.get("examples", []) if e.strip()]
    captions = extract_figure_captions(parsed.get("raw", ""))

    # Description questions — answer must match the actual question being asked
    if desc and fn:
        pairs.append((f"What is {fn}?", desc))
        pairs.append((f"What does {fn} do?", desc))

        sentences = re.split(r"(?<=[.!?])\s+", desc)
        first_sentence = sentences[0].strip() if sentences else ""
        if first_sentence and len(first_sentence) < len(desc) - 20:
            pairs.append((f"Define {fn} in one sentence.", first_sentence))

        func_type_keywords = {"window", "aggregate", "table function", "ordered analytical", "olap", "analytic"}
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in func_type_keywords):
            for sent in sentences:
                if any(kw in sent.lower() for kw in func_type_keywords):
                    pairs.append((f"Is {fn} a window function, aggregate, or table function?", sent.strip()))
                    pairs.append((f"What category of SQL function is {fn}?", sent.strip()))
                    break

    # Syntax questions — use shared SYNTAX_QUESTIONS + SQL-specific extensions
    if syntax and fn:
        for tmpl in SYNTAX_QUESTIONS:
            pairs.append((tmpl.format(fn=fn), syntax))
        if has_sql_content(syntax):
            for sql_q in [
                f"Can {fn} be used in a subquery?",
                f"Does {fn} require a PARTITION BY clause?",
                f"What goes inside the OVER() clause in {fn}?",
            ]:
                pairs.append((sql_q, syntax))

    # Argument/parameter questions — use shared ARGUMENT_QUESTIONS
    if args and fn:
        for tmpl in ARGUMENT_QUESTIONS:
            pairs.append((tmpl.format(fn=fn), args))

    # Notes/restrictions questions — use shared NOTES_QUESTIONS
    if notes and fn:
        for tmpl in NOTES_QUESTIONS:
            pairs.append((tmpl.format(fn=fn), notes))
        if not any("use case" in q.lower() for q, _ in pairs):
            pairs.append((f"When should I use {fn}?", notes))
            pairs.append((f"What problem does {fn} solve?", f"{desc}\n\n{notes}" if desc else notes))

    # Example questions — use shared EXAMPLE_QUESTIONS + SQL-specific additions
    for i, ex in enumerate(examples):
        if len(ex) < 30:
            continue
        label = "another example of" if i > 0 else "an example of"
        sub = _split_example_parts(ex)
        sql_part = sub.get("sql", "")
        output_part = sub.get("output", "")

        if sql_part:
            func_name = extract_sql_function_name(sql_part) or fn
            if fn:
                for tmpl in EXAMPLE_QUESTIONS:
                    pairs.append((tmpl.format(fn=fn), ex))
                pairs.append((f"What would the result be if the {fn} query partition had only one row?", ex))
                pairs.append((f"What are the key SQL clauses in this {fn} example?", sql_part))
            pairs.append((f"Write a SQL query using {func_name}.", sql_part))
            if output_part and fn:
                pairs.append((f"What does {fn} return in this example?", output_part))
        else:
            if fn:
                pairs.append((f"Show me {label} {fn}.", ex))
                pairs.append((f"What is the purpose of this {fn} example?", ex))
                pairs.append((f"How would you modify this {fn} example to change the output?", ex))
            if has_sql_content(ex):
                func_name = extract_sql_function_name(ex) or fn
                if func_name:
                    pairs.append((f"What SQL demonstrates how to use {func_name}?", ex))
                if fn:
                    pairs.append((f"Explain the SQL logic in this {fn} example.", ex))
                    pairs.append((f"What would happen if you changed the ORDER BY in this {fn} example?", ex))

    for cap in captions:
        if fn:
            pairs.append((f"What does the diagram show in the {fn} section?", cap))

    # Fallback for general / non-technical content:
    # If no structured technical pairs were generated, apply GENERAL_* templates
    # so textbook chapters, prose sections, and non-SQL manuals still produce rich Q&A.
    if not pairs:
        raw = (parsed.get("raw") or "").strip()
        if raw and len(raw) > 60 and fn:
            pairs.extend(_generate_generic_qa(fn, raw))
        elif raw and len(raw) > 60:
            pairs.append(("What does this section describe?", raw))

    return pairs


def _generate_generic_qa(topic: str, text: str) -> List[Tuple[str, str]]:
    """
    Generate Q&A pairs for general (non-technical) text using GENERAL_* templates.

    Applies different template groups depending on what the text signals:
    - All sections get overview + detail templates.
    - Sections mentioning comparisons, causes, or requirements get those groups too.
    - Each template group produces 3-8 pairs per section (vs 1 in the old fallback).
    """
    pairs: List[Tuple[str, str]] = []
    lower = text.lower()

    # Overview and detail are always useful
    for tmpl in GENERAL_OVERVIEW_QUESTIONS:
        pairs.append((tmpl.format(fn=topic), text))
    for tmpl in GENERAL_DETAIL_QUESTIONS:
        pairs.append((tmpl.format(fn=topic), text))

    # Key concepts — if the section defines terms or lists principles
    definition_signals = ("is defined as", "refers to", "means", "is a", "is the")
    if any(sig in lower for sig in definition_signals):
        for tmpl in GENERAL_KEYCONCEPT_QUESTIONS:
            pairs.append((tmpl.format(fn=topic), text))

    # Application — if the section gives examples or describes use
    application_signals = ("for example", "such as", "in practice", "can be used", "is used")
    if any(sig in lower for sig in application_signals):
        for tmpl in GENERAL_APPLICATION_QUESTIONS:
            pairs.append((tmpl.format(fn=topic), text))

    # Comparison — if the section contrasts approaches
    comparison_signals = ("compared to", "unlike", "whereas", "in contrast", "however", "advantage", "disadvantage")
    if any(sig in lower for sig in comparison_signals):
        for tmpl in GENERAL_COMPARISON_QUESTIONS:
            pairs.append((tmpl.format(fn=topic), text))

    # Causes/effects — causal language
    causal_signals = ("because", "therefore", "as a result", "leads to", "causes", "results in")
    if any(sig in lower for sig in causal_signals):
        for tmpl in GENERAL_CAUSES_QUESTIONS:
            pairs.append((tmpl.format(fn=topic), text))

    # Requirements — conditional or prerequisite language
    requirement_signals = ("must", "required", "necessary", "prerequisite", "need to", "in order to")
    if any(sig in lower for sig in requirement_signals):
        for tmpl in GENERAL_REQUIREMENTS_QUESTIONS:
            pairs.append((tmpl.format(fn=topic), text))

    return pairs


# ---------------------------------------------------------------------------
# Multi-turn conversation generation
# ---------------------------------------------------------------------------

def generate_multiturn_conversation(parsed: Dict) -> Optional[List[Dict]]:
    """
    Build a multi-turn ShareGPT conversation from a parsed function section.
    Returns None if there isn't enough content for at least 2 turns.
    """
    heading = (parsed.get("heading") or "").strip()
    desc = "\n\n".join(parsed.get("description", [])).strip()
    syntax = "\n\n".join(parsed.get("syntax", [])).strip()
    examples = [e.strip() for e in parsed.get("examples", []) if e.strip()]

    if not heading or not desc:
        return None

    turns = [
        {"role": "user", "content": f"What is {heading}?"},
        {"role": "assistant", "content": desc},
    ]
    if syntax:
        turns += [
            {"role": "user", "content": f"What is the syntax for {heading}?"},
            {"role": "assistant", "content": syntax},
        ]
    if examples:
        turns += [
            {"role": "user", "content": f"Can you show me an example of {heading}?"},
            {"role": "assistant", "content": examples[0]},
        ]
    return turns if len(turns) >= 4 else None


# ---------------------------------------------------------------------------
# Near-duplicate removal
# ---------------------------------------------------------------------------

def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized[:400].encode()).hexdigest()


def deduplicate_qa_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove exact duplicate (question, answer) pairs.
    Different questions with the same answer are preserved — they are valuable
    training signal teaching the model to respond to varied phrasings.
    """
    seen: set = set()
    out = []
    for q, a in pairs:
        fp = _fingerprint(q + "\n" + a)
        if fp not in seen:
            seen.add(fp)
            out.append((q, a))
    return out


# ---------------------------------------------------------------------------
# Manual label
# ---------------------------------------------------------------------------

def manual_label_from_path(pdf_path: Path) -> str:
    """Derive a filesystem-safe label from the PDF filename."""
    stem = pdf_path.stem
    return re.sub(r"[^\w\-]", "_", stem).lower().strip("_")


# ---------------------------------------------------------------------------
# Main extraction API
# ---------------------------------------------------------------------------

def extract_manual(
    pdf_path: Path,
    extractor: "PDFExtractor",
    skip_toc: bool = True,
    skip_boilerplate: bool = True,
    skip_index: bool = True,
    use_sections: bool = True,
) -> Dict:
    """
    Extract and structure a manual PDF for training data generation.

    Pipeline:
      1. Raw PDF extraction (text + tables per page)
      2. Filter junk pages (TOC, boilerplate, index, near-empty)
      3. Detect and strip running headers/footers
      4. Re-filter after cleaning
      5. Group pages into logical sections by heading detection

    Returns dict with keys:
      source_file, label, full_text, pages, sections,
      headers_detected, footers_detected, metadata, extraction_method.
    """
    raw = extractor.extract(pdf_path)
    pages = raw.get("pages", [])

    # Step 1: filter junk pages
    filtered = filter_pages(
        pages,
        skip_toc=skip_toc,
        skip_boilerplate=skip_boilerplate,
        skip_index=skip_index,
    )

    # Step 2: detect and strip running headers/footers
    headers, footers = detect_running_headers_footers(filtered)
    for p in filtered:
        p["text"] = strip_running_headers_footers(p.get("text", ""), headers, footers)

    # Step 3: re-filter after cleaning (some pages may now be too short)
    filtered = [p for p in filtered if is_substantive_page(p.get("text", ""))]

    # Step 4: build full text
    full_text = "\n\n".join(
        (p.get("text") or "").strip()
        for p in filtered
        if (p.get("text") or "").strip()
    )

    # Step 5: group into logical sections
    sections = group_pages_into_sections(filtered) if use_sections else []

    label = manual_label_from_path(pdf_path)
    return {
        "source_file": pdf_path.name,
        "label": label,
        "full_text": full_text,
        "pages": filtered,
        "sections": sections,
        "headers_detected": sorted(headers),
        "footers_detected": sorted(footers),
        "metadata": {
            **raw.get("metadata", {}),
            "num_pages_total": len(pages),
            "num_pages_kept": len(filtered),
            "num_sections": len(sections),
        },
        "extraction_method": raw.get("extraction_method", "unknown"),
    }
