#!/usr/bin/env python3
"""
Extract training-relevant content from manual PDFs.
v2: section-aware extraction, header/footer stripping, index detection,
    typed SQL Q&A generation, multi-turn conversations, near-duplicate removal.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.pdf_extractor import PDFExtractor


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
    if "docs.teradata.com" in lower and len(t) < 300:
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
# SQL content helpers
# ---------------------------------------------------------------------------

_SQL_KW_RE = re.compile(
    r"\b(SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|JOIN|ON|CREATE|INSERT|"
    r"UPDATE|DELETE|WITH|UNION|EXCEPT|INTERSECT|PARTITION\s+BY|OVER|"
    r"ROWS\s+BETWEEN|RANGE\s+BETWEEN|CSUM|MSUM|MAVG|MDIFF|MLINREG|QUANTILE|"
    r"RANK|DENSE_RANK|ROW_NUMBER|QUALIFY|USING|RETURNS?)\b",
    re.IGNORECASE,
)


def has_sql_content(text: str) -> bool:
    """True if text contains at least 2 SQL keywords (likely SQL code)."""
    return len(_SQL_KW_RE.findall(text)) >= 2


def extract_sql_blocks(text: str) -> List[str]:
    """Return paragraphs that contain SQL content."""
    return [
        para.strip()
        for para in re.split(r"\n\n+", text)
        if len(_SQL_KW_RE.findall(para)) >= 2
    ]


# Matches Teradata table-operator pattern: FROM FunctionName(
_TD_FUNC_RE = re.compile(r"\bFROM\s+(\w+)\s*\(", re.IGNORECASE)

# Suffixes that indicate a regular table/view name, not a function name
_NON_FUNC_SUFFIX_RE = re.compile(
    r"(?:_table|_test|_schema|_data|_view|_input|_output|_result|_sample|_train|_val)$",
    re.IGNORECASE,
)


def extract_sql_function_name(sql_text: str) -> Optional[str]:
    """
    Extract a Teradata table-operator function name from SQL like 'FROM nPath(...)'.
    Returns the function name string or None if not found / looks like a data table.
    """
    for m in _TD_FUNC_RE.finditer(sql_text):
        candidate = m.group(1)
        # Skip names that look like data tables rather than functions
        if _NON_FUNC_SUFFIX_RE.search(candidate):
            continue
        # Skip very short tokens (single letters) or all-lowercase short words
        if len(candidate) <= 2:
            continue
        return candidate
    return None


def _split_example_parts(example_text: str) -> Dict[str, str]:
    """
    Split an example block into input/sql/output sub-sections.
    Recognises standalone lines: Input, SQL Call, Output (and variants).
    Returns dict with keys 'input', 'sql', 'output' (values may be empty).
    """
    _INPUT_RE = re.compile(r"^\s*input(?:\s+(?:data|tables?))?\s*:?\s*$", re.I)
    _SQL_RE = re.compile(
        r"^\s*(sql\s+call|sql\s+query|sql\s+example|sql\s+statement)\s*:?\s*$", re.I
    )
    _OUTPUT_RE = re.compile(r"^\s*output(?:\s+(?:data|tables?))?\s*:?\s*$", re.I)

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
    Produces questions for: description, syntax, arguments, notes, examples, diagrams.
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

    # Description questions
    if desc and fn:
        pairs.append((f"What is {fn}?", desc))
        if len(desc) > 120:
            pairs.append((f"Describe the purpose of {fn}.", desc))
            pairs.append((f"When should {fn} be used?", desc))
            pairs.append((f"What problem does {fn} solve?", desc))

    # Syntax questions
    if syntax and fn:
        pairs.append((f"What is the syntax for {fn}?", syntax))
        if has_sql_content(syntax):
            pairs.append((f"How do you write a {fn} expression in SQL?", syntax))

    # Argument/parameter questions
    if args and fn:
        pairs.append((f"What are the arguments to {fn}?", args))
        pairs.append((f"What parameters does {fn} accept?", args))

    # Notes/restrictions questions
    if notes and fn:
        pairs.append((f"What are the usage notes for {fn}?", notes))
        pairs.append((f"What are the prerequisites for {fn}?", notes))
        pairs.append((f"Are there any gotchas or restrictions when using {fn}?", notes))

    # Example questions — check for structured Input/SQL Call/Output blocks
    for i, ex in enumerate(examples):
        if len(ex) < 30:
            continue
        label = "another example of" if i > 0 else "an example of"

        # Check for structured sub-sections
        sub = _split_example_parts(ex)
        sql_part = sub.get("sql", "")
        output_part = sub.get("output", "")

        if sql_part:
            # We have a structured example with a SQL call
            func_name = extract_sql_function_name(sql_part) or fn
            if fn:
                pairs.append((f"Show me a complete example of {fn} with input and output.", ex))
            pairs.append((f"Write a SQL query using {func_name}.", sql_part))
            if output_part and fn:
                pairs.append((f"What does {fn} return in this example?", output_part))
        else:
            # Generic example
            if fn:
                pairs.append((f"Show me {label} {fn}.", ex))
            if has_sql_content(ex):
                func_name = extract_sql_function_name(ex) or fn
                if func_name:
                    pairs.append((f"What SQL demonstrates how to use {func_name}?", ex))

    for cap in captions:
        if fn:
            pairs.append((f"What does the diagram show in the {fn} section?", cap))

    # Fallback: don't discard sections that didn't parse into labeled parts
    if not pairs:
        raw = (parsed.get("raw") or "").strip()
        if raw and len(raw) > 60:
            q = (
                f"What does the documentation say about {fn}?"
                if fn else "What does this section describe?"
            )
            pairs.append((q, raw))

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
    """Remove pairs where the answer is a near-duplicate of a previously seen answer."""
    seen: set = set()
    out = []
    for q, a in pairs:
        fp = _fingerprint(a)
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
