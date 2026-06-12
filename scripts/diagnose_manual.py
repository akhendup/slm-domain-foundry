#!/usr/bin/env python3
"""
Diagnostic script for manual PDF extraction.

Runs the full extraction pipeline on a single PDF and prints a detailed
report at each stage so you can see exactly what training data is produced
and where content is lost or transformed.

Usage:
    python scripts/diagnose_manual.py path/to/manual.pdf
    python scripts/diagnose_manual.py path/to/manual.pdf --verbose
    python scripts/diagnose_manual.py path/to/manual.pdf -o report.txt
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pdf_extractor import PDFExtractor
from data.manual_extractor import (
    is_toc_page,
    is_boilerplate_page,
    is_index_page,
    is_substantive_page,
    filter_pages,
    detect_running_headers_footers,
    strip_running_headers_footers,
    group_pages_into_sections,
    parse_function_section,
    generate_typed_qa,
    generate_multiturn_conversation,
    deduplicate_qa_pairs,
    has_sql_content,
    extract_sql_blocks,
    extract_figure_captions,
)
from data.chunking import chunk_text, chunk_text_sql_aware, is_sql_paragraph


def hr(char: str = "─", width: int = 72) -> str:
    return char * width


def section_header(title: str) -> str:
    return f"\n{hr('═')}\n  {title}\n{hr('═')}"


def diagnose(pdf_path: Path, verbose: bool = False, output_path: Path = None) -> dict:
    lines: list = []

    def emit(*args) -> None:
        text = " ".join(str(a) for a in args)
        lines.append(text)
        if output_path is None:
            print(text)

    emit(section_header(f"DIAGNOSTIC REPORT: {pdf_path.name}"))

    # ── Stage 0: Raw extraction ──────────────────────────────────────────
    emit(f"\n[Stage 0] Raw PDF extraction")
    emit(hr())
    extractor = PDFExtractor(use_ocr=False)
    raw = extractor.extract(pdf_path)
    all_pages = raw.get("pages", [])
    char_counts = [len(p.get("text") or "") for p in all_pages]
    emit(f"  Extraction method  : {raw.get('extraction_method', '?')}")
    emit(f"  Total pages        : {len(all_pages)}")
    emit(
        f"  Chars/page (min/avg/max): "
        f"{min(char_counts, default=0)} / "
        f"{int(sum(char_counts) / max(len(char_counts), 1))} / "
        f"{max(char_counts, default=0)}"
    )
    table_counts = [len(p.get("tables", [])) for p in all_pages]
    emit(f"  Pages with tables  : {sum(1 for t in table_counts if t > 0)}")
    emit(f"  Total tables       : {sum(table_counts)}")
    if verbose:
        emit("\n  First 3 pages (raw, first 250 chars):")
        for p in all_pages[:3]:
            emit(f"    Page {p['page']:>3}: {repr((p.get('text') or '')[:250])}")

    # ── Stage 1: Page classification ────────────────────────────────────
    emit(f"\n[Stage 1] Page classification")
    emit(hr())
    toc_pages = [p["page"] for p in all_pages if is_toc_page(p.get("text", ""))]
    boilerplate_pages = [
        p["page"]
        for p in all_pages
        if is_boilerplate_page(p.get("text", ""), p.get("page", 0))
    ]
    index_pages = [p["page"] for p in all_pages if is_index_page(p.get("text", ""))]
    non_sub = [p["page"] for p in all_pages if not is_substantive_page(p.get("text", ""))]

    emit(f"  TOC pages          : {toc_pages or 'none'}")
    emit(f"  Boilerplate pages  : {boilerplate_pages or 'none'}")
    emit(f"  Index pages        : {index_pages or 'none'}")
    emit(f"  Non-substantive    : {len(non_sub)} pages ({non_sub[:10]}{'...' if len(non_sub) > 10 else ''})")

    filtered = filter_pages(all_pages, skip_toc=True, skip_boilerplate=True, skip_index=True)
    emit(f"  Pages after filter : {len(filtered)} / {len(all_pages)}")

    # ── Stage 2: Header/footer detection ────────────────────────────────
    emit(f"\n[Stage 2] Running header/footer detection")
    emit(hr())
    headers, footers = detect_running_headers_footers(filtered)
    emit(f"  Headers detected   : {len(headers)}")
    for h in sorted(headers)[:8]:
        emit(f"    • {h}")
    if len(headers) > 8:
        emit(f"    ... and {len(headers) - 8} more")
    emit(f"  Footers detected   : {len(footers)}")
    for f in sorted(footers)[:8]:
        emit(f"    • {f}")
    if len(footers) > 8:
        emit(f"    ... and {len(footers) - 8} more")

    for p in filtered:
        p["text"] = strip_running_headers_footers(p.get("text", ""), headers, footers)

    after_strip = [p for p in filtered if is_substantive_page(p.get("text", ""))]
    emit(f"  Pages after strip  : {len(after_strip)} / {len(filtered)}")

    if verbose and filtered:
        emit("\n  Sample page after header/footer strip (page 5 or first):")
        sample = next((p for p in after_strip if p["page"] >= 5), after_strip[0] if after_strip else None)
        if sample:
            emit(f"    Page {sample['page']}:\n{sample['text'][:400]}")

    # ── Stage 3: Section detection ───────────────────────────────────────
    emit(f"\n[Stage 3] Section detection and page stitching")
    emit(hr())
    sections = group_pages_into_sections(after_strip)
    emit(f"  Sections detected  : {len(sections)}")
    emit(f"\n  {'Pages':<10} {'Heading'}")
    emit(f"  {'-'*8}  {'-'*50}")
    for s in sections[:30]:
        span = f"{s['page_start']}-{s['page_end']}"
        emit(f"  {span:<10} {s['heading'][:60]}")
    if len(sections) > 30:
        emit(f"  ... and {len(sections) - 30} more sections")

    # ── Stage 4: Section parsing ─────────────────────────────────────────
    emit(f"\n[Stage 4] Section content parsing (first 5 sections)")
    emit(hr())
    for s in sections[:5]:
        parsed = parse_function_section(s["text"], s["heading"])
        desc_len = sum(len(x) for x in parsed["description"])
        syn_len = sum(len(x) for x in parsed["syntax"])
        arg_len = sum(len(x) for x in parsed["arguments"])
        note_len = sum(len(x) for x in parsed["notes"])
        ex_count = len(parsed["examples"])
        sql_blocks = extract_sql_blocks(s["text"])
        captions = extract_figure_captions(s["text"])
        emit(f"\n  ── {parsed['heading']} (pages {s['page_start']}–{s['page_end']}) ──")
        emit(f"     description : {desc_len} chars")
        emit(f"     syntax      : {syn_len} chars")
        emit(f"     arguments   : {arg_len} chars")
        emit(f"     notes       : {note_len} chars")
        emit(f"     examples    : {ex_count} block(s)")
        emit(f"     SQL blocks  : {len(sql_blocks)}")
        emit(f"     Fig captions: {len(captions)}")
        if verbose and parsed["description"]:
            emit(f"     desc[0]     : {parsed['description'][0][:200]}")
        if verbose and parsed["syntax"]:
            emit(f"     syntax[0]   : {parsed['syntax'][0][:200]}")

    # ── Stage 5: Q&A generation ──────────────────────────────────────────
    emit(f"\n[Stage 5] Q&A pair generation")
    emit(hr())
    all_pairs = []
    multiturn_convs = []
    for s in sections:
        parsed = parse_function_section(s["text"], s["heading"])
        all_pairs.extend(generate_typed_qa(parsed))
        conv = generate_multiturn_conversation(parsed)
        if conv:
            multiturn_convs.append(conv)

    emit(f"  Pairs before dedup : {len(all_pairs)}")
    deduped = deduplicate_qa_pairs(all_pairs)
    emit(f"  Pairs after dedup  : {len(deduped)}")
    emit(f"  Multi-turn convs   : {len(multiturn_convs)}")

    # Question type breakdown
    q_types: dict = {
        "What is / Describe / Purpose": 0,
        "When to use / Problem solved": 0,
        "Syntax / SQL expression": 0,
        "Arguments / Parameters": 0,
        "Example / Demonstrate / SQL": 0,
        "Usage notes / Prerequisites / Gotchas": 0,
        "Diagram": 0,
        "Other / Fallback": 0,
    }
    for q, _ in deduped:
        ql = q.lower()
        if "syntax" in ql or "write a sql" in ql or "expression" in ql or "write a " in ql:
            q_types["Syntax / SQL expression"] += 1
        elif "argument" in ql or "parameter" in ql:
            q_types["Arguments / Parameters"] += 1
        elif "example" in ql or "demonstrate" in ql or "show me" in ql or "sql demonstrates" in ql:
            q_types["Example / Demonstrate / SQL"] += 1
        elif "usage notes" in ql or "prerequisites" in ql or "gotchas" in ql or "restrictions" in ql:
            q_types["Usage notes / Prerequisites / Gotchas"] += 1
        elif "when should" in ql or "problem does" in ql:
            q_types["When to use / Problem solved"] += 1
        elif "diagram" in ql:
            q_types["Diagram"] += 1
        elif ql.startswith("what is") or ql.startswith("describe"):
            q_types["What is / Describe / Purpose"] += 1
        else:
            q_types["Other / Fallback"] += 1

    emit(f"\n  Question type breakdown:")
    for qtype, count in q_types.items():
        bar = "█" * min(count, 40)
        emit(f"    {qtype:<30} {count:>4}  {bar}")

    emit(f"\n  Sample Q&A pairs (first 8):")
    for q, a in deduped[:8]:
        emit(f"\n    Q: {q}")
        emit(f"    A: {a[:180]}{'...' if len(a) > 180 else ''}")

    # Sample multi-turn
    if multiturn_convs:
        emit(f"\n  Sample multi-turn conversation (first):")
        for turn in multiturn_convs[0]:
            role = turn["role"].upper()
            content = turn["content"][:120]
            emit(f"    [{role}] {content}{'...' if len(turn['content']) > 120 else ''}")

    # ── Stage 6: Chunking comparison ────────────────────────────────────
    emit(f"\n[Stage 6] Chunking comparison")
    emit(hr())
    full_text = "\n\n".join(
        (p.get("text") or "").strip()
        for p in after_strip
        if (p.get("text") or "").strip()
    )
    old_chunks = chunk_text(full_text, chunk_size=800, chunk_overlap=150)
    new_chunks = chunk_text_sql_aware(full_text, chunk_size=800, chunk_overlap=150)

    sql_old = sum(1 for c in old_chunks if is_sql_paragraph(c))
    sql_new = sum(1 for c in new_chunks if is_sql_paragraph(c))

    emit(f"  chunk_text (legacy)     : {len(old_chunks)} chunks, {sql_old} contain SQL")
    emit(f"  chunk_text_sql_aware    : {len(new_chunks)} chunks, {sql_new} contain SQL")
    emit(f"  (SQL-aware preserves SQL blocks whole; overlap skips SQL paragraphs)")

    if verbose:
        # Show a SQL-containing chunk if any
        sql_sample = next((c for c in new_chunks if is_sql_paragraph(c)), None)
        if sql_sample:
            emit(f"\n  Sample SQL-containing chunk:")
            emit(f"  {sql_sample[:400]}")

    # ── Summary ──────────────────────────────────────────────────────────
    emit(section_header("SUMMARY"))
    summary = {
        "pdf": pdf_path.name,
        "pages_total": len(all_pages),
        "pages_kept": len(after_strip),
        "pages_filtered_toc": len(toc_pages),
        "pages_filtered_boilerplate": len(boilerplate_pages),
        "pages_filtered_index": len(index_pages),
        "headers_stripped": len(headers),
        "footers_stripped": len(footers),
        "sections": len(sections),
        "tables_extracted": sum(table_counts),
        "qa_pairs_raw": len(all_pairs),
        "qa_pairs_deduped": len(deduped),
        "multiturn_conversations": len(multiturn_convs),
        "chunks_legacy": len(old_chunks),
        "chunks_sql_aware": len(new_chunks),
    }
    col = 32
    for k, v in summary.items():
        emit(f"  {k:<{col}}: {v}")

    emit(f"\n  Run extraction:")
    emit(f"    python -m data.prepare_training_data --pdf-dir sample_data --manual --output-dir training_data")
    emit(f"  Output will be in: training_data/{pdf_path.stem.lower().replace(' ', '_')}/")

    if output_path:
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nReport written to {output_path}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose manual PDF extraction pipeline"
    )
    parser.add_argument("pdf", type=Path, help="PDF file to diagnose")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more detail")
    parser.add_argument("--output", "-o", type=Path, help="Write report to file")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: {args.pdf} does not exist", file=sys.stderr)
        return 1

    diagnose(args.pdf, verbose=args.verbose, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
