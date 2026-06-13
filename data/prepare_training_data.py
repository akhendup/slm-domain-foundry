#!/usr/bin/env python3
"""
Prepare training data from PDF/CSV into Alpaca and ShareGPT JSONL formats.
Run from project root: python -m data.prepare_training_data [options]
"""

import argparse
import json
import logging
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

_log = logging.getLogger(__name__)

from data.pdf_extractor import PDFExtractor
from data.manual_extractor import (
    extract_manual,
    parse_function_section,
    generate_typed_qa,
    generate_multiturn_conversation,
    deduplicate_qa_pairs,
    _is_heading_line,
    _split_example_parts,
)
from data.domain_config import (
    example_section_label_regexes,
    extract_named_pattern,
    has_structured_content,
)
from data.csv_loader import load_csv
from data.chunking import chunk_text, chunk_text_structured_aware
from data.yaml_pattern_loader import load_patterns_as_qa
from data.template_expander import expand_vocab_dir


def text_to_qa_heuristic(chunks: List[str], source: str = "doc") -> List[Tuple[str, str]]:
    """
    Turn text chunks into Q&A pairs.
    - Detects Input/structured/Output example blocks and generates structured questions.
    - Validates headings before using them (excludes data rows and section labels).
    - Generates domain-specific questions for structured-content chunks.
    - Falls back to a generic summarisation question for everything else.
    """
    qa = []
    short_src = Path(source).stem if source != "doc" else source
    labels = example_section_label_regexes()
    section_markers = "|".join(
        pat.pattern.replace(r"^\s*(?:", "").replace(r")\s*:?\s*$", "")
        for pat in (labels["input"], labels["structured"], labels["output"])
    )

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if re.search(rf"^\s*(?:{section_markers})\s*:?\s*$", chunk, re.I | re.M):
            sub = _split_example_parts(chunk)
            structured_part = sub.get("structured", "")
            if structured_part:
                pattern_name = extract_named_pattern(structured_part)
                if pattern_name:
                    qa.append((
                        f"Show me a complete example of {pattern_name} with input and output.",
                        chunk,
                    ))
                    qa.append((f"Show me a worked example using {pattern_name}.", structured_part))
                else:
                    qa.append((f"Show me a complete structured example from {short_src}.", chunk))
                continue
            elif has_structured_content(chunk):
                pattern_name = extract_named_pattern(chunk)
                q = (f"Show me an example using {pattern_name}." if pattern_name
                     else f"Show me a structured example from {short_src}.")
                qa.append((q, chunk))
                continue

        chunk_parts = chunk.split("\n\n", 1)
        if len(chunk_parts) == 2:
            heading = chunk_parts[0].strip()
            content = chunk_parts[1].strip()
            if len(heading) < 150 and len(content) > 30 and _is_heading_line(heading):
                q = heading if heading.endswith("?") else f"What is {heading}?"
                qa.append((q, content))
                continue

        if has_structured_content(chunk):
            pattern_name = extract_named_pattern(chunk)
            if pattern_name:
                qa.append((f"Show me an example using {pattern_name}.", chunk))
            else:
                qa.append((f"Show me a structured example from {short_src}.", chunk))
            continue

        if len(chunk) > 50:
            q = f"What does the documentation say about the following from {short_src}?"
            qa.append((q, chunk))
    return qa


def build_alpaca_examples(
    qa_pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Build Alpaca-format examples: instruction, input (empty), output."""
    return [
        {"instruction": q, "input": "", "output": a}
        for q, a in qa_pairs
    ]


def build_sharegpt_examples(qa_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Build ShareGPT-format examples: single-turn conversations."""
    return [
        {
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }
        for q, a in qa_pairs
    ]


def save_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _split_train_val(
    items: list, val_ratio: float
) -> Tuple[list, list]:
    n = len(items)
    n_val = max(1, int(n * val_ratio))
    return items[: n - n_val], items[n - n_val :]


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    pre.add_argument("--domain-config", type=Path, default=None, help="Path to domain_config.yaml")
    pre_args, remaining = pre.parse_known_args()

    from train.config import get_section, load_config, resolve_config_path, resolve_path
    from data.domain_config import load_domain_config

    cfg_path = pre_args.config or resolve_config_path()
    cfg = load_config(cfg_path) if cfg_path.exists() else {}

    domain_path = pre_args.domain_config
    if domain_path is None:
        domain_path = resolve_path(cfg, "paths", "domain_config", default="domain_config.yaml")
    load_domain_config(domain_path, reload=True)

    data_cfg = get_section(cfg, "data_prep", default={}) or {}
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    parser = argparse.ArgumentParser(description="Prepare training data from PDF/CSV")
    parser.add_argument(
        "--config",
        type=Path,
        default=cfg_path if cfg_path.exists() else None,
        help="Path to config.yaml (CLI flags override config values)",
    )
    parser.add_argument(
        "--domain-config",
        type=Path,
        default=domain_path,
        help="Path to domain_config.yaml for keyword/pattern extraction",
    )
    parser.add_argument("--pdf-dir", type=Path, help="Directory of PDF files")
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Manual mode: filter TOC/boilerplate/index, extract section-aware Q&A, "
            "and write per-manual to output-dir/<manual_label>/"
        ),
    )
    parser.add_argument("--csv", type=Path, help="CSV file (Q&A or text column)")
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        help=(
            "Directory of YAML pattern files (recursively searched). "
            "Each pattern generates typed Q&A covering description, use cases, "
            "parameters, examples, errors, and best practices."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=resolve_path(cfg, "paths", "training_data", default="training_data"))
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "both"], default="both")
    parser.add_argument("--chunk-size", type=int, default=int(data_cfg.get("chunk_size", 800)))
    parser.add_argument("--chunk-overlap", type=int, default=int(data_cfg.get("chunk_overlap", 150)))
    parser.add_argument("--val-ratio", type=float, default=float(data_cfg.get("val_ratio", 0.15)))
    parser.add_argument("--seed", type=int, default=int(data_cfg.get("seed", 42)))
    parser.add_argument(
        "--no-multiturn",
        action="store_true",
        help="Disable multi-turn conversation generation in manual mode",
    )
    default_vocab = data_cfg.get("vocab_dir")
    default_vocab_path = (
        resolve_path(cfg, "data_prep", "vocab_dir", default=str(default_vocab))
        if default_vocab
        else None
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=default_vocab_path,
        help=(
            "Directory containing *_vocabulary.yaml files (e.g. medical_vocabulary.yaml, "
            "financial_vocabulary.yaml). Each file is combinatorially expanded against "
            "all question templates to produce large Q&A datasets."
        ),
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing conversation_memory interactions.jsonl. "
            "Approved interactions (approved=true) are included in training data."
        ),
    )
    args = parser.parse_args(remaining)

    random.seed(args.seed)
    qa_pairs: List[Tuple[str, str]] = []
    vocab_multiturn: List[Dict] = []

    # From PDFs (standard or manual mode)
    if args.pdf_dir and args.pdf_dir.exists():
        pdf_files = sorted(Path(args.pdf_dir).glob("*.pdf"))
        n_pdfs = len(pdf_files)
        print(
            f"Found {n_pdfs} PDF(s) to process."
            + (
                " [manual mode: section-aware Q&A, per-manual output]"
                if args.manual
                else ""
            ),
            flush=True,
        )
        extractor = PDFExtractor(use_ocr=False)

        for i, pdf_path in enumerate(pdf_files, start=1):
            fname = pdf_path.name

            if args.manual:
                res = extract_manual(
                    pdf_path,
                    extractor,
                    skip_toc=True,
                    skip_boilerplate=True,
                    skip_index=True,
                    use_sections=True,
                )
                full_text = res.get("full_text", "")
                label = res.get("label", pdf_path.stem)
                meta = res.get("metadata", {})
                sections = res.get("sections", [])

                if not full_text:
                    print(
                        f"[{i}/{n_pdfs}] {fname} — no text after filtering, skipping.",
                        flush=True,
                    )
                    continue

                # 1. Section-based typed Q&A (primary)
                pairs: List[Tuple[str, str]] = []
                multiturn: List[Dict] = []
                for s in sections:
                    parsed = parse_function_section(s["text"], s["heading"])
                    pairs.extend(generate_typed_qa(parsed, label))
                    if not args.no_multiturn:
                        conv = generate_multiturn_conversation(parsed)
                        if conv:
                            multiturn.append({"conversations": conv})

                # 2. SQL-aware chunk-based Q&A (supplement for uncaptured content)
                chunks = chunk_text_structured_aware(
                    full_text,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
                pairs.extend(text_to_qa_heuristic(chunks, res.get("source_file", "pdf")))

                # 3. Deduplicate
                pairs = deduplicate_qa_pairs(pairs)

                print(
                    f"[{i}/{n_pdfs}] {fname} — label={label}, "
                    f"pages {meta.get('num_pages_kept', '?')}/{meta.get('num_pages_total', '?')}, "
                    f"sections={meta.get('num_sections', '?')}, "
                    f"{len(pairs)} Q&A pairs, {len(multiturn)} multi-turn convs",
                    flush=True,
                )

                if pairs:
                    manual_out = args.output_dir / label
                    manual_out.mkdir(parents=True, exist_ok=True)
                    random.shuffle(pairs)
                    train_pairs, val_pairs = _split_train_val(pairs, args.val_ratio)

                    if args.format in ("alpaca", "both"):
                        save_jsonl(
                            build_alpaca_examples(train_pairs),
                            manual_out / "train_alpaca.jsonl",
                        )
                        save_jsonl(
                            build_alpaca_examples(val_pairs),
                            manual_out / "val_alpaca.jsonl",
                        )
                    if args.format in ("sharegpt", "both"):
                        save_jsonl(
                            build_sharegpt_examples(train_pairs),
                            manual_out / "train_sharegpt.jsonl",
                        )
                        save_jsonl(
                            build_sharegpt_examples(val_pairs),
                            manual_out / "val_sharegpt.jsonl",
                        )
                        # Multi-turn conversations (ShareGPT only)
                        if multiturn and not args.no_multiturn:
                            random.shuffle(multiturn)
                            train_mt, val_mt = _split_train_val(multiturn, args.val_ratio)
                            save_jsonl(train_mt, manual_out / "train_multiturn.jsonl")
                            save_jsonl(val_mt, manual_out / "val_multiturn.jsonl")

                    print(f"  → wrote {manual_out}", flush=True)
                continue

            # Standard PDF extraction (non-manual mode)
            res = extractor.extract(pdf_path)
            res["source_file"] = fname
            full_text = res.get("full_text", "")
            if not full_text:
                print(f"[{i}/{n_pdfs}] {fname} — no text extracted, skipping.", flush=True)
                continue
            chunks = chunk_text(full_text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            pairs = text_to_qa_heuristic(chunks, fname)
            qa_pairs.extend(pairs)
            print(f"[{i}/{n_pdfs}] {fname} — {len(chunks)} chunks → {len(pairs)} Q&A pairs", flush=True)

    # From CSV
    if args.csv and args.csv.exists():
        print(f"Loading CSV: {args.csv.name}", flush=True)
        texts, csv_qa = load_csv(args.csv)
        qa_pairs.extend(csv_qa)
        if texts and not csv_qa:
            for t in texts:
                chunks = chunk_text(t, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                qa_pairs.extend(text_to_qa_heuristic(chunks, args.csv.name))
        print(f"CSV: {len(csv_qa)} Q&A pairs loaded.", flush=True)

    # From YAML pattern files
    yaml_multiturn: List[Dict] = []
    if args.yaml_dir and args.yaml_dir.exists():
        print(f"Loading YAML patterns from: {args.yaml_dir}", flush=True)
        yaml_qa, yaml_multiturn = load_patterns_as_qa(args.yaml_dir)
        yaml_qa = deduplicate_qa_pairs(yaml_qa)
        qa_pairs.extend(yaml_qa)
        print(
            f"YAML patterns: {len(yaml_qa)} Q&A pairs, "
            f"{len(yaml_multiturn)} multi-turn conversations",
            flush=True,
        )

    # From vocabulary YAML files (combinatorial expansion)
    vocab_multiturn: List[Dict] = []
    if args.vocab_dir and args.vocab_dir.exists():
        print(f"Expanding vocabulary from: {args.vocab_dir}", flush=True)
        vocab_pairs_raw = expand_vocab_dir(args.vocab_dir, multiturn=False)
        vocab_pairs: List[Tuple[str, str]] = [
            (p["question"], p["answer"]) for p in vocab_pairs_raw
        ]
        vocab_pairs = deduplicate_qa_pairs(vocab_pairs)
        qa_pairs.extend(vocab_pairs)
        vocab_multiturn = expand_vocab_dir(args.vocab_dir, multiturn=True)
        print(
            f"Vocabulary expansion: {len(vocab_pairs)} Q&A pairs, "
            f"{len(vocab_multiturn)} multi-turn conversations",
            flush=True,
        )

    # From approved conversation memory
    memory_multiturn: List[Dict] = []
    if args.memory_dir and args.memory_dir.exists():
        mem_jsonl = args.memory_dir / "interactions.jsonl"
        if mem_jsonl.exists():
            import json as _json
            approved_count = 0
            for line in mem_jsonl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    if rec.get("approved") is True:
                        q = (rec.get("question") or "").strip()
                        a = (rec.get("answer") or "").strip()
                        if q and a:
                            qa_pairs.append((q, a))
                            memory_multiturn.append({
                                "conversations": [
                                    {"role": "user",      "content": q},
                                    {"role": "assistant", "content": a},
                                ]
                            })
                            approved_count += 1
                except Exception as exc:
                    _log.warning("Skipping malformed memory record: %s", exc)
            print(f"Conversation memory: {approved_count} approved interactions loaded.", flush=True)

    # Manual mode wrote per-manual output; nothing to aggregate globally
    has_yaml = bool(args.yaml_dir and args.yaml_dir.exists())
    if args.pdf_dir and args.manual and not qa_pairs and not (args.csv and args.csv.exists()) and not has_yaml:
        written = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
        if any(d.is_dir() for d in written):
            print("Manual extraction complete. Output is under output-dir/<manual_label>/", flush=True)
            return 0
        print("No training examples from manuals. Check PDFs and extraction.", flush=True)
        return 1

    if not qa_pairs:
        print("No training examples generated. Provide --pdf-dir and/or --csv.")
        return 1

    print(f"Total: {len(qa_pairs)} Q&A pairs — shuffling and splitting…", flush=True)
    random.shuffle(qa_pairs)
    train_pairs, val_pairs = _split_train_val(qa_pairs, args.val_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("alpaca", "both"):
        train_alpaca = build_alpaca_examples(train_pairs)
        val_alpaca = build_alpaca_examples(val_pairs)
        save_jsonl(train_alpaca, args.output_dir / "train_alpaca.jsonl")
        save_jsonl(val_alpaca, args.output_dir / "val_alpaca.jsonl")
        print(f"Alpaca: train {len(train_alpaca)}, val {len(val_alpaca)}", flush=True)

    if args.format in ("sharegpt", "both"):
        train_sg = build_sharegpt_examples(train_pairs)
        val_sg = build_sharegpt_examples(val_pairs)
        save_jsonl(train_sg, args.output_dir / "train_sharegpt.jsonl")
        save_jsonl(val_sg, args.output_dir / "val_sharegpt.jsonl")
        print(f"ShareGPT: train {len(train_sg)}, val {len(val_sg)}", flush=True)
        # Multi-turn conversations: vocabulary expansion + YAML patterns + approved conversation memory
        all_multiturn = vocab_multiturn + yaml_multiturn + memory_multiturn
        if all_multiturn and not args.no_multiturn:
            random.shuffle(all_multiturn)
            train_mt, val_mt = _split_train_val(all_multiturn, args.val_ratio)
            save_jsonl(train_mt, args.output_dir / "train_multiturn.jsonl")
            save_jsonl(val_mt, args.output_dir / "val_multiturn.jsonl")
            print(
                f"Multi-turn: train {len(train_mt)}, val {len(val_mt)} "
                f"({len(yaml_multiturn)} from patterns, {len(memory_multiturn)} from memory)",
                flush=True,
            )

    print(f"Output directory: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
