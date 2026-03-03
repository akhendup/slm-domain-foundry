#!/usr/bin/env python3
"""
Prepare training data from PDF/CSV into Alpaca and ShareGPT JSONL formats.
Run from project root: python -m data.prepare_training_data [options]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data.pdf_extractor import PDFExtractor
from data.csv_loader import load_csv
from data.chunking import chunk_text


def text_to_qa_heuristic(chunks: List[str], source: str = "doc") -> List[Tuple[str, str]]:
    """
    Turn text chunks into simple Q&A pairs using headings as questions.
    For better quality, use an LLM in a separate step (not included here).
    """
    qa = []
    for chunk in chunks:
        parts = chunk.strip().split("\n\n")
        if len(parts) < 2:
            continue
        heading = parts[0].strip()
        content = "\n\n".join(parts[1:]).strip()
        if len(heading) < 150 and len(content) > 30:
            q = heading if heading.endswith("?") else f"What is {heading}?"
            qa.append((q, content))
    return qa


def build_alpaca_examples(
    qa_pairs: List[Tuple[str, str]],
    instructions_only: bool = False,
) -> List[Dict[str, Any]]:
    """Build Alpaca-format examples: instruction, input (optional), output."""
    examples = []
    for q, a in qa_pairs:
        examples.append({
            "instruction": q,
            "input": "" if instructions_only else "",
            "output": a,
        })
    return examples


def build_sharegpt_examples(qa_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Build ShareGPT-format examples: conversations with user/assistant turns."""
    examples = []
    for q, a in qa_pairs:
        examples.append({
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        })
    return examples


def save_jsonl(items: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from PDF/CSV")
    parser.add_argument("--pdf-dir", type=Path, help="Directory of PDF files")
    parser.add_argument("--csv", type=Path, help="CSV file (Q&A or text column)")
    parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--format", choices=["alpaca", "sharegpt", "both"], default="both")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    qa_pairs: List[Tuple[str, str]] = []

    # From PDFs
    if args.pdf_dir and args.pdf_dir.exists():
        extractor = PDFExtractor(use_ocr=False)
        for res in extractor.extract_directory(args.pdf_dir):
            full_text = res.get("full_text", "")
            if not full_text:
                continue
            chunks = chunk_text(full_text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
            qa_pairs.extend(text_to_qa_heuristic(chunks, res.get("source_file", "pdf")))

    # From CSV
    if args.csv and args.csv.exists():
        texts, csv_qa = load_csv(args.csv)
        qa_pairs.extend(csv_qa)
        if texts and not csv_qa:
            for t in texts:
                chunks = chunk_text(t, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
                qa_pairs.extend(text_to_qa_heuristic(chunks, args.csv.name))

    if not qa_pairs:
        print("No training examples generated. Provide --pdf-dir and/or --csv.")
        return 1

    random.shuffle(qa_pairs)
    n = len(qa_pairs)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    train_pairs = qa_pairs[:n_train]
    val_pairs = qa_pairs[n_train:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.format in ("alpaca", "both"):
        train_alpaca = build_alpaca_examples(train_pairs)
        val_alpaca = build_alpaca_examples(val_pairs)
        save_jsonl(train_alpaca, args.output_dir / "train_alpaca.jsonl")
        save_jsonl(val_alpaca, args.output_dir / "val_alpaca.jsonl")
        print(f"Alpaca: train {len(train_alpaca)}, val {len(val_alpaca)}")

    if args.format in ("sharegpt", "both"):
        train_sg = build_sharegpt_examples(train_pairs)
        val_sg = build_sharegpt_examples(val_pairs)
        save_jsonl(train_sg, args.output_dir / "train_sharegpt.jsonl")
        save_jsonl(val_sg, args.output_dir / "val_sharegpt.jsonl")
        print(f"ShareGPT: train {len(train_sg)}, val {len(val_sg)}")

    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
