#!/usr/bin/env python3
"""
Export ShareGPT JSONL to a single text file for from-scratch training
(e.g. for NanoGPT or a minimal HF run_clm script).
Each line: "Question: <q> Answer: <a>"
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("training_data/train_sharegpt.jsonl"))
    p.add_argument("--output", type=Path, default=Path("training_data/train_from_scratch.txt"))
    args = p.parse_args()
    if not args.input.exists():
        print(f"Not found: {args.input}")
        return 1
    lines = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            conv = obj.get("conversations", [])
            if len(conv) < 2:
                continue
            user = next((m["content"] for m in conv if m.get("role") == "user"), "")
            assistant = next((m["content"] for m in conv if m.get("role") == "assistant"), "")
            if user and assistant:
                lines.append(f"Question: {user} Answer: {assistant}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(lines)} lines to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
