# Training a Small Language Model From Scratch

**Option (a)** in this project is to train a very small model from scratch instead of fine-tuning an existing one. This is educational: you see how a transformer learns token-by-token from your data.

## What “from scratch” means

- **Vocabulary**: Build a tokenizer (e.g. BPE) on your text or reuse one (e.g. GPT-2 tokenizer).
- **Model**: A small transformer (e.g. 2–4 layers, small hidden size) that predicts the next token.
- **Training**: Standard causal language modeling on your prepared text or Q&A data (e.g. turn Q&A into “Question: … Answer: …” and train to predict the answer part).

## Practical options

1. **Minimal script with Hugging Face**  
   Use `transformers` and `datasets`: tokenize your `training_data/train_sharegpt.jsonl` (or raw text), then train a tiny `GPT2Config` (e.g. 2 layers, 256 hidden size) with `Trainer` and causal LM. This keeps everything in one repo and runs on CPU/GPU.

2. **NanoGPT / minGPT**  
   - [NanoGPT](https://github.com/karpathy/nanoGPT): train a small GPT on a single file (e.g. a concatenated text export of your data).  
   - [minGPT](https://github.com/karpathy/minGPT): similar idea, more didactic.  
   You’d export your PDF/CSV-derived content to a `.txt` (or one JSONL with concatenated “Q: … A: …” lines), then run their training script.

3. **Hugging Face example**  
   The [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) example can train a small GPT-2 from scratch if you pass a custom config and your dataset.

## Suggested flow for this repo

- **Data**: Use the same pipeline as for fine-tuning: PDF/CSV → `python -m data.prepare_training_data` → e.g. `train_sharegpt.jsonl`.
- **Export for from-scratch**: Run `python scripts/export_for_from_scratch.py` to read `train_sharegpt.jsonl` and write `train_from_scratch.txt` with lines like `Question: ... Answer: ...` for NanoGPT/minGPT or a minimal HF script.
- **Training**: Either a minimal `train/train_from_scratch.py` (tiny GPT-2 config + HF Trainer) or instructions to run NanoGPT/minGPT on `train_from_scratch.txt`.

The export script already exists:

```bash
python scripts/export_for_from_scratch.py \
  --input training_data/train_sharegpt.jsonl \
  --output training_data/train_from_scratch.txt
```

If you want a single “from scratch” script inside this repo, the next step is to add `train/train_from_scratch.py` that:
- Loads `training_data/train_sharegpt.jsonl` (or the .txt export),
- Builds or loads a tokenizer,
- Trains a 2–4 layer GPT-2-style model with `Trainer`,
- Saves the model so the same **app** (Q&A chat) can load it and run questions (with a small adapter for the different model type).

This keeps the flow: **data → prepare → train (finetune OR from scratch) → app**.
