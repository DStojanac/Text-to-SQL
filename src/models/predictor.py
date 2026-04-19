"""Prediction script with LoRA adapter support and top-K candidates.

Improvements over predict_finetuned.py:
  - Loads LoRA adapter on top of base model (--use_lora flag)
  - No no_repeat_ngram_size (harmful for SQL)
  - Supports generating multiple candidate SQLs per question (--num_candidates)
  - Configurable dev file path (to use linked/unlinked versions)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_model(model_dir: Path, use_lora: bool, base_model: str | None, device: torch.device):
    """Load model, optionally with LoRA adapter on top of base model."""
    if use_lora:
        from peft import PeftModel

        if base_model is None:
            raise ValueError("--base_model is required when using --use_lora")

        print(f"Loading base model: {base_model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        print(f"Loading LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, str(model_dir))
        model = model.merge_and_unload()
    else:
        print(f"Loading model from: {model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model checkpoint or LoRA adapter dir")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output filename (saved in outputs/predictions/)")
    parser.add_argument("--dev_file", type=str, default="data/processed/spider_dev_linked.jsonl",
                        help="Path to dev JSONL file (relative to project root)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Load as LoRA adapter on top of --base_model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="HuggingFace model name for LoRA base (e.g. google/flan-t5-base)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max dev samples to predict (default: all)")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam search width (1=greedy, 4=recommended)")
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of candidate SQLs per question (for reranking)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max tokens to generate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dev_path = project_root / args.dev_file
    model_path = project_root / args.model_dir
    output_path = project_root / "outputs" / "predictions" / args.output_file

    data = load_jsonl(dev_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(model_path, args.use_lora, args.base_model, device)

    predictions = []

    for i, row in enumerate(data, start=1):
        input_text = row["input_text"]

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                early_stopping=True,
            )

        # Decode all candidates
        candidates = [
            tokenizer.decode(seq, skip_special_tokens=True)
            for seq in outputs
        ]

        entry = {
            "index": i,
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row["target_sql"],
            "predicted_sql": candidates[0],
        }
        if args.num_candidates > 1:
            entry["candidates"] = candidates

        predictions.append(entry)

        if i % 50 == 0 or i == 1:
            print(f"[{i}/{len(data)}] {row['question'][:60]}...")
            print(f"  Gold: {row['target_sql'][:80]}")
            print(f"  Pred: {candidates[0][:80]}")

    save_json(output_path, predictions)
    print(f"\nSaved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
