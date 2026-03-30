import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MAX_NEW_TOKENS = 128


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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dev_path = project_root / "data" / "processed" / "spider_dev.jsonl"
    model_path = project_root / args.model_dir
    output_path = project_root / "outputs" / "predictions" / args.output_file

    data = load_jsonl(dev_path)[:args.max_samples]

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    predictions = []

    for i, row in enumerate(data, start=1):
        input_text = row["input_text"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS
            )

        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append({
            "index": i,
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row["target_sql"],
            "predicted_sql": predicted_sql
        })
        
        print("-" * 80)
        print(f"Example {i}")
        print(f"Question: {row['question']}")
        print(f"Gold SQL: {row['target_sql']}")
        print(f"Predicted SQL: {predicted_sql}")

    save_json(output_path, predictions)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()