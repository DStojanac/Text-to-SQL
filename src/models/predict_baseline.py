import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_SAMPLES = 30
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
    project_root = Path(__file__).resolve().parents[2]
    dev_path = project_root / "data" / "processed" / "spider_dev.jsonl"
    output_path = project_root / "outputs" / "predictions" / "baseline_predictions.json"

    data = load_jsonl(dev_path)[:MAX_INPUT_SAMPLES]

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
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

        item = {
            "index": i,
            "db_id": row["db_id"],
            "question": row["question"],
            "input_text": row["input_text"],
            "gold_sql": row["target_sql"],
            "predicted_sql": predicted_sql
        }
        predictions.append(item)

        print("-" * 80)
        print(f"Example {i}")
        print(f"DB ID: {row['db_id']}")
        print(f"Question: {row['question']}")
        print(f"Gold SQL: {row['target_sql']}")
        print(f"Predicted SQL: {predicted_sql}")

    save_json(output_path, predictions)
    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()