import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.input_builder import build_model_input


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_split(split_name: str, input_filename: str, output_filename: str) -> None:
    project_root = Path(__file__).resolve().parents[2]
    spider_dir = project_root / "data" / "raw" / "spider"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = spider_dir / input_filename
    output_path = processed_dir / output_filename

    data = load_json(input_path)

    processed_rows = []

    for sample in data:
        question = sample["question"]
        query = sample["query"]
        db_id = sample["db_id"]

        input_text = build_model_input(question=question, db_id=db_id)

        processed_rows.append({
            "db_id": db_id,
            "question": question,
            "target_sql": query,
            "input_text": input_text
        })

    save_jsonl(output_path, processed_rows)
    print(f"Saved {len(processed_rows)} rows to {output_path}")


def main():
    process_split("train", "train_spider.json", "spider_train.jsonl")
    process_split("dev", "dev.json", "spider_dev.jsonl")


if __name__ == "__main__":
    main()