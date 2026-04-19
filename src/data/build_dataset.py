import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.data.input_builder import build_model_input, build_model_input_with_linking


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_token_stats(texts: List[str], split_name: str) -> None:
    """Print token length statistics using whitespace tokenization as a proxy.

    Real tokenizer lengths will differ slightly, but whitespace tokens
    correlate well enough to verify that schema linking reduces input size.
    """
    lengths = [len(t.split()) for t in texts]
    lengths.sort()
    n = len(lengths)
    print(f"\n--- {split_name} input token stats (whitespace) ---")
    print(f"  Count:  {n}")
    print(f"  Min:    {lengths[0]}")
    print(f"  Median: {lengths[n // 2]}")
    print(f"  P90:    {lengths[int(n * 0.9)]}")
    print(f"  P95:    {lengths[int(n * 0.95)]}")
    print(f"  Max:    {lengths[-1]}")
    over_512 = sum(1 for l in lengths if l > 512)
    print(f"  >512:   {over_512} ({over_512 / n * 100:.1f}%)")


def process_split(
    split_name: str,
    input_filename: str,
    output_filename: str,
    use_schema_linking: bool = False,
    max_tables: int = 6,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    spider_dir = project_root / "data" / "raw" / "spider"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = spider_dir / input_filename
    output_path = processed_dir / output_filename

    data = load_json(input_path)

    processed_rows = []
    all_inputs = []

    for sample in data:
        question = sample["question"]
        query = sample["query"]
        db_id = sample["db_id"]

        if use_schema_linking:
            input_text = build_model_input_with_linking(
                question=question, db_id=db_id, max_tables=max_tables
            )
        else:
            input_text = build_model_input(question=question, db_id=db_id)

        processed_rows.append({
            "db_id": db_id,
            "question": question,
            "target_sql": query,
            "input_text": input_text
        })
        all_inputs.append(input_text)

    save_jsonl(output_path, processed_rows)
    print(f"Saved {len(processed_rows)} rows to {output_path}")
    print_token_stats(all_inputs, split_name)


def main():
    parser = argparse.ArgumentParser(
        description="Build preprocessed JSONL datasets from Spider."
    )
    parser.add_argument(
        "--use_schema_linking",
        action="store_true",
        help="Enable schema linking (prune schema to relevant tables only)"
    )
    parser.add_argument(
        "--max_tables",
        type=int,
        default=6,
        help="Max tables to select by relevance when using schema linking (default: 6)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Output file suffix, e.g. '_linked' -> spider_train_linked.jsonl"
    )
    args = parser.parse_args()

    suffix = args.suffix
    if args.use_schema_linking and not suffix:
        suffix = "_linked"

    mode_str = "WITH schema linking" if args.use_schema_linking else "WITHOUT schema linking"
    print(f"Building dataset {mode_str} (max_tables={args.max_tables})")

    process_split(
        "train", "train_spider.json", f"spider_train{suffix}.jsonl",
        use_schema_linking=args.use_schema_linking,
        max_tables=args.max_tables,
    )
    process_split(
        "dev", "dev.json", f"spider_dev{suffix}.jsonl",
        use_schema_linking=args.use_schema_linking,
        max_tables=args.max_tables,
    )


if __name__ == "__main__":
    main()