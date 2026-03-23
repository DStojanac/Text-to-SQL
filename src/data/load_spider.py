import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    project_root = Path(__file__).resolve().parents[2]
    spider_dir = project_root / "data" / "raw" / "spider"

    train_path = spider_dir / "train_spider.json"
    dev_path = spider_dir / "dev.json"

    if not train_path.exists():
        print(f"Missing file: {train_path}")
        return

    if not dev_path.exists():
        print(f"Missing file: {dev_path}")
        return

    train_data = load_json(train_path)
    dev_data = load_json(dev_path)

    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print("-" * 60)

    for i, sample in enumerate(train_data[:5], start=1):
        print(f"Example {i}")
        print(f"DB ID: {sample.get('db_id')}")
        print(f"Question: {sample.get('question')}")
        print(f"SQL: {sample.get('query')}")
        print("-" * 60)


if __name__ == "__main__":
    main()