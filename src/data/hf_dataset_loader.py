import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_jsonl_as_dataset(
    path: Path,
    subset_size: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    rows = load_jsonl_rows(path)

    if shuffle:
        random.seed(seed)
        random.shuffle(rows)

    if subset_size is not None:
        rows = rows[:subset_size]

    return Dataset.from_list(rows)