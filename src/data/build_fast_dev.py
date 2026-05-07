"""Build a small, deterministic fast-dev subset of the Spider dev split.

It takes the (already linked) Spider dev JSONL, group rows by ``db_id``, and
sample a fixed number of rows so that the per-database distribution is roughly
preserved. Seed and total size are arguments; the defaults are what every
Phase A experiment is expected to use.

Usage
-----
    python -m src.data.build_fast_dev \
        --input_file data/processed/spider_dev_linked_mt10.jsonl \
        --output_file data/processed/spider_dev_fast.jsonl \
        --total 200 \
        --seed 42

The output JSONL keeps the same row schema as the input
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def stratified_sample_by_db(
    rows: List[Dict[str, Any]],
    total: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Stratified sample preserving the per-``db_id`` proportions.

    Each db gets at least 1 row if its share rounds to 0; sizes are rebalanced
    to hit ``total`` exactly. Within a db, picks are random with ``seed``.
    Output order: original input order (stable) so two runs with the same seed
    over the same input produce the same JSONL byte-for-byte.
    """
    rng = random.Random(seed)

    by_db: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        by_db[r["db_id"]].append(idx)

    n_total = len(rows)
    quotas: Dict[str, int] = {}
    for db_id, idxs in by_db.items():
        share = len(idxs) / n_total * total
        quotas[db_id] = max(1, int(round(share)))

    diff = sum(quotas.values()) - total
    if diff != 0:
        ordered_dbs = sorted(by_db.keys(), key=lambda k: -len(by_db[k]))
        i = 0
        step = -1 if diff > 0 else 1
        while diff != 0:
            db_id = ordered_dbs[i % len(ordered_dbs)]
            new_q = quotas[db_id] + step
            if new_q >= 1:
                quotas[db_id] = new_q
                diff += step * -1
            i += 1
            if i > 10_000:
                break

    picked_idxs: set[int] = set()
    for db_id, idxs in by_db.items():
        q = min(quotas[db_id], len(idxs))
        for chosen in rng.sample(idxs, q):
            picked_idxs.add(chosen)

    return [rows[i] for i in sorted(picked_idxs)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="data/processed/spider_dev_linked_mt10.jsonl",
        help="Path to the full dev JSONL (relative to project root).",
    )
    parser.add_argument(
        "--output_file",
        default="data/processed/spider_dev_fast.jsonl",
        help="Path to write the small subset (relative to project root).",
    )
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / args.input_file
    out_path = project_root / args.output_file

    rows = _load_jsonl(in_path)
    if not rows:
        raise RuntimeError(f"No rows found in {in_path}")
    if args.total > len(rows):
        raise ValueError(f"--total {args.total} > number of rows {len(rows)}")

    sampled = stratified_sample_by_db(rows, total=args.total, seed=args.seed)
    _save_jsonl(out_path, sampled)

    db_counts: Dict[str, int] = defaultdict(int)
    for r in sampled:
        db_counts[r["db_id"]] += 1

    print(f"Wrote {len(sampled)} rows to {out_path}")
    print(f"Distinct db_ids represented: {len(db_counts)} (input had {len({r['db_id'] for r in rows})})")
    top = sorted(db_counts.items(), key=lambda kv: -kv[1])[:10]
    print("Top db_ids in subset:")
    for db_id, c in top:
        print(f"  {db_id}: {c}")


if __name__ == "__main__":
    main()
