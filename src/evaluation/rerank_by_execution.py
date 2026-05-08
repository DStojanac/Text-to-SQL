"""Execution-guided reranking over beam candidates.

---------------
The model's top-beam SQL is often wrong while a lower-ranked beam is right.
This script takes a predictions JSON whose rows carry a `candidates` list
(top-K beam outputs, typically K=8) and picks ONE SQL per row by running the
candidates on the real SQLite database.

Modes
-----
- ``first_executable`` (realistic, no gold used):
    Try candidates in beam order; pick the first one that runs without error.
    If none runs, keep the original top-1.

- ``first_executable_schema_first`` (realistic, no gold used):
    Same as above, but prefer the first candidate that both passes a static
    Spider schema check (sqlglot + ``tables.json`` index) **and** executes.
    If none qualify, fall back to the first executable candidate (same as
    ``first_executable``).

- ``majority_executable`` (realistic, no gold used):
    Run all candidates. Among those that execute successfully, pick the SQL
    whose *result set* was returned by the most candidates (majority vote).
    Ties are broken by beam rank (lower = preferred).  If none execute, fall
    back to top-1.  This is an ensemble-style picker that targets EX directly.

- ``oracle`` (upper-bound, uses gold result):
    Among candidates that run, pick the first whose result set equals the
    gold query's result set.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.schema_reader import build_schema_index
from src.evaluation.execute_eval import (
    execute_sql,
    get_db_path,
    normalize_sql_for_exec,
)
from src.evaluation.schema_sql_spider import (
    is_select_only,
    sql_references_valid_for_spider_db,
)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def rerank_row(
    row: Dict[str, Any],
    project_root: Path,
    mode: str,
    schema_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Pick a SQL for one prediction row.

    Returns:
        (updated_row, stats_dict, elapsed_ms)
    elapsed_ms is the wall-clock time spent executing candidates for this row.
    """
    t_start = time.perf_counter()

    db_id = row["db_id"]
    gold_sql = normalize_sql_for_exec(row.get("gold_sql", ""))
    top1 = row.get("predicted_sql", "")
    candidates: List[str] = row.get("candidates") or [top1]

    # Safety: discard any candidate that is not a SELECT statement.
    # This protects evaluation DBs (and later, production DBs) from writes.
    candidates = [
        normalize_sql_for_exec(c)
        for c in candidates
        if c is not None and is_select_only(c)
    ]
    if not candidates:
        # If all candidates were filtered out (e.g. all DDL/DML), keep top1
        # as a last resort but mark it clearly.
        top1_norm = normalize_sql_for_exec(top1)
        candidates = [top1_norm]

    db_path = get_db_path(db_id, project_root)

    gold_ok, gold_result = execute_sql(db_path, gold_sql) if gold_sql else (False, None)

    chosen_sql = top1
    chosen_rank = 0
    any_executable = False
    picked_reason = "fallback_top1"
    schema_valid_chosen: Optional[bool] = None

    if mode == "first_executable":
        for rank, cand in enumerate(candidates):
            ok, _ = execute_sql(db_path, cand)
            if ok:
                any_executable = True
                chosen_sql = cand
                chosen_rank = rank
                picked_reason = "first_executable"
                break

    elif mode == "first_executable_schema_first":
        if schema_index is None:
            raise ValueError(
                "schema_index is required for mode first_executable_schema_first"
            )
        schema = schema_index[db_id]
        chosen_done = False
        for rank, cand in enumerate(candidates):
            schema_ok, _ = sql_references_valid_for_spider_db(cand, schema)
            if not schema_ok:
                continue
            ok, _ = execute_sql(db_path, cand)
            if ok:
                any_executable = True
                chosen_sql = cand
                chosen_rank = rank
                picked_reason = "schema_first_executable"
                schema_valid_chosen = True
                chosen_done = True
                break
        if not chosen_done:
            for rank, cand in enumerate(candidates):
                ok, _ = execute_sql(db_path, cand)
                if ok:
                    any_executable = True
                    chosen_sql = cand
                    chosen_rank = rank
                    picked_reason = "first_executable_schema_fallback_exec"
                    s_ok, _ = sql_references_valid_for_spider_db(cand, schema)
                    schema_valid_chosen = s_ok
                    chosen_done = True
                    break
        if not chosen_done:
            picked_reason = "fallback_top1"
            schema_valid_chosen = False

    elif mode == "majority_executable":
        # Run all candidates once; vote on result sets; pick the most common result.
        # Ties go to the lowest beam rank (most confident beam).
        vote_counts: Counter = Counter()
        result_to_best_rank: Dict[Any, int] = {}  # result_key → lowest rank that produced it
        result_to_sql: Dict[Any, str] = {}         # result_key → SQL for that rank

        for rank, cand in enumerate(candidates):
            ok, result = execute_sql(db_path, cand)
            if not ok:
                continue
            any_executable = True
            # Make the result hashable: execute_sql returns a set of tuples.
            result_key = frozenset(result) if isinstance(result, set) else result
            vote_counts[result_key] += 1
            if result_key not in result_to_best_rank:
                result_to_best_rank[result_key] = rank
                result_to_sql[result_key] = cand

        if result_to_best_rank:
            # Pick the result with the most votes; tie → lowest beam rank (most confident)
            best_key = max(
                vote_counts.keys(),
                key=lambda k: (vote_counts[k], -result_to_best_rank[k]),
            )
            chosen_sql = result_to_sql[best_key]
            chosen_rank = result_to_best_rank[best_key]
            picked_reason = "majority_executable"

    elif mode == "oracle":
        first_exec_idx = None
        first_match_idx = None
        for rank, cand in enumerate(candidates):
            ok, result = execute_sql(db_path, cand)
            if ok:
                any_executable = True
                if first_exec_idx is None:
                    first_exec_idx = rank
                if gold_ok and result == gold_result and first_match_idx is None:
                    first_match_idx = rank
                    break
        if first_match_idx is not None:
            chosen_sql = candidates[first_match_idx]
            chosen_rank = first_match_idx
            picked_reason = "oracle_match"
        elif first_exec_idx is not None:
            chosen_sql = candidates[first_exec_idx]
            chosen_rank = first_exec_idx
            picked_reason = "first_executable_no_match"
        else:
            picked_reason = "fallback_top1"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    out_row = dict(row)
    out_row["predicted_sql"] = chosen_sql
    rerank_meta: Dict[str, Any] = {
        "mode": mode,
        "chosen_rank": chosen_rank,
        "num_candidates": len(candidates),
        "any_executable": any_executable,
        "picked_reason": picked_reason,
        "original_top1": top1,
        "changed": chosen_sql.strip() != top1.strip(),
        "elapsed_ms": round(elapsed_ms, 2),
    }
    if schema_valid_chosen is not None:
        rerank_meta["schema_valid_chosen"] = schema_valid_chosen
    out_row["rerank"] = rerank_meta

    stats = {
        "any_executable": any_executable,
        "changed": out_row["rerank"]["changed"],
        "picked_reason": picked_reason,
        "elapsed_ms": elapsed_ms,
    }
    return out_row, stats, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_file",
        required=True,
        help="Filename in outputs/predictions/ (must contain a 'candidates' list per row).",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Filename to save in outputs/predictions/ (new reranked predictions).",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "first_executable",
            "first_executable_schema_first",
            "majority_executable",
            "oracle",
        ],
        default="first_executable",
        help="Rerank rule (default: first_executable).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: only process the first N rows (useful for a quick smoke test).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    schema_index: Optional[Dict[str, Dict[str, Any]]] = None
    if args.mode == "first_executable_schema_first":
        schema_index = build_schema_index()
    predictions_path = project_root / "outputs" / "predictions" / args.predictions_file
    output_path = project_root / "outputs" / "predictions" / args.output_file

    rows = load_json(predictions_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    out_rows: List[Dict[str, Any]] = []
    any_exec = 0
    changed = 0
    reasons: Dict[str, int] = {}
    elapsed_ms_list: List[float] = []

    for i, row in enumerate(rows, start=1):
        out_row, stats, elapsed_ms = rerank_row(
            row, project_root, args.mode, schema_index=schema_index
        )
        out_rows.append(out_row)
        elapsed_ms_list.append(elapsed_ms)

        if stats["any_executable"]:
            any_exec += 1
        if stats["changed"]:
            changed += 1
        reasons[stats["picked_reason"]] = reasons.get(stats["picked_reason"], 0) + 1

        if i % 100 == 0 or i == 1:
            print(f"[{i}/{len(rows)}] reranked; changed so far: {changed}")

    save_json(output_path, out_rows)

    total = len(out_rows)
    elapsed_ms_list.sort()
    p50 = elapsed_ms_list[int(total * 0.50)] if total else 0
    p95 = elapsed_ms_list[int(total * 0.95)] if total else 0
    p99 = elapsed_ms_list[int(total * 0.99)] if total else 0
    avg = sum(elapsed_ms_list) / total if total else 0

    print()
    print("=" * 80)
    print("Rerank summary")
    print("=" * 80)
    print(f"Mode:                          {args.mode}")
    print(f"Total rows:                    {total}")
    print(f"Had >= 1 executable candidate: {any_exec}/{total} "
          f"({any_exec / total:.2%})" if total else "N/A")
    print(f"Rows where chosen != top-1:    {changed}/{total} "
          f"({changed / total:.2%})" if total else "N/A")
    print("Pick reason counts:")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print(f"Latency per row (ms):  avg={avg:.1f}  p50={p50:.1f}  p95={p95:.1f}  p99={p99:.1f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
