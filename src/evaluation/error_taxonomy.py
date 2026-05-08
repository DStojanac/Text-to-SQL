"""Error taxonomy — classify why predictions fail execution or EM.

What this does
--------------
Reads an execution-eval report (output of ``execute_eval.py``) and puts every
failing row into one of these buckets:

Category            | Meaning
--------------------|------------------------------------------------------------
parse_fail          | Empty or None predicted SQL — model generated nothing.
write_statement     | sqlglot *positively identified* an INSERT/UPDATE/DELETE/DROP.
unparseable_sql     | sqlglot cannot parse the SQL (malformed SELECT, missing op,
                    | unclosed quote, etc.) — NOT a write statement, just broken.
no_such_table       | SQLite error "no such table".
no_such_column      | SQLite error "no such column".
sqlite_error_other  | Other SQLite error (type error, aggregate misuse, etc.).
exec_ok_wrong_rows  | SQL executed but returned different rows/values from gold.
empty_vs_non_empty  | One of {gold, pred} returned empty; the other did not.
gold_fail           | Gold SQL itself failed to execute (data issue, not our fault).

The script prints a summary table and optionally writes a JSON breakdown so
you can drill into individual examples.

Usage
-----
    python -m src.evaluation.error_taxonomy \\
        --exec_report outputs/reports/flan_t5_base_medium_exec_report.json \\
        --output_file outputs/reports/error_taxonomy.json

Optional: add --predictions_file to enrich each row with ``rerank`` metadata
(latency, schema_valid, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlglot
from sqlglot.errors import SqlglotError

from src.evaluation.schema_sql_spider import is_select_only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NO_SUCH_TABLE = re.compile(r"no such table", re.IGNORECASE)
_NO_SUCH_COLUMN = re.compile(r"no such column", re.IGNORECASE)


def _classify_sqlite_error(error_msg: str) -> str:
    """Map a raw SQLite error string to a taxonomy bucket."""
    if not error_msg:
        return "sqlite_error_other"
    if _NO_SUCH_TABLE.search(error_msg):
        return "no_such_table"
    if _NO_SUCH_COLUMN.search(error_msg):
        return "no_such_column"
    return "sqlite_error_other"


def classify_row(row: Dict[str, Any]) -> str:
    """Return the taxonomy label for one evaluated row.

    ``row`` is one element from the ``examples`` list in an exec-eval report.
    """
    predicted_sql: str = (row.get("predicted_sql") or "").strip()
    pred_executed: bool = bool(row.get("pred_executed"))
    results_match: bool = bool(row.get("results_match"))
    gold_executed: bool = bool(row.get("gold_executed"))
    pred_error: Optional[str] = row.get("pred_error")

    # Gold SQL itself broken — not the model's fault
    if not gold_executed:
        return "gold_fail"

    if not predicted_sql:
        return "parse_fail"

    # Try to parse with sqlglot first — this tells us what kind of statement it is.
    try:
        parsed = sqlglot.parse_one(predicted_sql, read="sqlite")
    except SqlglotError:
        # sqlglot cannot parse it at all → likely a broken SELECT (missing operator, unclosed
        # string, etc.) rather than a write statement.  Separate bucket so we don't conflate
        # "model generated a write statement" with "model generated garbled SQL".
        return "unparseable_sql"

    if parsed is None:
        return "unparseable_sql"

    # Positively identified write/DDL statement
    from sqlglot import exp as _exp
    _WRITE = (
        _exp.Insert, _exp.Update, _exp.Delete, _exp.Drop, _exp.Create,
        _exp.Alter, _exp.Command, _exp.Transaction, _exp.Commit, _exp.Rollback,
    )
    if isinstance(parsed, _WRITE):
        return "write_statement"

    # SQLite execution failed
    if not pred_executed:
        return _classify_sqlite_error(str(pred_error or ""))

    # SQL executed but result set differs
    if not results_match:
        gold_result = row.get("gold_result")
        pred_result = row.get("pred_result")
        # Some reports embed the result sets; if not, check for empty-vs-non-empty
        # via the error fields being None when both executed
        if gold_result is not None and pred_result is not None:
            g_empty = len(gold_result) == 0
            p_empty = len(pred_result) == 0
            if g_empty != p_empty:
                return "empty_vs_non_empty"
        return "exec_ok_wrong_rows"

    # Execution matched — this row is correct, should not appear in failures
    return "correct"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify execution-eval failures into error categories."
    )
    parser.add_argument(
        "--exec_report",
        required=True,
        help="Path to exec-eval report JSON (output of execute_eval.py).",
    )
    parser.add_argument(
        "--output_file",
        required=False,
        default=None,
        help=(
            "Optional: filename for per-row taxonomy JSON in outputs/reports/. "
            "If not given, only the summary is printed."
        ),
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=5,
        help="Number of per-category examples to print (default: 5).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    exec_report_path = Path(args.exec_report)
    if not exec_report_path.is_absolute():
        exec_report_path = project_root / exec_report_path

    report = load_json(exec_report_path)
    examples: List[Dict[str, Any]] = report.get("examples", [])

    if not examples:
        print("No examples found in the report. Nothing to classify.")
        return

    taxonomy_rows: List[Dict[str, Any]] = []
    counts: Counter = Counter()

    for row in examples:
        label = classify_row(row)
        counts[label] += 1
        taxonomy_rows.append({
            "label": label,
            "db_id": row.get("db_id"),
            "question": row.get("question"),
            "gold_sql": row.get("gold_sql"),
            "predicted_sql": row.get("predicted_sql"),
            "pred_error": row.get("pred_error"),
            "results_match": row.get("results_match"),
        })

    total = len(examples)
    correct = counts.pop("correct", 0)
    failure_total = total - correct

    # Print summary table
    print()
    print("=" * 70)
    print("Error Taxonomy Summary")
    print("=" * 70)
    print(f"{'Category':<30}  {'Count':>6}  {'% of failures':>14}  {'% of total':>10}")
    print("-" * 70)
    for label, cnt in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct_fail = cnt / failure_total * 100 if failure_total else 0
        pct_total = cnt / total * 100 if total else 0
        print(f"  {label:<28}  {cnt:>6}  {pct_fail:>13.1f}%  {pct_total:>9.1f}%")
    print("-" * 70)
    print(f"  {'TOTAL FAILURES':<28}  {failure_total:>6}  {'100.0':>13}%  "
          f"{failure_total / total * 100:>9.1f}%")
    print(f"  {'correct':<28}  {correct:>6}                    "
          f"{correct / total * 100:>9.1f}%")
    print(f"  {'TOTAL':<28}  {total:>6}")
    print()

    # Print example rows per category
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in taxonomy_rows:
        by_label.setdefault(r["label"], []).append(r)

    for label in sorted(counts.keys(), key=lambda k: -counts[k]):
        bucket = by_label.get(label, [])
        print(f"--- {label} ({len(bucket)} rows) --- first {args.max_examples} examples ---")
        for ex in bucket[: args.max_examples]:
            print(f"  [{ex['db_id']}] {ex['question']}")
            print(f"  Gold: {ex['gold_sql']}")
            print(f"  Pred: {ex['predicted_sql']}")
            if ex.get("pred_error"):
                print(f"  Err : {ex['pred_error']}")
            print()

    # Save detailed output
    if args.output_file:
        out_path = project_root / "outputs" / "reports" / args.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "source_report": str(exec_report_path),
            "total": total,
            "correct": correct,
            "failure_counts": dict(counts),
            "rows": taxonomy_rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved taxonomy to: {out_path}")


if __name__ == "__main__":
    main()
