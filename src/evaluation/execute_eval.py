"""Execution-based evaluation for Text-to-SQL predictions.

Why this matters:
    Exact string match is too strict. Two SQL queries can look completely
    different but return identical results:

        Gold:     SELECT name FROM t WHERE a=1 AND b=2
        Predict:  SELECT name FROM t WHERE b=2 AND a=1

    These fail exact match but are semantically identical. Execution-based
    evaluation runs both queries on the actual SQLite database and compares
    results. This is a fairer, more meaningful metric.

    Spider provides .sqlite databases in data/raw/spider/database/{db_id}/

Usage:
    python -m src.evaluation.execute_eval \
        --predictions_file flan_t5_base_medium_predictions.json \
        --report_file flan_t5_base_medium_exec_report.json
"""

import argparse
import concurrent.futures
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

# Maximum rows fetched from any single SQL query.
MAX_RESULT_ROWS: int = 10_000


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_db_path(db_id: str, project_root: Path) -> Path:
    """Find the .sqlite file for a given database ID."""
    return project_root / "data" / "raw" / "spider" / "database" / db_id / f"{db_id}.sqlite"


def _run_query(db_path: Path, sql: str) -> Tuple[bool, Any]:
    """Execute a query inside a worker thread (called by execute_sql).

    Opens a fresh read-only connection, runs the query, and returns
    a (success, result_set) pair.  Exceptions propagate naturally to
    the ThreadPoolExecutor future so the caller can handle them.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(MAX_RESULT_ROWS)
        return True, set(tuple(row) for row in rows)
    finally:
        conn.close()


def execute_sql(db_path: Path, sql: str, timeout: int = 5) -> Tuple[bool, Any]:
    """Execute a SQL query on a SQLite database and return results.

    Args:
        db_path:  Path to the .sqlite file.
        sql:      SQL query string.
        timeout:  Wall-clock seconds before the query is abandoned.
                 Implemented via a background thread so that
                  runaway queries (e.g. accidental cartesian products) do
                  not hang the caller indefinitely.

    Returns:
        (success: bool, result: set of tuples or error string)

    Result rows are returned as a set of tuples so comparison is
    order-independent (Spider EX metric ignores row order).
    At most MAX_RESULT_ROWS rows are returned; the cap is high enough
    that it never affects Spider evaluation.
    """
    if not db_path.exists():
        return False, f"Database file not found: {db_path}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_query, db_path, sql)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False, f"Query timed out after {timeout}s"
        except sqlite3.OperationalError as e:
            return False, f"SQL error: {e}"
        except sqlite3.Warning as e:
            return False, f"SQL warning: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"


def normalize_sql_for_exec(text: str) -> str:
    """Light cleanup before execution — strip whitespace, remove trailing semicolons."""
    text = text.strip()
    text = re.sub(r";\s*$", "", text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="Filename in outputs/predictions/")
    parser.add_argument("--report_file", type=str, required=True,
                        help="Filename to save in outputs/reports/")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    predictions_path = project_root / "outputs" / "predictions" / args.predictions_file
    report_path = project_root / "outputs" / "reports" / args.report_file

    predictions = load_json(predictions_path)
    total = len(predictions)

    # Counters
    exec_match = 0          # Both execute and return same results
    gold_exec_success = 0   # Gold SQL executed successfully
    pred_exec_success = 0   # Predicted SQL executed successfully
    both_exec_success = 0   # Both executed successfully

    # Detailed results for each example
    evaluated_rows = []

    for row in predictions:
        db_id = row["db_id"]
        gold_sql = normalize_sql_for_exec(row["gold_sql"])
        pred_sql = normalize_sql_for_exec(row["predicted_sql"])
        db_path = get_db_path(db_id, project_root)

        # Execute both gold and predicted SQL
        gold_ok, gold_result = execute_sql(db_path, gold_sql)
        pred_ok, pred_result = execute_sql(db_path, pred_sql)

        if gold_ok:
            gold_exec_success += 1
        if pred_ok:
            pred_exec_success += 1

        # Compare results only if both executed successfully
        if gold_ok and pred_ok:
            both_exec_success += 1
            # Results match if both queries return the same set of rows
            results_match = (gold_result == pred_result)
            if results_match:
                exec_match += 1
        else:
            results_match = False

        evaluated_rows.append({
            "index": row.get("index"),
            "db_id": db_id,
            "question": row["question"],
            "gold_sql": row["gold_sql"],
            "predicted_sql": row["predicted_sql"],
            "gold_executed": gold_ok,
            "pred_executed": pred_ok,
            "results_match": results_match,
            "gold_error": gold_result if not gold_ok else None,
            "pred_error": pred_result if not pred_ok else None,
        })

    report = {
        "total_samples": total,
        "execution_accuracy": exec_match / total if total else 0,
        "execution_match_count": exec_match,
        "gold_exec_success": gold_exec_success,
        "pred_exec_success": pred_exec_success,
        "both_exec_success": both_exec_success,
        "pred_exec_rate": pred_exec_success / total if total else 0,
        "examples": evaluated_rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 80)
    print("Execution-Based Evaluation Report")
    print("=" * 80)
    print(f"Total samples:            {total}")
    print(f"Execution accuracy:       {exec_match}/{total} = {exec_match / total:.2%}" if total else "N/A")
    print(f"Pred execution success:   {pred_exec_success}/{total} = {pred_exec_success / total:.2%}" if total else "N/A")
    print(f"Gold execution success:   {gold_exec_success}/{total}")
    print(f"Both executed:            {both_exec_success}/{total}")
    print()

    # Show a few failure examples for debugging
    failures = [r for r in evaluated_rows if not r["results_match"] and r["pred_executed"]]
    if failures:
        print(f"Showing first 5 failures where predicted SQL executed but results differ:")
        print("-" * 80)
        for row in failures[:5]:
            print(f"  [{row['db_id']}] {row['question']}")
            print(f"  Gold: {row['gold_sql']}")
            print(f"  Pred: {row['predicted_sql']}")
            print()

    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
