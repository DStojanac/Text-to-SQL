import argparse
import json
import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot.errors import ParseError


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sql(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Normalize quote styles: in SQLite both ' and " are valid for string
    text = text.replace('"', "'").replace('`', "'")
    return text.lower()


def canonicalize_sql(text: str) -> str | None:
    try:
        parsed = sqlglot.parse_one(text, read="sqlite")
        return parsed.sql(dialect="sqlite")
    except Exception:
        return None


def exact_match(gold_sql: str, predicted_sql: str) -> bool:
    gold_canonical = canonicalize_sql(gold_sql)
    pred_canonical = canonicalize_sql(predicted_sql)

    if gold_canonical is not None and pred_canonical is not None:
        return normalize_sql(gold_canonical) == normalize_sql(pred_canonical)

    return normalize_sql(gold_sql) == normalize_sql(predicted_sql)


def parses_as_sql(predicted_sql: str) -> bool:
    try:
        sqlglot.parse_one(predicted_sql, read="sqlite")
        return True
    except ParseError:
        return False
    except Exception:
        return False


def categorize_error(gold_sql: str, predicted_sql: str) -> str:
    if exact_match(gold_sql, predicted_sql):
        return "exact_match"
    if not parses_as_sql(predicted_sql):
        return "invalid_sql"
    if "select" not in predicted_sql.lower():
        return "not_sql_like"
    return "valid_but_incorrect"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    predictions_path = project_root / "outputs" / "predictions" / args.predictions_file
    report_path = project_root / "outputs" / "reports" / args.report_file

    predictions = load_json(predictions_path)

    total = len(predictions)
    exact_matches = 0
    parse_success = 0
    error_counts = {}

    evaluated_rows = []

    for row in predictions:
        gold_sql = row["gold_sql"]
        predicted_sql = row["predicted_sql"]

        is_exact = exact_match(gold_sql, predicted_sql)
        is_parseable = parses_as_sql(predicted_sql)
        error_type = categorize_error(gold_sql, predicted_sql)

        if is_exact:
            exact_matches += 1
        if is_parseable:
            parse_success += 1

        error_counts[error_type] = error_counts.get(error_type, 0) + 1

        evaluated_rows.append({
            **row,
            "exact_match": is_exact,
            "parse_success": is_parseable,
            "error_type": error_type
        })

    report = {
        "total_samples": total,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0,
        "parse_success_count": parse_success,
        "parse_success_rate": parse_success / total if total else 0,
        "error_counts": error_counts,
        "examples": evaluated_rows
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Prediction Evaluation Report")
    print("=" * 80)
    print(f"Total samples: {report['total_samples']}")
    print(f"Exact match count: {report['exact_match_count']}")
    print(f"Exact match rate: {report['exact_match_rate']:.2%}")
    print(f"Parse success count: {report['parse_success_count']}")
    print(f"Parse success rate: {report['parse_success_rate']:.2%}")
    print("Error counts:")
    for error_type, count in report["error_counts"].items():
        print(f"  - {error_type}: {count}")

    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()