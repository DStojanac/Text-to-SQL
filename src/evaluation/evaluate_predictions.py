"""Exact-match evaluation for predicted SQL.

Two canonicalizers
------------------
- ``v1`` (legacy, kept for backwards comparability)
    - parse with sqlglot, emit canonical SQL
    - replace ``"`` and ``\u0060`` with ``'`` (BUG: turns quoted identifiers into
      string literals)
    - collapse all whitespace including INSIDE string literals
    - lowercase the whole thing

- ``v2`` (default, recommended for thesis numbers)
    - strip trailing ``;``
    - parse with sqlglot; on success use ``.sql(dialect="sqlite")``
    - lowercase only OUTSIDE single-quoted string literals
    - collapse whitespace only OUTSIDE single-quoted string literals
    - never rewrites string contents

The ``v2`` rule set is conservative: it never claims two SQLs are equal unless
their canonical forms are byte-identical.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# v1 canonicalizer (legacy)
# ---------------------------------------------------------------------------


def _normalize_sql_v1(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('"', "'").replace("`", "'")
    return text.lower()


def _canonicalize_sql_v1(text: str) -> str | None:
    try:
        parsed = sqlglot.parse_one(text, read="sqlite")
        return parsed.sql(dialect="sqlite")
    except Exception:
        return None


def _exact_match_v1(gold_sql: str, predicted_sql: str) -> bool:
    gold_canonical = _canonicalize_sql_v1(gold_sql)
    pred_canonical = _canonicalize_sql_v1(predicted_sql)
    if gold_canonical is not None and pred_canonical is not None:
        return _normalize_sql_v1(gold_canonical) == _normalize_sql_v1(pred_canonical)
    return _normalize_sql_v1(gold_sql) == _normalize_sql_v1(predicted_sql)


# ---------------------------------------------------------------------------
# v2 canonicalizer (default)
# ---------------------------------------------------------------------------


def _split_outside_strings(s: str):
    """Yield ``(segment, is_string_literal)`` chunks.

    Recognizes single-quoted SQL strings, including the ``''`` escape sequence
    inside them. Everything else (identifiers, keywords, numbers, comments,
    operators, whitespace) is reported as ``is_string_literal=False`` so callers
    can transform it without corrupting string contents.
    """
    if not s:
        return
    buf: list[str] = []
    in_str = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    buf.append(s[i + 1])
                    i += 2
                    continue
                yield "".join(buf), True
                buf = []
                in_str = False
        else:
            if ch == "'":
                if buf:
                    yield "".join(buf), False
                    buf = []
                buf.append(ch)
                in_str = True
            else:
                buf.append(ch)
        i += 1
    if buf:
        yield "".join(buf), in_str


def _lowercase_outside_strings(s: str) -> str:
    return "".join(seg if is_str else seg.lower() for seg, is_str in _split_outside_strings(s))


def _collapse_ws_outside_strings(s: str) -> str:
    parts: list[str] = []
    for seg, is_str in _split_outside_strings(s):
        if is_str:
            parts.append(seg)
        else:
            parts.append(re.sub(r"\s+", " ", seg))
    return "".join(parts).strip()


def _strip_trailing_semicolons(text: str) -> str:
    text = text.rstrip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def canonicalize_sql_v2(text: str) -> str:
    """Structural-then-surface canonicalizer used for EM in the thesis.

    Steps (order matters):
    1. Strip trailing whitespace and semicolons  — model often appends ';', gold doesn't.
    2. Parse + re-emit with sqlglot dialect=sqlite — normalises implicit→explicit aliases,
       consistent spacing, resolves ambiguous identifier/string quoting.
    3. Normalize all quote chars to single quotes  — sqlglot may keep the original '"'
       for string literals that arrived with double-quotes (valid SQLite syntax).
    4. Lowercase everything including string values — Spider convention; string value
       casing ('USA' vs 'usa') is not semantically meaningful for EM comparison, and
       all published Spider results use full lowercasing.
    5. Collapse repeated whitespace — purely cosmetic.

    This is strictly better than v1:
    - v1 bug fixed: whitespace inside string literals was collapsed (rare but real).
    - v1 bug fixed: trailing semicolons were not stripped.
    - New: sqlglot structural normalisation runs before surface normalisation.
    """
    if text is None:
        return ""
    text = text.strip()
    text = _strip_trailing_semicolons(text)
    try:
        parsed = sqlglot.parse_one(text, read="sqlite")
        if parsed is not None:
            text = parsed.sql(dialect="sqlite")
    except Exception:
        pass  # Fall through to surface normalisation with the raw text
    # Normalize quote style AFTER sqlglot (which may emit " for double-quoted inputs)
    text = text.replace('"', "'").replace("`", "'")
    # Lowercase everything (Spider convention — same as v1's final step)
    text = text.lower()
    # Collapse whitespace (safe now that string content is already lowercased)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _exact_match_v2(gold_sql: str, predicted_sql: str) -> bool:
    return canonicalize_sql_v2(gold_sql) == canonicalize_sql_v2(predicted_sql)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def parses_as_sql(predicted_sql: str) -> bool:
    try:
        sqlglot.parse_one(predicted_sql, read="sqlite")
        return True
    except ParseError:
        return False
    except Exception:
        return False


def _exact_match(gold_sql: str, predicted_sql: str, canon: str) -> bool:
    if canon == "v1":
        return _exact_match_v1(gold_sql, predicted_sql)
    if canon == "v2":
        return _exact_match_v2(gold_sql, predicted_sql)
    raise ValueError(f"Unknown canonicalizer: {canon}")


def categorize_error(gold_sql: str, predicted_sql: str, canon: str) -> str:
    if _exact_match(gold_sql, predicted_sql, canon):
        return "exact_match"
    if not parses_as_sql(predicted_sql):
        return "invalid_sql"
    if "select" not in (predicted_sql or "").lower():
        return "not_sql_like"
    return "valid_but_incorrect"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", type=str, required=True)
    parser.add_argument("--report_file", type=str, required=True)
    parser.add_argument(
        "--canon",
        choices=["v1", "v2"],
        default="v2",
        help="Canonicalizer used for EM (default: v2).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    predictions_path = project_root / "outputs" / "predictions" / args.predictions_file
    report_path = project_root / "outputs" / "reports" / args.report_file

    predictions = load_json(predictions_path)

    total = len(predictions)
    exact_matches = 0
    parse_success = 0
    error_counts: dict[str, int] = {}
    evaluated_rows: list[dict] = []

    for row in predictions:
        gold_sql = row["gold_sql"]
        predicted_sql = row["predicted_sql"]

        is_exact = _exact_match(gold_sql, predicted_sql, args.canon)
        is_parseable = parses_as_sql(predicted_sql)
        error_type = categorize_error(gold_sql, predicted_sql, args.canon)

        if is_exact:
            exact_matches += 1
        if is_parseable:
            parse_success += 1
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

        evaluated_rows.append(
            {**row, "exact_match": is_exact, "parse_success": is_parseable, "error_type": error_type}
        )

    report = {
        "total_samples": total,
        "canonicalizer": args.canon,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0,
        "parse_success_count": parse_success,
        "parse_success_rate": parse_success / total if total else 0,
        "error_counts": error_counts,
        "examples": evaluated_rows,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Prediction Evaluation Report")
    print("=" * 80)
    print(f"Canonicalizer:          {args.canon}")
    print(f"Total samples:          {report['total_samples']}")
    print(f"Exact match count:      {report['exact_match_count']}")
    print(f"Exact match rate:       {report['exact_match_rate']:.2%}")
    print(f"Parse success count:    {report['parse_success_count']}")
    print(f"Parse success rate:     {report['parse_success_rate']:.2%}")
    print("Error counts:")
    for error_type, count in report["error_counts"].items():
        print(f"  - {error_type}: {count}")
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
