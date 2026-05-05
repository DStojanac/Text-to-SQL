"""Spider schema compliance for predicted SQL (sqlglot AST + tables.json index).

Uses the same schema index as linking / ``schema_reader.build_schema_index``.
This is a **static** check: referenced base tables and columns must exist in the
dataset metadata. Qualifiers that do not map to a physical Spider table are skipped to limit false negatives.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


def _norm_ident(name: Any) -> str:
    if name is None:
        return ""
    if isinstance(name, exp.Identifier):
        return str(name.this).strip().lower().strip('"').strip("`")
    s = str(name).strip()
    return s.lower().strip('"').strip("`")


def _allowed_columns_by_table(schema: Dict[str, Any]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for table_name, cols in schema["columns_by_table"].items():
        t = _norm_ident(table_name)
        out[t] = {_norm_ident(c["column_name"]) for c in cols}
    return out


def _collect_cte_names(expr: exp.Expression) -> Set[str]:
    names: Set[str] = set()
    for cte in expr.find_all(exp.CTE):
        if cte.alias:
            names.add(_norm_ident(cte.alias))
    return names


def _collect_alias_map(expr: exp.Expression) -> Dict[str, str]:
    """Map alias or table name -> physical table name (lowercased)."""
    alias_to_physical: Dict[str, str] = {}
    cte_names = _collect_cte_names(expr)
    for table in expr.find_all(exp.Table):
        physical = _norm_ident(table.name)
        if not physical:
            continue
        if isinstance(table.this, exp.Subquery):
            continue
        if physical in cte_names:
            continue
        alias_name = _norm_ident(table.alias) if table.alias else ""
        if alias_name:
            alias_to_physical[alias_name] = physical
        alias_to_physical[physical] = physical
    return alias_to_physical


def _physical_tables_in_query(
    expr: exp.Expression,
    allowed_tables: Set[str],
    cte_names: Set[str],
) -> Set[str]:
    """Physical Spider tables referenced as base tables in the AST."""
    found: Set[str] = set()
    for table in expr.find_all(exp.Table):
        if isinstance(table.this, exp.Subquery):
            continue
        physical = _norm_ident(table.name)
        if physical and physical in allowed_tables and physical not in cte_names:
            found.add(physical)
    return found


def sql_references_valid_for_spider_db(
    sql: str,
    schema: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Return (True, None) if references look consistent with Spider ``schema``.

    On failure returns (False, short reason) for debugging / logging.
    """
    text = (sql or "").strip()
    if not text:
        return False, "empty"

    allowed_tables = {_norm_ident(t) for t in schema["tables"]}
    col_index = _allowed_columns_by_table(schema)

    try:
        parsed = sqlglot.parse_one(text, read="sqlite")
    except SqlglotError as e:
        # ParseError, TokenError (broken string literals, etc.) — treat as not schema-valid.
        return False, f"sqlglot_error:{e}"

    if parsed is None:
        return False, "parse_error:none"

    cte_names = _collect_cte_names(parsed)
    alias_map = _collect_alias_map(parsed)
    physical_present = _physical_tables_in_query(parsed, allowed_tables, cte_names)
    columns_union: Set[str] = set()
    for t in physical_present:
        columns_union |= col_index.get(t, set())

    for table in parsed.find_all(exp.Table):
        if isinstance(table.this, exp.Subquery):
            continue
        physical = _norm_ident(table.name)
        if not physical:
            continue
        if physical in cte_names:
            continue
        if physical not in allowed_tables:
            return False, f"unknown_table:{physical}"

    for col in parsed.find_all(exp.Column):
        cname = _norm_ident(col.name)
        if not cname or cname == "*":
            continue
        qual = _norm_ident(col.table) if col.table else ""

        if qual:
            if qual in cte_names:
                continue
            physical = alias_map.get(qual)
            if physical is None:
                continue
            if physical not in allowed_tables:
                return False, f"unknown_table:{physical}"
            row = col_index.get(physical)
            if row is None or cname not in row:
                return False, f"unknown_column:{physical}.{cname}"
        else:
            if not columns_union:
                continue
            if cname not in columns_union:
                return False, f"unknown_column:{cname}"

    return True, None
