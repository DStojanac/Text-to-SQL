"""SQLite PRAGMA introspection -> schema dict matching ``schema_reader`` / Spider shape."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def schema_from_sqlite(
    db_path: Path,
    db_id: str | None = None,
) -> Dict[str, Any]:
    """Return tables, columns, PKs, FKs for ``db_path`` (read-only). ``db_id`` defaults to file stem."""
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {db_path}")

    resolved_db_id = db_id or db_path.stem

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        schema = _extract_schema(conn, resolved_db_id)
    finally:
        conn.close()

    return schema


def _extract_schema(conn: sqlite3.Connection, db_id: str) -> Dict[str, Any]:
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name"
    )
    table_names: List[str] = [row[0] for row in cursor.fetchall()]

    column_names_original: List[List[Any]] = [[-1, "*"]]
    columns_by_table: Dict[str, List[Dict[str, Any]]] = {}
    primary_keys: Set[int] = set()

    col_lookup: Dict[Tuple[str, str], int] = {}
    global_col_idx = 1

    for table_idx, table_name in enumerate(table_names):
        columns_by_table[table_name] = []

        cursor.execute(f"PRAGMA table_info('{_escape(table_name)}')")
        col_rows = cursor.fetchall()

        for _cid, col_name, col_type, _notnull, _dflt, is_pk in col_rows:
            column_names_original.append([table_idx, col_name])
            col_type_norm = (col_type or "text").lower()

            columns_by_table[table_name].append(
                {
                    "column_name": col_name,
                    "column_type": col_type_norm,
                    "column_index": global_col_idx,
                }
            )

            if is_pk:
                primary_keys.add(global_col_idx)

            col_lookup[(table_name.lower(), col_name.lower())] = global_col_idx
            global_col_idx += 1

    foreign_keys: List[List[int]] = []

    for table_name in table_names:
        cursor.execute(f"PRAGMA foreign_key_list('{_escape(table_name)}')")
        fk_rows = cursor.fetchall()

        for _fk_id, _seq, ref_table, from_col, to_col, *_ in fk_rows:
            src_key = (table_name.lower(), from_col.lower())
            tgt_key = (ref_table.lower(), to_col.lower())

            if src_key in col_lookup and tgt_key in col_lookup:
                foreign_keys.append(
                    [col_lookup[src_key], col_lookup[tgt_key]]
                )

    return {
        "db_id": db_id,
        "tables": table_names,
        "columns_by_table": columns_by_table,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "column_names_original": column_names_original,
    }


def _escape(name: str) -> str:
    return name.replace("'", "''")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", help="Path to the .sqlite file")
    parser.add_argument(
        "--format",
        choices=["summary", "json"],
        default="summary",
    )
    args = parser.parse_args()

    schema = schema_from_sqlite(Path(args.db_path))

    if args.format == "json":
        out = dict(schema)
        out["primary_keys"] = sorted(out["primary_keys"])
        print(json.dumps(out, indent=2))
    else:
        print(f"db_id  : {schema['db_id']}")
        print(f"tables : {schema['tables']}")
        for tname in schema["tables"]:
            cols = schema["columns_by_table"][tname]
            col_strs = [
                f"{c['column_name']} ({c['column_type']})"
                + (" PK" if c["column_index"] in schema["primary_keys"] else "")
                for c in cols
            ]
            print(f"  {tname}: {', '.join(col_strs)}")
        print(f"FKs    : {schema['foreign_keys']}")
