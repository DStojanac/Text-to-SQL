import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tables_json() -> List[Dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    tables_path = project_root / "data" / "raw" / "spider" / "tables.json"
    return load_json(tables_path)


def build_schema_index() -> Dict[str, Dict[str, Any]]:
    tables_data = load_tables_json()
    schema_index = {}

    for db in tables_data:
        db_id = db["db_id"]

        table_names_original = db["table_names_original"]
        column_names_original = db["column_names_original"]
        column_types = db["column_types"]
        primary_keys = db["primary_keys"]
        foreign_keys = db["foreign_keys"]

        tables = {i: name for i, name in enumerate(table_names_original)}

        columns_by_table = {table_name: [] for table_name in table_names_original}

        for col_idx, (table_idx, column_name) in enumerate(column_names_original):
            if table_idx == -1:
                continue  # skip special "*" entry
            table_name = tables[table_idx]
            columns_by_table[table_name].append({
                "column_name": column_name,
                "column_type": column_types[col_idx],
                "column_index": col_idx
            })

        pk_columns = set(primary_keys)
        fk_pairs = foreign_keys

        schema_index[db_id] = {
            "db_id": db_id,
            "tables": table_names_original,
            "columns_by_table": columns_by_table,
            "primary_keys": pk_columns,
            "foreign_keys": fk_pairs,
            "column_names_original": column_names_original
        }

    return schema_index


def get_schema_for_db(db_id: str) -> Dict[str, Any]:
    schema_index = build_schema_index()
    if db_id not in schema_index:
        raise ValueError(f"Database '{db_id}' not found in Spider tables.json")
    return schema_index[db_id]


if __name__ == "__main__":
    schema = get_schema_for_db("department_management")
    print(schema["db_id"])
    print(schema["tables"])
    print(schema["columns_by_table"])
    print(schema["foreign_keys"])