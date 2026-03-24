from typing import Any, Dict, List

from src.data.schema_reader import get_schema_for_db


def format_foreign_keys(schema: Dict[str, Any]) -> List[str]:
    column_names_original = schema["column_names_original"]
    fk_pairs = schema["foreign_keys"]

    fk_lines = []
    for source_col_idx, target_col_idx in fk_pairs:
        src_table_idx, src_col_name = column_names_original[source_col_idx]
        tgt_table_idx, tgt_col_name = column_names_original[target_col_idx]

        src_table = schema["tables"][src_table_idx]
        tgt_table = schema["tables"][tgt_table_idx]

        fk_lines.append(f"{src_table}.{src_col_name} -> {tgt_table}.{tgt_col_name}")

    return fk_lines


def serialize_schema(db_id: str) -> str:
    schema = get_schema_for_db(db_id)

    lines = []
    lines.append("Database schema:")

    for table_name in schema["tables"]:
        lines.append(f"table: {table_name}")
        column_entries = schema["columns_by_table"][table_name]
        col_text = ", ".join([c["column_name"] for c in column_entries])
        lines.append(f"columns: {col_text}")
        lines.append("")

    fk_lines = format_foreign_keys(schema)
    if fk_lines:
        lines.append("foreign keys:")
        for fk in fk_lines:
            lines.append(fk)

    return "\n".join(lines).strip()


if __name__ == "__main__":
    text = serialize_schema("department_management")
    print(text)