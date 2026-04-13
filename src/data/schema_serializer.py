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

        fk_lines.append(f"{src_table}.{src_col_name}={tgt_table}.{tgt_col_name}")

    return fk_lines


def serialize_schema(db_id: str) -> str:
    schema = get_schema_for_db(db_id)
    pk_columns = schema["primary_keys"]

    table_parts = []
    for table_name in schema["tables"]:
        col_strs = []
        for col in schema["columns_by_table"][table_name]:
            pk_tag = "*" if col["column_index"] in pk_columns else ""
            col_strs.append(f"{col['column_name']}{pk_tag}")
        table_parts.append(f"{table_name}({', '.join(col_strs)})")

    result = " | ".join(table_parts)

    fk_lines = format_foreign_keys(schema)
    if fk_lines:
        result += " FK: " + ", ".join(fk_lines)

    return result


if __name__ == "__main__":
    text = serialize_schema("department_management")
    print(text)