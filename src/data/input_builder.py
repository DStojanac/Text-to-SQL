from src.data.schema_serializer import serialize_schema


def build_model_input(question: str, db_id: str) -> str:
    schema_text = serialize_schema(db_id)

    prompt = (
        "Given the database schema below, translate the natural language question "
        "into a valid SQLite SELECT-only SQL query.\n"
        "Rules:\n"
        "- Use ONLY the tables and columns provided in the schema.\n"
        "- Prefer join paths that follow the listed foreign keys.\n"
        "- Never invent new table or column names.\n"
        "- Return only the SQL query, no explanation.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )
    return prompt


if __name__ == "__main__":
    question = "How many heads of the departments are older than 56?"
    db_id = "department_management"
    result = build_model_input(question, db_id)
    print(result)