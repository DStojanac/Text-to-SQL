from src.data.schema_serializer import serialize_schema


def build_model_input(question: str, db_id: str) -> str:
    schema_text = serialize_schema(db_id)

    prompt = (
        f"Translate to SQL: {schema_text}\n"
        f"Q: {question}\n"
        "SQL:"
    )
    return prompt


if __name__ == "__main__":
    question = "How many heads of the departments are older than 56?"
    db_id = "department_management"
    result = build_model_input(question, db_id)
    print(result)