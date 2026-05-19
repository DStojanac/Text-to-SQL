from pathlib import Path

from src.data.schema_serializer import serialize_schema, serialize_pruned_schema
from src.data.schema_linker import prune_schema, prune_schema_dict


def build_model_input(question: str, db_id: str) -> str:
    schema_text = serialize_schema(db_id)

    prompt = (
        f"Translate to SQL: {schema_text}\n"
        f"Q: {question}\n"
        "SQL:"
    )
    return prompt


def build_model_input_with_linking(
    question: str,
    db_id: str,
    max_tables: int = 6,
) -> str:
    """Build model input with schema linking (pruning).

    Instead of dumping the entire database schema into the prompt, this
    first selects only the tables relevant to the question. This dramatically
    reduces input length and solves the truncation problem.

    Args:
        question: Natural language question.
        db_id: Spider database identifier.
        max_tables: Maximum tables to select by relevance score.

    Returns:
        Formatted prompt string with pruned schema.
    """
    pruned = prune_schema(db_id, question, max_tables=max_tables)
    schema_text = serialize_pruned_schema(pruned)

    prompt = (
        f"Translate to SQL: {schema_text}\n"
        f"Q: {question}\n"
        "SQL:"
    )
    return prompt


def build_model_input_with_sqlite(
    question: str,
    db_path: Path,
    max_tables: int = 6,
) -> str:
    """Build model input from any SQLite file (custom databases, API, demo).

    Drop-in replacement for build_model_input_with_linking when there is no
    Spider tables.json — reads schema directly from the .sqlite file.
    """
    from src.data.schema_from_sqlite import schema_from_sqlite

    schema = schema_from_sqlite(db_path)
    pruned = prune_schema_dict(schema, question, max_tables=max_tables)
    schema_text = serialize_pruned_schema(pruned)

    return (
        f"Translate to SQL: {schema_text}\n"
        f"Q: {question}\n"
        "SQL:"
    )