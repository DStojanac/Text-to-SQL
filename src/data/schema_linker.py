import re
from typing import Any, Dict, List, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.schema_reader import build_schema_index

# Module-level cache so we don't reload tables.json on every call
_schema_index_cache: Dict[str, Dict[str, Any]] | None = None


def _get_schema_index() -> Dict[str, Dict[str, Any]]:
    """Return cached schema index, building it once on first call."""
    global _schema_index_cache
    if _schema_index_cache is None:
        _schema_index_cache = build_schema_index()
    return _schema_index_cache


def _tokenize_name(name: str) -> str:
    """Convert a SQL identifier like 'singer_in_concert' into 'singer in concert'.

    This helps TF-IDF match natural language words to SQL column/table names.
    For example, the question "How many singers..." will match the table
    'singer' and column 'singer_id' after this normalization.
    """
    # Replace underscores and camelCase boundaries with spaces
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _build_table_description(table_name: str, columns: List[Dict]) -> str:
    """Build a text description for a table from its name and columns.

    Example:
        table_name='singer', columns=[{column_name: 'Singer_ID'}, {column_name: 'Name'}, ...]
        -> 'singer singer id name country song name age is male'
    """
    parts = [_tokenize_name(table_name)]
    for col in columns:
        parts.append(_tokenize_name(col["column_name"]))
    return " ".join(parts)


def score_table_relevance(
    question: str,
    schema: Dict[str, Any],
) -> Dict[str, float]:
    """Score each table's relevance to the question using TF-IDF cosine similarity.

    Args:
        question: The natural language question.
        schema: Schema dict from schema_reader (has 'tables', 'columns_by_table').

    Returns:
        Dict mapping table_name -> relevance score (0.0 to 1.0).
    """
    table_names = schema["tables"]
    columns_by_table = schema["columns_by_table"]

    if not table_names:
        return {}

    # Build text descriptions
    descriptions = []
    for tname in table_names:
        desc = _build_table_description(tname, columns_by_table.get(tname, []))
        descriptions.append(desc)

    # TF-IDF: fit on all documents (question + table descriptions)
    # The question is the first document, tables follow
    all_docs = [question.lower()] + descriptions

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Cosine similarity between question (row 0) and each table (rows 1..N)
    question_vec = tfidf_matrix[0:1]
    table_vecs = tfidf_matrix[1:]
    similarities = cosine_similarity(question_vec, table_vecs)[0]

    return {tname: float(sim) for tname, sim in zip(table_names, similarities)}


def _get_fk_neighbor_tables(
    selected_tables: Set[str],
    schema: Dict[str, Any],
) -> Set[str]:
    """Find tables connected to selected tables via foreign keys.

    If table A is selected and has a FK to table B, include B too.
    This ensures the model can generate correct JOIN conditions.
    """
    column_names = schema["column_names_original"]
    table_names = schema["tables"]
    fk_pairs = schema["foreign_keys"]

    neighbors = set()
    for src_col_idx, tgt_col_idx in fk_pairs:
        src_table_idx = column_names[src_col_idx][0]
        tgt_table_idx = column_names[tgt_col_idx][0]

        if src_table_idx < 0 or tgt_table_idx < 0:
            continue

        src_table = table_names[src_table_idx]
        tgt_table = table_names[tgt_table_idx]

        if src_table in selected_tables and tgt_table not in selected_tables:
            neighbors.add(tgt_table)
        elif tgt_table in selected_tables and src_table not in selected_tables:
            neighbors.add(src_table)

    return neighbors


def select_relevant_tables(
    question: str,
    schema: Dict[str, Any],
    max_tables: int = 6,
    include_fk_neighbors: bool = True,
) -> List[str]:
    """Select the most relevant tables for a question.

    Args:
        question: The natural language question.
        schema: Schema dict from schema_reader.
        max_tables: Maximum number of tables to select by relevance score.
        include_fk_neighbors: If True, also include tables connected via FK
            to any selected table (even if they exceed max_tables).

    Returns:
        List of selected table names, in their original schema order.
    """
    table_names = schema["tables"]

    # If the schema is already small, keep everything
    if len(table_names) <= max_tables:
        return list(table_names)

    scores = score_table_relevance(question, schema)

    # Sort by relevance, pick top-K
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = set()
    for tname, _score in ranked[:max_tables]:
        selected.add(tname)

    # Add FK neighbors so JOINs are possible
    if include_fk_neighbors:
        neighbors = _get_fk_neighbor_tables(selected, schema)
        selected.update(neighbors)

    # Return in original schema order (preserves consistency)
    return [t for t in table_names if t in selected]


def prune_schema(
    db_id: str,
    question: str,
    max_tables: int = 6,
) -> Dict[str, Any]:
    """Return a pruned copy of the schema with only relevant tables.

    This is the main entry point. It:
      1. Loads the full schema for db_id
      2. Selects relevant tables using TF-IDF
      3. Returns a new schema dict with only those tables and their columns

    Args:
        db_id: Spider database identifier.
        question: The natural language question.
        max_tables: Max tables to select by relevance (FK neighbors may add more).

    Returns:
        A schema dict with the same structure as schema_reader output, but
        containing only the selected tables, their columns, and relevant FKs.
    """
    schema_index = _get_schema_index()
    if db_id not in schema_index:
        raise ValueError(f"Database '{db_id}' not found in schema index")

    schema = schema_index[db_id]
    selected_tables = select_relevant_tables(question, schema, max_tables)
    selected_set = set(selected_tables)

    # Build pruned columns_by_table
    pruned_columns = {
        tname: schema["columns_by_table"][tname]
        for tname in selected_tables
    }

    # Build pruned foreign keys (only between selected tables)
    column_names = schema["column_names_original"]
    table_names = schema["tables"]
    pruned_fks = []
    for src_col_idx, tgt_col_idx in schema["foreign_keys"]:
        src_table_idx = column_names[src_col_idx][0]
        tgt_table_idx = column_names[tgt_col_idx][0]
        if src_table_idx < 0 or tgt_table_idx < 0:
            continue
        src_table = table_names[src_table_idx]
        tgt_table = table_names[tgt_table_idx]
        if src_table in selected_set and tgt_table in selected_set:
            pruned_fks.append([src_col_idx, tgt_col_idx])

    return {
        "db_id": db_id,
        "tables": selected_tables,
        "columns_by_table": pruned_columns,
        "primary_keys": schema["primary_keys"],
        "foreign_keys": pruned_fks,
        "column_names_original": schema["column_names_original"],
        "original_tables": schema["tables"],
    }
