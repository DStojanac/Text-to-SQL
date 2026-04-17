# Zero-shot prediction using gaussalgo/T5-LM-Large-text2sql-spider.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.data.schema_reader import build_schema_index


# ── Schema formatting for gaussalgo model ─────────────────────────────────

# Map Spider column types to the types gaussalgo used during training
TYPE_MAP = {
    "number": "int",
    "text": "text",
    "time": "text",
    "boolean": "bool",
    "others": "text",
}


def build_gaussalgo_schema(db_id: str, schema_index: Dict) -> str:
    """Build schema string in the format gaussalgo's model expects.

    Format per table:
      "table_name" "col1" type , "col2" type , ... foreign_key: "fk_col" type
      from "ref_table" "ref_col" , ... primary key: "pk_col" [SEP]
    """
    schema = schema_index[db_id]
    col_names = schema["column_names_original"]
    pk_set = schema["primary_keys"]

    # Build FK lookup: for each table index, list of (col_name, col_type, ref_table, ref_col)
    fk_by_table = {}
    for src_idx, tgt_idx in schema["foreign_keys"]:
        src_table_idx, src_col_name = col_names[src_idx]
        tgt_table_idx, tgt_col_name = col_names[tgt_idx]
        src_table = schema["tables"][src_table_idx]
        tgt_table = schema["tables"][tgt_table_idx]

        # Find column type
        src_col_type = "text"
        for col in schema["columns_by_table"][src_table]:
            if col["column_name"] == src_col_name:
                src_col_type = TYPE_MAP.get(col["column_type"], "text")
                break

        fk_by_table.setdefault(src_table, []).append(
            (src_col_name, src_col_type, tgt_table, tgt_col_name)
        )

    table_parts = []
    for table_idx, table_name in enumerate(schema["tables"]):
        columns = schema["columns_by_table"][table_name]

        # Columns that are foreign keys (skip them from regular column list)
        fk_col_names = set()
        if table_name in fk_by_table:
            for fk_src_col, _, _, _ in fk_by_table[table_name]:
                fk_col_names.add(fk_src_col)

        # Regular columns
        col_strs = []
        for col in columns:
            col_type = TYPE_MAP.get(col["column_type"], "text")
            col_strs.append(f'"{col["column_name"]}" {col_type}')

        # Foreign keys
        fk_strs = []
        if table_name in fk_by_table:
            for fk_src_col, fk_type, fk_ref_table, fk_ref_col in fk_by_table[table_name]:
                fk_strs.append(
                    f'"{fk_src_col}" {fk_type} from "{fk_ref_table}" "{fk_ref_col}"'
                )

        # Primary keys for this table
        pk_strs = []
        for col in columns:
            if col["column_index"] in pk_set:
                pk_strs.append(f'"{col["column_name"]}"')

        # Assemble table part
        part = f'"{table_name}" {" , ".join(col_strs)}'
        if fk_strs:
            part += f' , foreign_key: {" , ".join(fk_strs)}'
        else:
            part += " , foreign_key: "
        if pk_strs:
            part += f' primary key: {" ".join(pk_strs)}'

        table_parts.append(part)

    return " [SEP] ".join(table_parts)


def build_gaussalgo_input(question: str, schema_text: str) -> str:
    return f"Question: {question} Schema: {schema_text}"


# ── Prediction logic ──────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str,
                        default="gaussalgo_predictions.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max dev samples to predict (default: all)")
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dev_path = project_root / "data" / "processed" / "spider_dev.jsonl"
    output_path = project_root / "outputs" / "predictions" / args.output_file

    data = load_jsonl(dev_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "gaussalgo/T5-LM-Large-text2sql-spider"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Pre-build schema index once
    schema_index = build_schema_index()

    # Pre-build schema strings (one per db_id, cached)
    schema_cache = {}

    predictions = []
    for i, row in enumerate(data, start=1):
        db_id = row["db_id"]

        if db_id not in schema_cache:
            schema_cache[db_id] = build_gaussalgo_schema(db_id, schema_index)

        input_text = build_gaussalgo_input(row["question"], schema_cache[db_id])

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=args.num_beams,
                early_stopping=True,
            )

        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append({
            "index": i,
            "db_id": db_id,
            "question": row["question"],
            "gold_sql": row["target_sql"],
            "predicted_sql": predicted_sql,
        })

        if i % 10 == 0 or i <= 5:
            print(f"[{i}/{len(data)}] Q: {row['question']}")
            print(f"  Gold: {row['target_sql']}")
            print(f"  Pred: {predicted_sql}")
            print()

    save_json(output_path, predictions)
    print(f"\nSaved {len(predictions)} predictions to: {output_path}")


if __name__ == "__main__":
    main()
