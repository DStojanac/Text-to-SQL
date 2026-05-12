"""QLoRA SFT for decoder-only causal LMs (e.g. Qwen2.5-Coder).

Unlike `trainer.py` (Seq2SeqTrainer / FLAN-T5), this path builds one chat
sequence and trains with loss only on the SQL tail: prompt tokens use label -100.
QLoRA loads 4-bit weights (bitsandbytes), trains LoRA adapters, saves
`lora_adapter/` only — no merged checkpoint.

Keep `SYSTEM_PROMPT` identical to `predictor.py` for train/inference alignment.
Install bitsandbytes on GPU (e.g. Colab): `pip install bitsandbytes>=0.43.0`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.hf_dataset_loader import load_jsonl_as_dataset

# Must match predictor.py — same chat template context at train and inference.
SYSTEM_PROMPT = (
    "You are an expert SQL generator. "
    "Given a natural language question and a database schema, "
    "output ONLY the SQL query — no explanation, no markdown, no extra text."
)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_env_info() -> None:
    """Print library versions for reproducibility."""
    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = "<not installed>"
    try:
        import bitsandbytes as bnb
        bnb_version = bnb.__version__
    except Exception:
        bnb_version = "<not installed — run: pip install bitsandbytes>=0.43.0>"
    try:
        import accelerate
        accel_version = accelerate.__version__
    except Exception:
        accel_version = "<not installed>"

    print("=" * 65)
    print("[ENV] Library versions")
    print(f"  transformers:  {transformers.__version__}")
    print(f"  peft:          {peft_version}")
    print(f"  bitsandbytes:  {bnb_version}")
    print(f"  torch:         {torch.__version__}")
    print(f"  accelerate:    {accel_version}")
    if torch.cuda.is_available():
        print(f"  GPU:           {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 65)


def format_and_tokenize(
    row: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_total_length: int,
) -> Dict[str, List[int]]:
    """One row → chat input_ids / attention_mask / labels (-100 on prompt only)."""

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["input_text"]},
    ]
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_str = prompt_str + row["target_sql"] + tokenizer.eos_token

    full_enc = tokenizer(
        full_str,
        max_length=max_total_length,
        truncation=True,
        add_special_tokens=False,
    )
    prompt_enc = tokenizer(
        prompt_str,
        max_length=max_total_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"]
    prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))

    labels = [-100] * prompt_len + input_ids[prompt_len:]

    assert len(labels) == len(input_ids), (
        f"Label length {len(labels)} != input_ids length {len(input_ids)}"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


class CausalSQLDataset(Dataset):
    """Spider rows as tokenized causal LM examples."""

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_total_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_total_length = max_total_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return format_and_tokenize(
            self.rows[idx], self.tokenizer, self.max_total_length
        )


class PaddingCollator:
    """Left-pad batches; labels -100 on pads. Causal LMs need real tokens at the end."""

    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_masks, batch_labels = [], [], []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append([self.pad_id] * pad_len + f["input_ids"])
            batch_masks.append([0] * pad_len + f["attention_mask"])
            batch_labels.append([-100] * pad_len + f["labels"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_qlora_model(config: Dict[str, Any]) -> AutoModelForCausalLM:
    """4-bit base + k-bit prep + LoRA. Do not call model.to() after device_map load."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\n[MODEL] Loading {config['model_name']} in 4-bit NF4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    use_gc = config.get("gradient_checkpointing", True)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gc
    )
    if use_gc:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def diagnose_first_example(
    dataset: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_total_length: int,
) -> None:
    """Decode first example to verify masking and JSONL fields."""
    row = dataset[0]
    result = format_and_tokenize(row, tokenizer, max_total_length)

    input_ids = result["input_ids"]
    labels = result["labels"]

    n_masked = sum(1 for l in labels if l == -100)
    n_sql = len(labels) - n_masked

    sql_token_ids = [t for t in labels if t != -100]
    decoded_sql = tokenizer.decode(sql_token_ids, skip_special_tokens=True)

    print("\n" + "=" * 65)
    print("[DIAG] First training example — format check")
    print(f"  input_text (first 200 chars):")
    print(f"    {row.get('input_text', '<missing>')[:200]}")
    print(f"  target_sql: {row.get('target_sql', '<missing>')}")
    print(f"  Total tokens: {len(input_ids)} (max_total_length={max_total_length})")
    print(f"  Prompt tokens (masked with -100): {n_masked}")
    print(f"  SQL tokens (trainable): {n_sql}")
    print(f"  Decoded SQL from labels: {decoded_sql}")
    if n_sql == 0:
        print("  [WARN] ALL tokens are masked! The model has nothing to learn.")
        print("         Check that target_sql is not empty in your JSONL file.")
    if n_masked == 0:
        print("  [WARN] No prompt tokens are masked! Check format_and_tokenize().")
    print("=" * 65 + "\n")


def diagnose_sequence_lengths(
    dataset: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_total_length: int,
    sample_size: int = 2000,
) -> None:
    """Length quantiles; truncation cuts the SQL tail."""
    n = min(sample_size, len(dataset))
    lengths = []
    truncated = 0
    for row in dataset[:n]:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input_text"]},
        ]
        prompt_str = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_str = prompt_str + row["target_sql"] + tokenizer.eos_token
        ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if len(ids) > max_total_length:
            truncated += 1

    lengths.sort()

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(n * p))]

    print("\n" + "=" * 65)
    print(f"[DIAG] Sequence length stats (full prompt+SQL, {n} samples)")
    print(f"  min / p50 / p90 / p95 / p99 / max:")
    print(f"  {lengths[0]} / {pct(0.5)} / {pct(0.9)} / {pct(0.95)} / {pct(0.99)} / {lengths[-1]}")
    print(f"  max_total_length config: {max_total_length}")
    print(f"  Truncated sequences:     {truncated} ({truncated / n:.1%})")
    if truncated / n > 0.02:
        print(f"  [WARN] >2% of sequences will be truncated (SQL cut off).")
        print(f"         Consider raising max_input_length in the config YAML.")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen2.5-Coder (causal LM)."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/qwen25_coder_15b_qlora.yaml)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    print_env_info()
    print(f"\nConfig:  {args.config}")
    print(f"Model:   {config['model_name']}")
    print(f"Seed:    {seed}")

    train_path = project_root / config["train_file"]
    dev_path = project_root / config["dev_file"]
    output_dir = project_root / config["output_dir"]

    train_hf = load_jsonl_as_dataset(train_path, subset_size=config.get("train_subset_size"))
    dev_hf = load_jsonl_as_dataset(dev_path, subset_size=config.get("dev_subset_size"))

    train_rows = [train_hf[i] for i in range(len(train_hf))]
    dev_rows = [dev_hf[i] for i in range(len(dev_hf))]

    print(f"Train: {len(train_rows)} rows | Dev: {len(dev_rows)} rows")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] pad_token was not set — using eos_token as pad_token.")

    max_total_length = int(config.get("max_input_length", 1024))

    diagnose_first_example(train_rows, tokenizer, max_total_length)
    diagnose_sequence_lengths(train_rows, tokenizer, max_total_length)

    model = build_qlora_model(config)

    train_dataset = CausalSQLDataset(train_rows, tokenizer, max_total_length)
    dev_dataset = CausalSQLDataset(dev_rows, tokenizer, max_total_length)
    collator = PaddingCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=config.get("eval_strategy", "steps"),
        eval_steps=config.get("eval_steps", 200),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 200),
        save_total_limit=int(config.get("save_total_limit", 3)),
        load_best_model_at_end=bool(config.get("load_best_model_at_end", True)),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(config.get("greater_is_better", False)),
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_ratio=float(config.get("warmup_ratio", 0.03)),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        fp16=False,
        bf16=bool(config.get("bf16", True)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        logging_steps=int(config.get("logging_steps", 25)),
        report_to="none",
        seed=seed,
        dataloader_pin_memory=False,
    )

    patience = int(config.get("early_stopping_patience", 5))
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("\n" + "=" * 65)
    print("[TRAIN] Starting QLoRA fine-tuning ...")
    print(f"  Effective batch size: "
          f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Total epochs: {training_args.num_train_epochs}")
    print(f"  Eval every {training_args.eval_steps} steps")
    print(f"  Early stop patience: {patience} evals")
    print("=" * 65 + "\n")

    trainer.train()

    # Adapter only (no 4-bit merge here); predictor loads base + adapter the same way.
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print("\n" + "=" * 65)
    print(f"[DONE] Training complete.")
    print(f"  LoRA adapter saved to: {adapter_dir}")
    print()
    print("  Next steps — run on Colab:")
    print(f"  python -m src.models.predictor \\")
    print(f"    --model_dir {adapter_dir.relative_to(project_root)} \\")
    print(f"    --base_model {config['model_name']} \\")
    print(f"    --output_file qwen_beam8_predictions.json \\")
    print(f"    --dev_file data/processed/spider_dev_linked_mt10.jsonl \\")
    print(f"    --arch causal --use_lora --load_in_4bit \\")
    print(f"    --num_beams 8 --num_candidates 8")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
