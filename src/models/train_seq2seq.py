import argparse
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.hf_dataset_loader import load_jsonl_as_dataset


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tokenize_function(examples, tokenizer, max_input_length: int, max_target_length: int):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["target_sql"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def normalize_sql(text: str) -> str:
    """Lowercase, collapse whitespace, strip — for comparing SQL strings."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def build_compute_metrics(tokenizer):
    """Returns a compute_metrics function that has access to the tokenizer.

    Why a closure? The HuggingFace Trainer calls compute_metrics(eval_preds)
    with only one argument. But we need the tokenizer to decode token IDs
    back into text. A closure captures the tokenizer in the outer scope.
    """

    def compute_metrics(eval_preds):
        predictions, label_ids = eval_preds

        predictions = np.where(
            predictions != -100, predictions, tokenizer.pad_token_id
        )
        label_ids = np.where(
            label_ids != -100, label_ids, tokenizer.pad_token_id
        )

        # Decode token IDs back into text strings
        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # Count exact matches after normalization
        exact = sum(
            normalize_sql(pred) == normalize_sql(gold)
            for pred, gold in zip(decoded_preds, decoded_labels)
        )
        total = len(decoded_preds)

        return {
            "exact_match": exact / total if total > 0 else 0.0,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Relative path to YAML config file from project root"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / args.config
    config = load_config(config_path)

    train_path = project_root / config["train_file"]
    dev_path = project_root / config["dev_file"]
    output_dir = project_root / config["output_dir"]

    print(f"Using config: {config_path}")
    print(f"Model: {config['model_name']}")

    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = load_jsonl_as_dataset(
        train_path,
        subset_size=config.get("train_subset_size")
    )
    dev_dataset = load_jsonl_as_dataset(
        dev_path,
        subset_size=config.get("dev_subset_size")
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    model.to(device)

    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            config["max_input_length"],
            config["max_target_length"]
        ),
        batched=True
    )

    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_function(
            x,
            tokenizer,
            config["max_input_length"],
            config["max_target_length"]
        ),
        batched=True
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        weight_decay=float(config["weight_decay"]),
        num_train_epochs=float(config["num_train_epochs"]),
        logging_steps=int(config["logging_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        predict_with_generate=bool(config["predict_with_generate"]),
        generation_max_length=int(config["generation_max_length"]),
        fp16=bool(config["fp16"]),
        warmup_ratio=float(config["warmup_ratio"]),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
    )

    # Early stopping: if exact_match doesn't improve for 2 consecutive
    # evaluations (epochs), stop training. This prevents overfitting —

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,  # stop after 2 epochs with no improvement
        early_stopping_threshold=0.01,  # improvement must be > 1% to count
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[early_stopping],
    )

    trainer.train()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"Saved final model to: {final_model_dir}")


if __name__ == "__main__":
    main()