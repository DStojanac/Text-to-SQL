"""Training script with LoRA (PEFT) support and early stopping.

This is the v2 trainer that replaces train_seq2seq.py.
Key improvements:
  - LoRA via PEFT: trains ~1% of parameters instead of all of them
  - Early stopping: stops if eval metric doesn't improve for 3 evals
  - Saves both LoRA adapter (~7MB) and merged model (~1GB)
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
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
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        predictions, label_ids = eval_preds

        predictions = np.where(
            predictions != -100, predictions, tokenizer.pad_token_id
        )
        label_ids = np.where(
            label_ids != -100, label_ids, tokenizer.pad_token_id
        )

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        exact = sum(
            normalize_sql(pred) == normalize_sql(gold)
            for pred, gold in zip(decoded_preds, decoded_labels)
        )
        total = len(decoded_preds)

        return {
            "exact_match": exact / total if total > 0 else 0.0,
        }

    return compute_metrics


def apply_lora(model, config: Dict[str, Any]):
    """Wrap model with LoRA adapters using PEFT.

    Only the adapter parameters (~1-2% of total) will be trained.
    Everything else is frozen. This dramatically reduces memory usage
    and training time while maintaining quality.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("lora_target_modules", ["q", "v"]),
        lora_dropout=config.get("lora_dropout", 0.1),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="Relative path to YAML config file from project root",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / args.config
    config = load_config(config_path)

    train_path = project_root / config["train_file"]
    dev_path = project_root / config["dev_file"]
    output_dir = project_root / config["output_dir"]

    print(f"Config: {config_path}")
    print(f"Model: {config['model_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_dataset = load_jsonl_as_dataset(
        train_path, subset_size=config.get("train_subset_size")
    )
    dev_dataset = load_jsonl_as_dataset(
        dev_path, subset_size=config.get("dev_subset_size")
    )
    print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    # Apply LoRA if configured
    use_lora = config.get("use_lora", False)
    if use_lora:
        model = apply_lora(model, config)
    else:
        print("LoRA disabled — full fine-tuning mode")

    model.to(device)

    # Tokenize
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, config["max_input_length"], config["max_target_length"]
        ),
        batched=True,
    )
    tokenized_dev = dev_dataset.map(
        lambda x: tokenize_function(
            x, tokenizer, config["max_input_length"], config["max_target_length"]
        ),
        batched=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy=config.get("eval_strategy", "epoch"),
        eval_steps=config.get("eval_steps"),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps"),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
        num_train_epochs=float(config["num_train_epochs"]),
        logging_steps=int(config["logging_steps"]),
        save_total_limit=int(config.get("save_total_limit", 3)),
        predict_with_generate=bool(config.get("predict_with_generate", True)),
        generation_max_length=int(config.get("generation_max_length", 256)),
        fp16=bool(config.get("fp16", False)),
        bf16=bool(config.get("bf16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "exact_match"),
        greater_is_better=config.get("greater_is_better", True),
    )

    # Callbacks
    callbacks = []
    patience = int(config.get("early_stopping_patience", 3))
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=callbacks,
    )

    trainer.train()

    # Save
    if use_lora:
        # Save lightweight adapter (~7MB)
        adapter_dir = output_dir / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"Saved LoRA adapter to: {adapter_dir}")

        # Save full merged model (for easy inference without PEFT)
        merged_dir = output_dir / "merged_model"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Saved merged model to: {merged_dir}")
    else:
        final_dir = output_dir / "final_model"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Saved model to: {final_dir}")


if __name__ == "__main__":
    main()
