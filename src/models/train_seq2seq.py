import json
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
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


def compute_simple_metrics(eval_preds):
    predictions, labels = eval_preds

    return {}


def main():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "flan_t5_small_debug.yaml"
    config = load_config(config_path)

    train_path = project_root / config["train_file"]
    dev_path = project_root / config["dev_file"]
    output_dir = project_root / config["output_dir"]

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

    # Detect GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        load_best_model_at_end=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=compute_simple_metrics
    )

    trainer.train()

    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final model saved to: {final_model_dir}")
    print(f"Checkpoint directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()