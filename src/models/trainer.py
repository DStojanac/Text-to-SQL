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
import transformers
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.data.hf_dataset_loader import load_jsonl_as_dataset


def print_env_info() -> None:
    """Log library versions + TF32 status.

    TF32 is enabled by default on Ampere (A100) for float32 matmul; it silently
    truncates the mantissa to 10 bits. This can be enough to destabilize LoRA
    on T5 where the effective learning signal is already small. We print the
    status so it's visible in every training log.
    """
    try:
        import peft

        peft_version = peft.__version__
    except Exception:
        peft_version = "<not installed>"
    try:
        import accelerate

        accel_version = accelerate.__version__
    except Exception:
        accel_version = "<not installed>"

    print("=" * 60)
    print("[ENV] Library versions")
    print(f"  transformers: {transformers.__version__}")
    print(f"  peft:         {peft_version}")
    print(f"  torch:        {torch.__version__}")
    print(f"  accelerate:   {accel_version}")
    if torch.cuda.is_available():
        print(f"  TF32 matmul:  {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  TF32 cudnn:   {torch.backends.cudnn.allow_tf32}")
    print("=" * 60)


def disable_tf32() -> None:
    """Force pure FP32 matmul / convolutions on Ampere GPUs.

    TF32 is a silent precision downgrade on A100/H100 that does not apply on
    Turing (GTX 1660). Disabling it here makes A100 numerics match our local
    baseline, eliminating one variable when comparing runs across hardware.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception as e:
        print(f"[WARN] Could not disable TF32: {e}")


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
    _call_count = [0]  # mutable counter for closure

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

        # Debug: print sample predictions every eval round
        _call_count[0] += 1
        print(f"\n{'='*60}")
        print(f"[DEBUG] Eval round {_call_count[0]} — {exact}/{total} exact matches")
        for i in range(min(5, total)):
            match = "MATCH" if normalize_sql(decoded_preds[i]) == normalize_sql(decoded_labels[i]) else "MISS"
            print(f"  [{match}] PRED: {decoded_preds[i][:120]}")
            print(f"         GOLD: {decoded_labels[i][:120]}")
        print(f"{'='*60}\n")

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


def diagnose_lora_placement(model) -> None:
    """Count LoRA-trainable params in encoder vs decoder.

    A healthy T5 + LoRA run should have roughly balanced trainable params in
    both blocks. If encoder shows 0, LoRA was attached only to the decoder
    (a silent failure that collapses schema understanding to the base model).
    """
    encoder_params = 0
    decoder_params = 0
    other_params = 0
    lora_module_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_module_names.append(name)
        if ".encoder." in name or name.startswith("encoder.") or "base_model.model.encoder." in name:
            encoder_params += param.numel()
        elif ".decoder." in name or name.startswith("decoder.") or "base_model.model.decoder." in name:
            decoder_params += param.numel()
        else:
            other_params += param.numel()

    print("\n" + "=" * 60)
    print("[DIAG] LoRA placement check")
    print(f"  Encoder trainable params: {encoder_params:>12,}")
    print(f"  Decoder trainable params: {decoder_params:>12,}")
    print(f"  Other trainable params:   {other_params:>12,}")
    print(f"  Total LoRA modules:       {len(lora_module_names)}")
    if encoder_params == 0:
        print("  [WARN] No trainable params in encoder! LoRA likely misconfigured.")
    if decoder_params == 0:
        print("  [WARN] No trainable params in decoder! LoRA likely misconfigured.")
    if lora_module_names:
        print(f"  Sample LoRA modules (first 3): {lora_module_names[:3]}")
    print("=" * 60 + "\n")


def diagnose_target_lengths(dataset, tokenizer, max_target_length: int) -> None:
    """Histogram of target (SQL) token lengths to catch silent truncation.

    If p95 > max_target_length, ~5% of training labels are being truncated
    before the model sees them, which makes those examples unlearnable and
    can stall eval_loss at a high floor.
    """
    # Sample up to 2000 examples to keep this fast on big datasets
    sample_size = min(2000, len(dataset))
    sample = dataset.select(range(sample_size)) if hasattr(dataset, "select") else dataset[:sample_size]
    targets = sample["target_sql"] if hasattr(sample, "__getitem__") else [r["target_sql"] for r in sample]

    lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in targets]
    lengths.sort()
    n = len(lengths)

    def pct(p: float) -> int:
        return lengths[min(n - 1, int(n * p))]

    over_limit = sum(1 for l in lengths if l > max_target_length)
    print("\n" + "=" * 60)
    print(f"[DIAG] Target (SQL) token-length stats over {n} samples")
    print(f"  min / p50 / p90 / p95 / p99 / max:")
    print(f"  {lengths[0]} / {pct(0.5)} / {pct(0.9)} / {pct(0.95)} / {pct(0.99)} / {lengths[-1]}")
    print(f"  max_target_length config: {max_target_length}")
    print(f"  truncated samples (>limit): {over_limit} ({over_limit / n:.1%})")
    if over_limit / n > 0.02:
        print("  [WARN] >2% of targets truncated. Consider raising max_target_length.")
    print("=" * 60 + "\n")


def diagnose_first_example(dataset, tokenizer, max_input_length: int) -> None:
    """Decode the first train example back from token IDs.

    Catches data-upload corruption (common on Colab), tokenizer mismatch,
    and any case where the raw JSONL isn't what we think it is.
    """
    row = dataset[0]
    input_text = row.get("input_text", "<missing>")
    target_sql = row.get("target_sql", "<missing>")

    enc = tokenizer(input_text, max_length=max_input_length, truncation=True)
    decoded_input = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("[DIAG] First training example (round-trip through tokenizer)")
    print(f"  Raw input_text ({len(input_text)} chars):")
    print(f"    {input_text[:300]}{'...' if len(input_text) > 300 else ''}")
    print(f"  Token count: {len(enc['input_ids'])} (max_input_length={max_input_length})")
    print(f"  Decoded back ({len(decoded_input)} chars):")
    print(f"    {decoded_input[:300]}{'...' if len(decoded_input) > 300 else ''}")
    print(f"  Target SQL ({len(target_sql)} chars):")
    print(f"    {target_sql[:300]}{'...' if len(target_sql) > 300 else ''}")
    print("=" * 60 + "\n")


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

    # Reproducibility — set before any model init so that LoRA init is deterministic too
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Kill TF32 on Ampere (A100/H100) unless config explicitly opts in.
    # Default off: matches Turing (GTX 1660) numerics so cross-hardware runs are comparable.
    if not bool(config.get("allow_tf32", False)):
        disable_tf32()

    print_env_info()

    print(f"Config: {config_path}")
    print(f"Model: {config['model_name']}")
    print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

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

    # [DIAG 1] Round-trip the first training example through the tokenizer.
    # Catches data-upload corruption on Colab, tokenizer mismatch, missing fields.
    diagnose_first_example(
        train_dataset, tokenizer, max_input_length=config["max_input_length"]
    )

    # [DIAG 2] Target-length histogram. If p95 > max_target_length, we're silently
    # truncating SQL labels which makes those examples unlearnable and inflates eval_loss.
    diagnose_target_lengths(
        train_dataset, tokenizer, max_target_length=config["max_target_length"]
    )

    # Apply LoRA if configured
    use_lora = config.get("use_lora", False)
    if use_lora:
        model = apply_lora(model, config)
        # [DIAG 3] Confirm LoRA was attached to BOTH encoder and decoder.
        # If encoder params == 0, the model cannot learn to read the schema.
        diagnose_lora_placement(model)
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
        eval_accumulation_steps=config.get("eval_accumulation_steps"),
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
        seed=seed,
        data_seed=seed,
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
