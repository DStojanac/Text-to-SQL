"""Prediction script — seq2seq (T5 family) and causal LMs (Qwen2.5-Coder etc.).

Supports two model architectures via --arch:

  seq2seq (default)
      Standard encoder-decoder models (FLAN-T5, T5-large, etc.).
      Uses AutoModelForSeq2SeqLM.
      Input text goes in as-is; the decoder generates the SQL.
      Output tokens are decoded directly (no prefix to strip).

  causal
      Decoder-only models (Qwen2.5-Coder, Mistral, etc.).
      Uses AutoModelForCausalLM with optional 4-bit quantization.
      Input is wrapped in the model's chat template so the model knows the
      task (system message + user message containing the schema-linked prompt).
      The model generates a continuation; we strip the prompt tokens and
      decode only the newly generated part.

Other features:
  - LoRA adapter on top of base model (--use_lora)
  - Top-K beam candidates per question (--num_candidates) for reranking
  - Configurable max samples for fast smoke tests (--max_samples)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer


# System instruction used for causal models (Qwen chat template).
# Kept here so the thesis can report the exact prompt used.
_CAUSAL_SYSTEM_PROMPT = (
    "You are an expert SQL generator. "
    "Given a natural language question and a database schema, "
    "output ONLY the SQL query — no explanation, no markdown, no extra text."
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading — seq2seq path
# ---------------------------------------------------------------------------

def load_seq2seq_model(
    model_dir: Path,
    use_lora: bool,
    base_model: Optional[str],
    device: torch.device,
):
    """Load a seq2seq model (T5 family), optionally with LoRA adapter."""
    from transformers import AutoModelForSeq2SeqLM

    if use_lora:
        from peft import PeftModel

        if base_model is None:
            raise ValueError("--base_model is required when using --use_lora")

        print(f"[seq2seq] Loading base model: {base_model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        print(f"[seq2seq] Loading LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, str(model_dir))
        model = model.merge_and_unload()
    else:
        print(f"[seq2seq] Loading model from: {model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Model loading — causal path
# ---------------------------------------------------------------------------

def load_causal_model(
    model_dir: Path,
    use_lora: bool,
    base_model: Optional[str],
    device: torch.device,
    load_in_4bit: bool = False,
):
    """Load a causal LM (Qwen-family or similar), optionally with QLoRA.

    Parameters
    ----------
    load_in_4bit:
        When True, loads with bitsandbytes NF4 quantization (QLoRA inference).
        Reduces GPU memory by ~4×.  Requires the ``bitsandbytes`` package.
        Automatically set when the caller passes ``--load_in_4bit``.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if use_lora:
        from peft import PeftModel

        if base_model is None:
            raise ValueError("--base_model is required when using --use_lora")

        print(f"[causal] Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        print(f"[causal] Loading LoRA adapter from: {model_dir}")
        model = PeftModel.from_pretrained(model, str(model_dir))
        if not load_in_4bit:
            model = model.merge_and_unload()
    else:
        print(f"[causal] Loading model from: {model_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            quantization_config=bnb_config,
            device_map="auto" if load_in_4bit else None,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    if not load_in_4bit:
        model.to(device)

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation — seq2seq
# ---------------------------------------------------------------------------

def predict_seq2seq(
    model,
    tokenizer,
    input_text: str,
    device: torch.device,
    num_beams: int,
    num_candidates: int,
    max_length: int,
) -> List[str]:
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=num_beams,
            num_return_sequences=num_candidates,
            early_stopping=True,
        )

    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]


# ---------------------------------------------------------------------------
# Generation — causal
# ---------------------------------------------------------------------------

def predict_causal(
    model,
    tokenizer,
    input_text: str,
    device: torch.device,
    num_beams: int,
    num_candidates: int,
    max_length: int,
) -> List[str]:
    """Run beam-search generation for a causal LM.

    The input_text (schema-linked prompt from build_dataset) is wrapped in
    a chat template so the model understands it as an instruction.  We record
    the number of prompt tokens so we can strip them from the output and return
    only the model-generated SQL.
    """
    messages = [
        {"role": "system", "content": _CAUSAL_SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    # apply_chat_template returns a string with the template-encoded prompt.
    # add_generation_prompt=True appends the assistant turn opener so the model
    # starts generating right away.
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted, return_tensors="pt", truncation=True, max_length=1024
    )
    prompt_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=num_beams,
            num_return_sequences=num_candidates,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the tokens that were generated (strip the prompt prefix).
    candidates = []
    for seq in outputs:
        new_tokens = seq[prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        candidates.append(text)

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run beam-search prediction for seq2seq or causal LMs."
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model checkpoint or LoRA adapter dir")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output filename (saved in outputs/predictions/)")
    parser.add_argument("--dev_file", type=str,
                        default="data/processed/spider_dev_linked.jsonl",
                        help="Path to dev JSONL file (relative to project root)")
    parser.add_argument("--arch", type=str, choices=["seq2seq", "causal"],
                        default="seq2seq",
                        help=(
                            "Model architecture: 'seq2seq' for T5-family "
                            "(AutoModelForSeq2SeqLM), 'causal' for Qwen/decoder-only "
                            "(AutoModelForCausalLM). Default: seq2seq."
                        ))
    parser.add_argument("--use_lora", action="store_true",
                        help="Load as LoRA adapter on top of --base_model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="HuggingFace model name for LoRA base")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help=(
                            "[causal only] Load model in 4-bit (NF4 QLoRA). "
                            "Saves ~4× GPU memory. Requires bitsandbytes."
                        ))
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max dev samples to predict (default: all)")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam search width (1=greedy, 4=recommended)")
    parser.add_argument("--num_candidates", type=int, default=1,
                        help="Number of candidate SQLs per question (for reranking)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max NEW tokens to generate")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dev_path = project_root / args.dev_file
    model_path = project_root / args.model_dir
    output_path = project_root / "outputs" / "predictions" / args.output_file

    data = load_jsonl(dev_path)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  arch: {args.arch}")

    if args.arch == "seq2seq":
        if args.load_in_4bit:
            print("WARNING: --load_in_4bit is only used with --arch causal; ignoring.")
        model, tokenizer = load_seq2seq_model(
            model_path, args.use_lora, args.base_model, device
        )
        predict_fn = lambda text: predict_seq2seq(
            model, tokenizer, text, device, args.num_beams, args.num_candidates, args.max_length
        )
    else:
        model, tokenizer = load_causal_model(
            model_path, args.use_lora, args.base_model, device, args.load_in_4bit
        )
        predict_fn = lambda text: predict_causal(
            model, tokenizer, text, device, args.num_beams, args.num_candidates, args.max_length
        )

    predictions = []

    for i, row in enumerate(data, start=1):
        input_text = row["input_text"]
        candidates = predict_fn(input_text)

        entry = {
            "index": i,
            "db_id": row["db_id"],
            "question": row["question"],
            "gold_sql": row["target_sql"],
            "predicted_sql": candidates[0],
        }
        if args.num_candidates > 1:
            entry["candidates"] = candidates

        predictions.append(entry)

        if i % 50 == 0 or i == 1:
            print(f"[{i}/{len(data)}] {row['question'][:60]}...")
            print(f"  Gold: {row['target_sql'][:80]}")
            print(f"  Pred: {candidates[0][:80]}")

    save_json(output_path, predictions)
    print(f"\nSaved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
