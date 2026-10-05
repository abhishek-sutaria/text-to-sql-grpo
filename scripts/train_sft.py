#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for text-to-SQL (Qwen2.5-Instruct)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_to_sql_grpo.prompts import build_messages  # noqa: E402
from text_to_sql_grpo.spider import load_examples_jsonl  # noqa: E402
from text_to_sql_grpo.utils import ensure_dir, load_yaml, set_seed  # noqa: E402


def examples_to_hf_dataset(examples):
    from datasets import Dataset

    rows = []
    for ex in examples:
        messages = build_messages(ex.db_id, ex.schema, ex.question)
        messages.append({"role": "assistant", "content": ex.gold_sql})
        rows.append({"messages": messages, "db_id": ex.db_id})
    return Dataset.from_list(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override train JSONL path from config",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_path = Path(args.data or cfg["data"]["train_file"])
    examples = load_examples_jsonl(data_path)
    if cfg["data"].get("max_samples"):
        examples = examples[: int(cfg["data"]["max_samples"])]
    print(f"Loaded {len(examples)} SFT examples from {data_path}")

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_name = cfg["model"]["name"]
    out_dir = ensure_dir(cfg["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if cfg["model"].get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if quant is not None and torch.cuda.is_available():
        model_kwargs["quantization_config"] = quant
        model_kwargs["device_map"] = "auto"
    elif not torch.cuda.is_available():
        print("WARNING: No CUDA — loading model in CPU float32 (smoke only).")
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    lora = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"].get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    train_ds = examples_to_hf_dataset(examples)
    train_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["train"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["train"].get("grad_accum", 1)),
        learning_rate=float(cfg["train"]["lr"]),
        num_train_epochs=float(cfg["train"].get("epochs", 1)),
        max_steps=int(cfg["train"].get("max_steps", -1)),
        logging_steps=int(cfg["train"].get("logging_steps", 1)),
        save_steps=int(cfg["train"].get("save_steps", 50)),
        bf16=bool(cfg["train"].get("bf16", False)) and torch.cuda.is_available(),
        fp16=bool(cfg["train"].get("fp16", False)) and torch.cuda.is_available(),
        max_seq_length=int(cfg["train"].get("max_seq_length", 1024)),
        report_to=cfg["train"].get("report_to", "none"),
        packing=False,
        dataset_text_field=None,
        seed=int(cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    print(f"Saved adapter → {out_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
