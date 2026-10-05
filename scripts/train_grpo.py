#!/usr/bin/env python3
"""GRPO post-training with execution-based rewards (TRL GRPOTrainer)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_to_sql_grpo.prompts import build_prompt, extract_sql  # noqa: E402
from text_to_sql_grpo.rewards import score_prediction  # noqa: E402
from text_to_sql_grpo.spider import load_examples_jsonl  # noqa: E402
from text_to_sql_grpo.utils import ensure_dir, load_yaml, set_seed  # noqa: E402


def build_dataset(examples):
    from datasets import Dataset

    rows = []
    for ex in examples:
        rows.append(
            {
                "prompt": build_prompt(ex.db_id, ex.schema, ex.question),
                "db_path": ex.db_path,
                "gold_sql": ex.gold_sql,
                "question": ex.question,
                "db_id": ex.db_id,
            }
        )
    return Dataset.from_list(rows)


def make_reward_fn(weights: dict[str, float]):
    """
    TRL GRPO reward signature varies slightly by version.
    We accept prompts/completions plus optional kwargs carrying columns.
    """

    def reward_func(completions, **kwargs):
        db_paths = kwargs.get("db_path")
        gold_sqls = kwargs.get("gold_sql")
        # Some TRL versions nest completion strings differently
        scores = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)
            db_path = db_paths[i] if db_paths is not None else kwargs.get("db_paths", [None])[i]
            gold = gold_sqls[i] if gold_sqls is not None else kwargs.get("gold_sqls", [""])[i]
            breakdown = score_prediction(
                text, db_path=db_path, gold_sql=gold, weights=weights
            )
            scores.append(breakdown.total)
        return scores

    reward_func.__name__ = "sql_execution_reward"
    return reward_func


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Optional SFT adapter to continue from",
    )
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_path = Path(args.data or cfg["data"]["train_file"])
    examples = load_examples_jsonl(data_path)
    if cfg["data"].get("max_samples"):
        examples = examples[: int(cfg["data"]["max_samples"])]
    print(f"Loaded {len(examples)} GRPO prompts from {data_path}")

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    model_name = cfg["model"]["name"]
    out_dir = ensure_dir(cfg["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = None
    if cfg["model"].get("load_in_4bit", True) and torch.cuda.is_available():
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {"trust_remote_code": True, "torch_dtype": "auto"}
    if quant is not None:
        model_kwargs["quantization_config"] = quant
        model_kwargs["device_map"] = "auto"
    else:
        print("WARNING: Loading without 4-bit (CPU/smoke).")
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    adapter = args.sft_adapter or cfg.get("sft_adapter")
    if adapter:
        print(f"Loading SFT adapter from {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg["lora"].get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

    ds = build_dataset(examples)
    reward_weights = cfg.get(
        "reward_weights",
        {"validity": 0.2, "execution": 0.3, "correctness": 0.5},
    )

    grpo_args = GRPOConfig(
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
        report_to=cfg["train"].get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
        num_generations=int(cfg["train"].get("num_generations", 4)),
        max_completion_length=int(cfg["train"].get("max_completion_length", 256)),
        max_prompt_length=int(cfg["train"].get("max_prompt_length", 1024)),
        temperature=float(cfg["train"].get("temperature", 0.7)),
    )

    trainer_kwargs = dict(
        model=model,
        reward_funcs=make_reward_fn(reward_weights),
        args=grpo_args,
        train_dataset=ds,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    # processing_class vs tokenizer depends on TRL version
    try:
        trainer = GRPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = GRPOTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    save_path = out_dir / "adapter"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved GRPO adapter → {save_path}")
    # Keep extract_sql import referenced for debugging notebooks
    _ = extract_sql
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
