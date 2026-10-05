#!/usr/bin/env python3
"""Evaluate base / SFT / GRPO models on execution metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_to_sql_grpo.prompts import build_messages, extract_sql  # noqa: E402
from text_to_sql_grpo.rewards import score_prediction, summarize_scores  # noqa: E402
from text_to_sql_grpo.spider import load_examples_jsonl  # noqa: E402
from text_to_sql_grpo.utils import ensure_dir, load_yaml, save_json, set_seed  # noqa: E402


def load_model(model_name: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        adapter or model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, messages, *, max_new_tokens: int, device: str) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def eval_sql_only(examples) -> dict:
    """Oracle: score gold SQL against itself (sanity / harness check)."""
    scores = [
        score_prediction(ex.gold_sql, db_path=ex.db_path, gold_sql=ex.gold_sql)
        for ex in examples
    ]
    return summarize_scores(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True, help="dev/test JSONL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("outputs/eval.json"))
    parser.add_argument(
        "--mode",
        choices=["model", "oracle", "heuristic"],
        default="model",
        help="oracle=gold self-score; heuristic=no LM (template guess); model=generate",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_yaml(args.config)
        args.model = cfg.get("model", {}).get("name", args.model)
        args.max_new_tokens = int(
            cfg.get("eval", {}).get("max_new_tokens", args.max_new_tokens)
        )

    set_seed(42)
    examples = load_examples_jsonl(args.data)
    if args.max_samples:
        examples = examples[: args.max_samples]

    details = []
    if args.mode == "oracle":
        for ex in examples:
            b = score_prediction(ex.gold_sql, db_path=ex.db_path, gold_sql=ex.gold_sql)
            details.append({"id": ex.question_id, "pred": ex.gold_sql, **b.to_dict()})
    elif args.mode == "heuristic":
        # Cheap smoke path: emit a trivial wrong/right mix without loading a LM
        for ex in examples:
            # Intentionally weak baseline: count(*) on first table name if present
            pred = "SELECT 1"
            if "CREATE TABLE" in ex.schema:
                table = ex.schema.split("CREATE TABLE", 1)[1].split("(", 1)[0].strip()
                pred = f"SELECT count(*) FROM {table}"
            b = score_prediction(pred, db_path=ex.db_path, gold_sql=ex.gold_sql)
            details.append(
                {
                    "id": ex.question_id,
                    "question": ex.question,
                    "pred": extract_sql(pred),
                    **b.to_dict(),
                }
            )
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model(args.model, args.adapter, device)
        for i, ex in enumerate(examples):
            messages = build_messages(ex.db_id, ex.schema, ex.question)
            text = generate_sql(
                model,
                tokenizer,
                messages,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            b = score_prediction(text, db_path=ex.db_path, gold_sql=ex.gold_sql)
            details.append(
                {
                    "id": ex.question_id,
                    "question": ex.question,
                    "pred": b.predicted_sql,
                    "raw": text,
                    **b.to_dict(),
                }
            )
            print(
                f"[{i+1}/{len(examples)}] reward={b.total:.2f} "
                f"correct={b.correctness:.0f} :: {ex.question[:60]}"
            )

    from text_to_sql_grpo.rewards import RewardBreakdown

    scores = [
        RewardBreakdown(
            validity=d["validity"],
            execution=d["execution"],
            correctness=d["correctness"],
            total=d["total"],
            predicted_sql=d.get("predicted_sql") or d.get("pred", ""),
            error=d.get("error"),
        )
        for d in details
    ]
    summary = summarize_scores(scores)
    summary["mode"] = args.mode
    summary["model"] = args.model
    summary["adapter"] = args.adapter
    summary["n_examples"] = len(examples)

    ensure_dir(args.out.parent)
    save_json(args.out, {"summary": summary, "details": details})
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
