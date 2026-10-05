#!/usr/bin/env python3
"""Prepare train/dev JSONL from Spider or the built-in sample set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_to_sql_grpo.spider import (  # noqa: E402
    load_examples_jsonl,
    load_spider_split,
    write_examples_jsonl,
)
from text_to_sql_grpo.utils import ensure_dir, maybe_limit  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=["sample", "spider"],
        default="sample",
        help="sample = bundled mini DBs; spider = full Spider layout",
    )
    p.add_argument("--spider-root", type=Path, default=Path("data/spider"))
    p.add_argument("--sample-root", type=Path, default=Path("data/sample"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--dev-limit", type=int, default=None)
    args = p.parse_args()

    out = ensure_dir(args.out_dir)

    if args.source == "sample":
        train = load_examples_jsonl(args.sample_root / "train.jsonl")
        # Rewrite db_path to be relative-stable from CWD
        for ex in train:
            ex.db_path = str(
                (args.sample_root / "databases" / ex.db_id / f"{ex.db_id}.sqlite").resolve()
            )
        dev = load_examples_jsonl(args.sample_root / "dev.jsonl")
        for ex in dev:
            ex.db_path = str(
                (args.sample_root / "databases" / ex.db_id / f"{ex.db_id}.sqlite").resolve()
            )
    else:
        train = load_spider_split(args.spider_root, "train", limit=args.train_limit)
        # Optionally merge train_others
        others_path = args.spider_root / "train_others.json"
        if others_path.exists() and args.train_limit is None:
            train += load_spider_split(args.spider_root, "train_others")
        dev = load_spider_split(args.spider_root, "dev", limit=args.dev_limit)

    train = maybe_limit(train, args.train_limit)
    dev = maybe_limit(dev, args.dev_limit)

    write_examples_jsonl(out / "train.jsonl", train)
    write_examples_jsonl(out / "dev.jsonl", dev)
    print(f"Wrote {len(train)} train + {len(dev)} dev → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
