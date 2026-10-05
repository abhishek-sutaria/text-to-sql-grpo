"""Dataset loading smoke tests."""

from __future__ import annotations

from pathlib import Path

from text_to_sql_grpo.spider import load_examples_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_load_sample_splits():
    train = load_examples_jsonl(ROOT / "data/sample/train.jsonl")
    dev = load_examples_jsonl(ROOT / "data/sample/dev.jsonl")
    assert len(train) >= 8
    assert len(dev) >= 4
    for ex in train + dev:
        assert ex.question
        assert ex.gold_sql
        assert Path(ex.db_path).exists()
        assert "CREATE TABLE" in ex.schema
