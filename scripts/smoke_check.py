#!/usr/bin/env python3
"""Tiny syntax / import smoke check (no GPU, no model download)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_ast(paths: list[Path]) -> None:
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        print(f"AST OK  {path.relative_to(ROOT)}")


def main() -> int:
    py_files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    py_files.append(ROOT / "app.py")
    check_ast(py_files)

    from text_to_sql_grpo.prompts import extract_sql
    from text_to_sql_grpo.rewards import score_prediction
    from text_to_sql_grpo.spider import load_examples_jsonl

    ex = load_examples_jsonl(ROOT / "data/sample/dev.jsonl")[0]
    b = score_prediction(ex.gold_sql, db_path=ex.db_path, gold_sql=ex.gold_sql)
    assert b.correctness == 1.0, b
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    print("Import + reward smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
