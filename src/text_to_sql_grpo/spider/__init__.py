"""Spider / Spider-like dataset loading."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from text_to_sql_grpo.spider.executor import schema_from_sqlite


@dataclass
class SpiderExample:
    question_id: str
    db_id: str
    question: str
    gold_sql: str
    schema: str
    db_path: str
    split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_db_path(data_root: Path, db_id: str) -> Path:
    """Support both full Spider layout and our sample layout."""
    candidates = [
        data_root / "database" / db_id / f"{db_id}.sqlite",
        data_root / "databases" / db_id / f"{db_id}.sqlite",
        data_root / db_id / f"{db_id}.sqlite",
        data_root / f"{db_id}.sqlite",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find SQLite DB for db_id={db_id!r} under {data_root}"
    )


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "data" / "sample").exists():
            return parent
    return start.parents[2] if len(start.parents) > 2 else start.parent


def _resolve_stored_db_path(raw: str, jsonl_path: Path) -> str:
    """Resolve absolute or repo-relative db_path values."""
    p = Path(raw)
    if p.exists():
        return str(p.resolve())
    repo_root = _find_repo_root(jsonl_path.resolve().parent)
    candidates = [
        Path.cwd() / raw,
        repo_root / raw,
        jsonl_path.parent / raw,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return raw


def load_examples_jsonl(path: Path | str) -> list[SpiderExample]:
    path = Path(path)
    examples: list[SpiderExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "db_path" in obj:
                obj["db_path"] = _resolve_stored_db_path(obj["db_path"], path)
            examples.append(SpiderExample(**obj))
    return examples


def write_examples_jsonl(path: Path | str, examples: list[SpiderExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_spider_split(
    spider_root: Path | str,
    split: str = "train",
    *,
    limit: Optional[int] = None,
    tables_json: Optional[Path | str] = None,
) -> list[SpiderExample]:
    """
    Load a Spider split from the official directory layout.

    Expected:
      spider_root/train_spider.json (or dev.json)
      spider_root/tables.json
      spider_root/database/<db_id>/<db_id>.sqlite
    """
    root = Path(spider_root)
    split_files = {
        "train": root / "train_spider.json",
        "dev": root / "dev.json",
        "train_others": root / "train_others.json",
    }
    if split not in split_files:
        raise ValueError(f"Unknown split: {split}")
    split_path = split_files[split]
    if not split_path.exists():
        raise FileNotFoundError(f"Missing Spider split file: {split_path}")

    tables_path = Path(tables_json) if tables_json else root / "tables.json"
    schema_by_db: dict[str, str] = {}
    if tables_path.exists():
        schema_by_db = _schemas_from_tables_json(tables_path)

    raw = json.loads(split_path.read_text(encoding="utf-8"))
    examples: list[SpiderExample] = []
    for i, row in enumerate(raw):
        if limit is not None and i >= limit:
            break
        db_id = row["db_id"]
        db_path = _resolve_db_path(root, db_id)
        schema = schema_by_db.get(db_id) or schema_from_sqlite(db_path)
        examples.append(
            SpiderExample(
                question_id=f"{split}-{i}",
                db_id=db_id,
                question=row["question"],
                gold_sql=row["query"],
                schema=schema,
                db_path=str(db_path),
                split=split,
            )
        )
    return examples


def _schemas_from_tables_json(tables_path: Path) -> dict[str, str]:
    """Build CREATE-like schema strings from Spider tables.json."""
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in tables:
        db_id = entry["db_id"]
        table_names = entry["table_names_original"]
        col_names = entry["column_names_original"]
        col_types = entry["column_types"]
        by_table: dict[int, list[str]] = {i: [] for i in range(len(table_names))}
        for (t_idx, c_name), c_type in zip(col_names, col_types):
            if t_idx < 0:
                continue
            by_table[t_idx].append(f"{c_name} {c_type}")
        chunks = []
        for t_idx, t_name in enumerate(table_names):
            cols = ", ".join(by_table.get(t_idx, []))
            chunks.append(f"CREATE TABLE {t_name} ({cols});")
        out[db_id] = "\n".join(chunks)
    return out


def iter_batches(
    examples: list[SpiderExample], batch_size: int
) -> Iterator[list[SpiderExample]]:
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]
