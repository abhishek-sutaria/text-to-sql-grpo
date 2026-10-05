"""SQLite execution helpers for Spider-style databases."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


FORBIDDEN_SQL = re.compile(
    r"\b(ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ALTER|DROP|CREATE|INSERT|UPDATE|DELETE|REPLACE)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: Optional[list[tuple[Any, ...]]] = None
    error: Optional[str] = None
    timed_out: bool = False


def connect_readonly(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    # URI read-only mode prevents accidental writes
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


def is_readonly_select(sql: str) -> bool:
    """Reject non-SELECT mutations / dangerous statements."""
    stripped = sql.strip().rstrip(";")
    if not stripped:
        return False
    if FORBIDDEN_SQL.search(stripped):
        # Allow CREATE only never; SELECT/WITH only
        upper = stripped.lstrip().upper()
        if upper.startswith("SELECT") or upper.startswith("WITH"):
            # Still forbid nested mutation keywords mid-query for safety
            if re.search(
                r"\b(ATTACH|DETACH|PRAGMA|VACUUM|DROP|ALTER|INSERT|UPDATE|DELETE|REPLACE)\b",
                stripped,
                re.IGNORECASE,
            ):
                return False
            return True
        return False
    upper = stripped.lstrip().upper()
    return upper.startswith("SELECT") or upper.startswith("WITH")


def execute_sql(
    db_path: Path | str,
    sql: str,
    *,
    max_rows: int = 500,
    timeout_ms: int = 5000,
) -> ExecResult:
    """Execute a read-only SQL query against a SQLite DB."""
    if not sql or not sql.strip():
        return ExecResult(ok=False, error="empty_sql")
    if not is_readonly_select(sql):
        return ExecResult(ok=False, error="non_readonly_sql")

    try:
        conn = connect_readonly(db_path)
    except FileNotFoundError as exc:
        return ExecResult(ok=False, error=str(exc))

    try:
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_ms)}")
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return ExecResult(ok=True, rows=list(rows))
    except sqlite3.Error as exc:
        return ExecResult(ok=False, error=f"sqlite:{exc}")
    except Exception as exc:  # noqa: BLE001 — surface unexpected exec failures
        return ExecResult(ok=False, error=f"runtime:{exc}")
    finally:
        conn.close()


def normalize_row(row: Sequence[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for cell in row:
        if isinstance(cell, str):
            out.append(cell.strip().lower())
        elif isinstance(cell, float):
            out.append(round(cell, 6))
        else:
            out.append(cell)
    return tuple(out)


def result_sets_equal(
    predicted: Optional[Iterable[Sequence[Any]]],
    gold: Optional[Iterable[Sequence[Any]]],
    *,
    order_sensitive: bool = False,
) -> bool:
    """Compare two SQL result sets (Spider-style bag equality by default)."""
    if predicted is None or gold is None:
        return False
    pred_rows = [normalize_row(r) for r in predicted]
    gold_rows = [normalize_row(r) for r in gold]
    if order_sensitive:
        return pred_rows == gold_rows
    return sorted(pred_rows) == sorted(gold_rows)


def schema_from_sqlite(db_path: Path | str, *, include_samples: bool = False) -> str:
    """Render a compact CREATE-style schema string for prompting."""
    conn = connect_readonly(db_path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        chunks: list[str] = []
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            chunks.append(f"CREATE TABLE {table} ({col_defs});")
            if include_samples:
                sample = conn.execute(f"SELECT * FROM '{table}' LIMIT 3").fetchall()
                if sample:
                    chunks.append(f"-- sample rows for {table}: {sample}")
        return "\n".join(chunks)
    finally:
        conn.close()
