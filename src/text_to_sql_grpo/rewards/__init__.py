"""Reward scoring for text-to-SQL rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import sqlglot
from sqlglot.errors import ParseError

from text_to_sql_grpo.prompts import extract_sql
from text_to_sql_grpo.spider.executor import (
    ExecResult,
    execute_sql,
    result_sets_equal,
)


@dataclass(frozen=True)
class RewardBreakdown:
    """Component rewards in [0, 1] plus total."""

    validity: float
    execution: float
    correctness: float
    total: float
    predicted_sql: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default weights — correctness dominates; validity/exec give shaping signal.
DEFAULT_WEIGHTS = {
    "validity": 0.2,
    "execution": 0.3,
    "correctness": 0.5,
}


def sql_is_valid(sql: str, dialect: str = "sqlite") -> bool:
    if not sql or not sql.strip():
        return False
    try:
        parsed = sqlglot.parse(sql, read=dialect)
        return bool(parsed) and all(p is not None for p in parsed)
    except ParseError:
        return False
    except Exception:  # noqa: BLE001
        return False


def score_prediction(
    predicted_text: str,
    *,
    db_path: Path | str,
    gold_sql: str,
    weights: Optional[dict[str, float]] = None,
    order_sensitive: bool = False,
) -> RewardBreakdown:
    """
    Score a model completion against a gold SQL on a SQLite DB.

    Rewards:
      - validity: syntactically parseable SQL (sqlglot)
      - execution: runs without error on the DB
      - correctness: result set matches gold (bag equality)
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    sql = extract_sql(predicted_text)

    validity = 1.0 if sql_is_valid(sql) else 0.0
    if validity == 0.0:
        return RewardBreakdown(
            validity=0.0,
            execution=0.0,
            correctness=0.0,
            total=0.0,
            predicted_sql=sql,
            error="invalid_sql",
        )

    pred_exec = execute_sql(db_path, sql)
    execution = 1.0 if pred_exec.ok else 0.0
    if not pred_exec.ok:
        return RewardBreakdown(
            validity=validity,
            execution=0.0,
            correctness=0.0,
            total=w["validity"] * validity,
            predicted_sql=sql,
            error=pred_exec.error,
        )

    gold_exec = execute_sql(db_path, gold_sql)
    if not gold_exec.ok:
        # Gold failure is a data issue; still credit validity+execution
        return RewardBreakdown(
            validity=validity,
            execution=execution,
            correctness=0.0,
            total=w["validity"] * validity + w["execution"] * execution,
            predicted_sql=sql,
            error=f"gold_exec_failed:{gold_exec.error}",
        )

    correctness = (
        1.0
        if result_sets_equal(
            pred_exec.rows, gold_exec.rows, order_sensitive=order_sensitive
        )
        else 0.0
    )
    total = (
        w["validity"] * validity
        + w["execution"] * execution
        + w["correctness"] * correctness
    )
    return RewardBreakdown(
        validity=validity,
        execution=execution,
        correctness=correctness,
        total=total,
        predicted_sql=sql,
        error=None,
    )


def reward_fn_from_batch(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    db_paths: Sequence[str],
    gold_sqls: Sequence[str],
    weights: Optional[dict[str, float]] = None,
) -> list[float]:
    """TRL-compatible reward: one float per completion."""
    rewards: list[float] = []
    for completion, db_path, gold_sql in zip(completions, db_paths, gold_sqls):
        # Completions may arrive as chat message lists from some TRL versions
        text = completion
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        breakdown = score_prediction(
            str(text), db_path=db_path, gold_sql=gold_sql, weights=weights
        )
        rewards.append(breakdown.total)
    return rewards


def summarize_scores(scores: Sequence[RewardBreakdown]) -> dict[str, float]:
    n = max(len(scores), 1)
    return {
        "n": float(len(scores)),
        "mean_reward": sum(s.total for s in scores) / n,
        "exec_acc": sum(s.correctness for s in scores) / n,
        "exec_success": sum(s.execution for s in scores) / n,
        "validity_rate": sum(s.validity for s in scores) / n,
    }
