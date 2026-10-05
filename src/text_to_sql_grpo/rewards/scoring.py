"""Reward scoring (explicit module path for imports/tests)."""

from text_to_sql_grpo.rewards import (  # noqa: F401
    DEFAULT_WEIGHTS,
    RewardBreakdown,
    reward_fn_from_batch,
    score_prediction,
    sql_is_valid,
    summarize_scores,
)
