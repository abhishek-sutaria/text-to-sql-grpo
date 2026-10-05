"""Unit tests for SQL extraction, execution, and reward scoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from text_to_sql_grpo.prompts import extract_sql
from text_to_sql_grpo.rewards import score_prediction, sql_is_valid, summarize_scores
from text_to_sql_grpo.spider.executor import (
    execute_sql,
    is_readonly_select,
    result_sets_equal,
    schema_from_sqlite,
)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/sample/databases/concert_singer/concert_singer.sqlite"
PETS = ROOT / "data/sample/databases/pets_1/pets_1.sqlite"


@pytest.fixture(scope="module")
def concert_db() -> Path:
    assert DB.exists(), "sample DB missing — regenerate with project setup"
    return DB


def test_extract_sql_plain():
    assert extract_sql("SELECT count(*) FROM singer") == "SELECT count(*) FROM singer"


def test_extract_sql_fenced():
    text = "Sure!\n```sql\nSELECT Name FROM singer;\n```\n"
    assert extract_sql(text) == "SELECT Name FROM singer"


def test_sql_validity():
    assert sql_is_valid("SELECT 1")
    assert not sql_is_valid("SELEC FROM")
    assert not sql_is_valid("")


def test_readonly_guard():
    assert is_readonly_select("SELECT * FROM singer")
    assert not is_readonly_select("DROP TABLE singer")
    assert not is_readonly_select("DELETE FROM singer")
    assert not is_readonly_select("INSERT INTO singer VALUES (1)")


def test_execute_gold(concert_db: Path):
    res = execute_sql(concert_db, "SELECT count(*) FROM singer")
    assert res.ok
    assert res.rows == [(4,)]


def test_execute_bad_sql(concert_db: Path):
    res = execute_sql(concert_db, "SELECT nope FROM missing_table")
    assert not res.ok
    assert res.error


def test_result_bag_equality():
    assert result_sets_equal([(1, "A"), (2, "B")], [(2, "B"), (1, "A")])
    assert not result_sets_equal([(1,)], [(2,)])


def test_schema_render(concert_db: Path):
    schema = schema_from_sqlite(concert_db)
    assert "CREATE TABLE singer" in schema
    assert "CREATE TABLE stadium" in schema


def test_reward_correct(concert_db: Path):
    gold = "SELECT count(*) FROM singer"
    b = score_prediction(gold, db_path=concert_db, gold_sql=gold)
    assert b.validity == 1.0
    assert b.execution == 1.0
    assert b.correctness == 1.0
    assert b.total == pytest.approx(1.0)


def test_reward_exec_but_wrong(concert_db: Path):
    gold = "SELECT count(*) FROM singer"
    pred = "SELECT count(*) FROM stadium"
    b = score_prediction(pred, db_path=concert_db, gold_sql=gold)
    assert b.validity == 1.0
    assert b.execution == 1.0
    assert b.correctness == 0.0
    assert b.total == pytest.approx(0.5)  # 0.2 + 0.3


def test_reward_invalid(concert_db: Path):
    b = score_prediction("not sql at all", db_path=concert_db, gold_sql="SELECT 1")
    assert b.total == 0.0
    assert b.error == "invalid_sql"


def test_reward_markdown_wrap(concert_db: Path):
    gold = "SELECT Name FROM singer WHERE Country = 'United States'"
    pred = "```sql\nSELECT Name FROM singer WHERE Country = 'United States'\n```"
    b = score_prediction(pred, db_path=concert_db, gold_sql=gold)
    assert b.correctness == 1.0


def test_pets_join_path():
    gold = "SELECT count(*) FROM Pets WHERE PetType = 'cat'"
    b = score_prediction(gold, db_path=PETS, gold_sql=gold)
    assert b.correctness == 1.0


def test_summarize():
    gold = "SELECT count(*) FROM singer"
    scores = [
        score_prediction(gold, db_path=DB, gold_sql=gold),
        score_prediction("SELECT 1", db_path=DB, gold_sql=gold),
    ]
    summary = summarize_scores(scores)
    assert summary["n"] == 2.0
    assert 0.0 < summary["mean_reward"] < 1.0
    assert summary["exec_acc"] == 0.5
