"""Prompt templates for text-to-SQL."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert SQLite SQL generator. "
    "Given a database schema and a natural language question, "
    "respond with a single valid SQLite query only. "
    "Do not include markdown fences, explanations, or comments."
)

USER_TEMPLATE = """Database: {db_id}

Schema:
{schema}

Question: {question}

SQL:"""


def build_messages(db_id: str, schema: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                db_id=db_id, schema=schema, question=question
            ),
        },
    ]


def build_prompt(db_id: str, schema: str, question: str) -> str:
    """Flat prompt for trainers that expect a single string."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        + USER_TEMPLATE.format(db_id=db_id, schema=schema, question=question)
    )


def extract_sql(text: str) -> str:
    """Pull SQL from model output, stripping markdown fences if present."""
    if not text:
        return ""
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        # Prefer fenced body: ```sql ... ``` or ``` ... ```
        for i, part in enumerate(parts):
            if i % 2 == 1:
                body = part.strip()
                if body.lower().startswith("sql"):
                    body = body[3:].lstrip()
                return body.strip().rstrip(";")
        cleaned = parts[0].strip()
    # Heuristic: take from first SELECT/WITH/INSERT/... keyword
    upper = cleaned.upper()
    for kw in ("WITH ", "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE "):
        idx = upper.find(kw)
        if idx != -1:
            return cleaned[idx:].strip().rstrip(";")
    return cleaned.rstrip(";")
