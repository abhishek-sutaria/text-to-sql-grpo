"""
Gradio demo for Text-to-SQL (HF Spaces–ready).

Free-track default: rule/heuristic + optional local transformers model.
Set env MODEL_ID (and optionally ADAPTER_ID) to load Qwen2.5-3B-Instruct.
On CPU Spaces without a GPU, generation may be slow — sample DBs still work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import spaces

from text_to_sql_grpo.prompts import build_messages, extract_sql
from text_to_sql_grpo.rewards import score_prediction
from text_to_sql_grpo.spider.executor import execute_sql, schema_from_sqlite

SAMPLE_DBS = {
    "concert_singer": ROOT / "data/sample/databases/concert_singer/concert_singer.sqlite",
    "pets_1": ROOT / "data/sample/databases/pets_1/pets_1.sqlite",
}

EXAMPLES = {
    "concert_singer": [
        "How many singers do we have?",
        "Which singers are from the United States?",
        "Show the stadium name and capacity ordered by capacity descending.",
    ],
    "pets_1": [
        "How many students are there?",
        "How many pets are cats?",
        "What are the first names of students older than 19?",
    ],
}

_model = None
_tokenizer = None
_load_error: str | None = None

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def _running_on_hf_space() -> bool:
    return bool(
        os.environ.get("SPACE_ID")
        or os.environ.get("SPACE_REPO_NAME")
        or os.environ.get("HF_SPACE_ID")
    )


def _resolved_model_id() -> str:
    explicit = os.environ.get("MODEL_ID", "").strip()
    if explicit:
        return explicit
    if os.environ.get("USE_STUB_GENERATOR", "").strip().lower() in {"1", "true", "yes"}:
        return ""
    if _running_on_hf_space():
        return DEFAULT_MODEL_ID
    return ""


def _demo_mode_label() -> str:
    model_id = _resolved_model_id()
    adapter = os.environ.get("ADAPTER_ID", "").strip()
    if not model_id:
        return "Schema-aware stub (no MODEL_ID). Not the trained GRPO checkpoint."
    if adapter:
        return f"Qwen2.5-3B + GRPO/QLoRA adapter (`{adapter}`)."
    return f"Base instruct model `{model_id}` (set ADAPTER_ID for GRPO LoRA weights)."


def _try_load_model():
    global _model, _tokenizer, _load_error
    if _model is not None or _load_error is not None:
        return
    model_id = _resolved_model_id()
    if not model_id:
        _load_error = "MODEL_ID not set — using schema-aware stub generator."
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = os.environ.get("ADAPTER_ID", "").strip() or None
        _tokenizer = AutoTokenizer.from_pretrained(
            adapter or model_id, trust_remote_code=True
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            _model = PeftModel.from_pretrained(_model, adapter)
        if not torch.cuda.is_available():
            _model = _model.to("cpu")
        _model.eval()
    except Exception as exc:  # noqa: BLE001
        _load_error = f"Model load failed: {exc}"
        _model = None
        _tokenizer = None


def stub_sql(schema: str, question: str) -> str:
    """Deterministic stub for demo without GPU — not a trained model."""
    q = question.lower()
    tables = []
    for line in schema.splitlines():
        if line.upper().startswith("CREATE TABLE"):
            tables.append(line.split()[2].split("(")[0])
    table = tables[0] if tables else "unknown"
    for t in tables:
        if t.lower() in q or t.lower().rstrip("s") in q:
            table = t
            break
    if "how many" in q or "count" in q or "total number" in q:
        # crude WHERE for simple filters in the sample set
        if "united states" in q:
            return f"SELECT count(*) FROM {table} WHERE Country = 'United States'"
        if "cat" in q:
            return "SELECT count(*) FROM Pets WHERE PetType = 'cat'"
        if "dog" in q:
            return "SELECT count(*) FROM Pets WHERE PetType = 'dog'"
        return f"SELECT count(*) FROM {table}"
    if "name and country" in q or "country of origin" in q:
        return "SELECT Name, Country FROM singer"
    if tables:
        return f"SELECT * FROM {table} LIMIT 5"
    return "SELECT 1"


@spaces.GPU
def generate(db_id: str, question: str, gold_sql: str):
    """Generation path decorated for HF ZeroGPU (works in stub mode too)."""
    if db_id not in SAMPLE_DBS:
        return "Unknown DB", "", "", ""
    db_path = SAMPLE_DBS[db_id]
    schema = schema_from_sqlite(db_path)
    _try_load_model()

    if _model is not None and _tokenizer is not None:
        import torch

        messages = build_messages(db_id, schema, question)
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(prompt, return_tensors="pt")
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        raw = _tokenizer.decode(gen, skip_special_tokens=True)
        adapter = os.environ.get("ADAPTER_ID", "").strip()
        note = f"Model: {_resolved_model_id()}"
        if adapter:
            note += f"\nAdapter: {adapter}"
    else:
        raw = stub_sql(schema, question)
        note = _load_error or "stub"

    sql = extract_sql(raw)
    exec_res = execute_sql(db_path, sql)
    rows_preview = str(exec_res.rows[:10]) if exec_res.ok else f"ERROR: {exec_res.error}"

    reward_txt = ""
    if gold_sql.strip():
        b = score_prediction(sql, db_path=db_path, gold_sql=gold_sql.strip())
        reward_txt = (
            f"validity={b.validity:.0f}  execution={b.execution:.0f}  "
            f"correctness={b.correctness:.0f}  total={b.total:.2f}"
        )

    return sql, rows_preview, reward_txt, f"{note}\n\nSchema:\n{schema}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Text-to-SQL GRPO Demo") as demo:
        gr.Markdown(
            f"""
# Text-to-SQL GRPO Demo
Spider-style SQLite DBs + execution-based reward scoring (validity / execution / correctness).

**This Space:** {_demo_mode_label()}

Portfolio visitors should see real Qwen generation here, not the heuristic stub. Set Space secret
`ADAPTER_ID` to your uploaded GRPO LoRA repo once training finishes.
            """.strip()
        )
        with gr.Row():
            db_id = gr.Dropdown(
                choices=list(SAMPLE_DBS.keys()),
                value="concert_singer",
                label="Database",
            )
            question = gr.Textbox(
                label="Question",
                value=EXAMPLES["concert_singer"][0],
                lines=2,
            )
        gold = gr.Textbox(
            label="Optional gold SQL (for reward scoring)",
            value="SELECT count(*) FROM singer",
            lines=2,
        )
        btn = gr.Button("Generate & execute", variant="primary")
        sql_out = gr.Code(label="Predicted SQL", language="sql")
        rows_out = gr.Textbox(label="Execution result (preview)", lines=4)
        reward_out = gr.Textbox(label="Reward breakdown", lines=1)
        meta_out = gr.Textbox(label="Notes / schema", lines=12)
        btn.click(
            generate,
            inputs=[db_id, question, gold],
            outputs=[sql_out, rows_out, reward_out, meta_out],
        )

        def _on_db(d):
            qs = EXAMPLES.get(d, [""])
            return qs[0]

        db_id.change(_on_db, inputs=[db_id], outputs=[question])
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
