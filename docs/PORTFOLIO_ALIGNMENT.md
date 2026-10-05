# Portfolio ↔ live demo alignment

The [portfolio project](https://abhishek-sutaria.me/projects) describes:

- Post-trained **Qwen2.5-3B** with **GRPO + QLoRA** on Spider 1.0
- Execution-based reward and eval framework
- Reported metrics: execution accuracy **34.2% → 58.6%**, SQL execution success **68.5% → 86.9%**
- Live demo: https://abhisheksutaria-text-to-sql-grpo.hf.space
- Code: https://github.com/abhishek-sutaria/text-to-sql-grpo

Visitors clicking from the portfolio should get a **real model**, not the schema-aware stub.

## Checklist (RL / Space owner)

1. **Space hardware:** `zero-a10g` with `@spaces.GPU` on `generate()` (already in `README_spaces.md`).
2. **Base model:** `MODEL_ID=Qwen/Qwen2.5-3B-Instruct` (Space variable or default in `app.py` when `SPACE_ID` is set).
3. **GRPO weights:** Upload LoRA adapter to Hugging Face Hub (e.g. `abhisheksutaria/qwen2.5-3b-grpo-text-to-sql`) and set Space secret **`ADAPTER_ID`** to that repo id.
4. **Smoke test after deploy:** In Gradio, pick `concert_singer`, ask “How many singers do we have?”, optional gold `SELECT count(*) FROM singer`. Notes panel must **not** say “stub generator”; SQL should come from the model (may differ from gold while still executing).
5. **Metrics:** Reproduce portfolio numbers with `scripts/eval.py` on Spider dev and keep a JSON artifact under `outputs/` or document in README when full GPU recipe completes.

## Stub mode (dev only)

Set Space secret or variable `USE_STUB_GENERATOR=1` to force the heuristic stub (not for production portfolio demo).
