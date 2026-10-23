# Portfolio ↔ live demo alignment

The [portfolio project](https://abhishek-sutaria.me/projects) describes:

- Post-trained **Qwen2.5-3B** with **QLoRA SFT + GRPO (3000×4)** on Spider execution rewards
- Execution-based reward and eval framework
- **Measured** Spider dev (n=1034): Base **60.83%** / SFT **61.70%** / GRPO **62.67%** exec accuracy (see [`results/MEASURED_RESULTS.md`](../results/MEASURED_RESULTS.md))
- Live demo: https://abhisheksutaria-text-to-sql-grpo.hf.space
- Code: https://github.com/abhishek-sutaria/text-to-sql-grpo
- Hub adapter: https://huggingface.co/abhisheksutaria/qwen2.5-3b-grpo-text-to-sql

Visitors clicking from the portfolio should get a **real model**, not the schema-aware stub.

## Checklist (RL / Space owner)

1. **Space hardware:** `zero-a10g` with `@spaces.GPU` on `generate()` (already in `README_spaces.md`).
2. **Base model:** `MODEL_ID=Qwen/Qwen2.5-3B-Instruct` (Space variable or default in `app.py` when `SPACE_ID` is set).
3. **GRPO weights:** Space secret **`ADAPTER_ID`** = `abhisheksutaria/qwen2.5-3b-grpo-text-to-sql`.
4. **Smoke test after deploy:** In Gradio, pick `concert_singer`, ask “How many singers do we have?”, optional gold `SELECT count(*) FROM singer`. Notes panel must **not** say “stub generator”; SQL should come from the model (may differ from gold while still executing).
5. **Metrics:** Reproduced with `scripts/eval.py` on Spider dev; JSON under `results/eval/` and table in README / `results/MEASURED_RESULTS.md`.

## Stub mode (dev only)

Set Space secret or variable `USE_STUB_GENERATOR=1` to force the heuristic stub (not for production portfolio demo).
