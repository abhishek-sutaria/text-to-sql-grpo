# Text-to-SQL GRPO (Free Track)

Post-train **Qwen2.5-3B-Instruct** for text-to-SQL with **QLoRA SFT + GRPO** on
execution rewards (Spider-style SQLite). This free track ships the **full codebase**,
smoke configs, unit tests, and a Gradio demo — without claiming paid-GPU resume metrics.

| Track | What you get |
|-------|----------------|
| **Free (this repo)** | Harness + sample DBs + smoke SFT/GRPO configs + eval + Gradio |
| **Paid GPU (later)** | Documented `configs/full_*.yaml` (~3k prompts × 4 generations) |

**Not** part of the Future of Jobs Vite app — standalone Python project.

## Quick start

```bash
git clone https://github.com/abhishek-sutaria/text-to-sql-grpo.git
cd text-to-sql-grpo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements.txt && pip install -e .
```

### Smoke tests (no GPU required)

```bash
# Unit tests: SQL exec + reward scoring
pytest -q

# Harness eval without loading a LLM (oracle + heuristic baselines)
python scripts/eval.py --data data/sample/dev.jsonl --mode oracle --out outputs/oracle.json
python scripts/eval.py --data data/sample/dev.jsonl --mode heuristic --out outputs/heuristic.json

# Prepare processed copies
python scripts/prepare_data.py --source sample
```

### Gradio demo (local)

```bash
pip install -r requirements-spaces.txt
python app.py
# open http://127.0.0.1:7860
```

Without `MODEL_ID`, the app uses a schema-aware stub so Spaces stay CPU-friendly.
With a GPU / enough RAM:

```bash
export MODEL_ID=Qwen/Qwen2.5-3B-Instruct
# optional LoRA after training:
# export ADAPTER_ID=outputs/smoke_sft/adapter
python app.py
```

## Training

### Smoke SFT / GRPO (tiny steps — needs GPU for Qwen-3B in practice)

```bash
# QLoRA SFT — 2 steps on 8 sample rows
python scripts/train_sft.py --config configs/smoke_sft.yaml

# GRPO — 2 steps, G=2 generations, execution reward
python scripts/train_grpo.py --config configs/smoke_grpo.yaml \
  --sft-adapter outputs/smoke_sft/adapter
```

CPU can load tiny models only; **Qwen2.5-3B smoke train expects CUDA + bitsandbytes**.
The reward/exec harness itself is CPU-only and fully tested.

### Full recipe (paid GPU — documented, not run on free track)

1. Download Spider 1.0:

```bash
python scripts/download_spider.py
# if auto-download fails: get the zip from https://yale-lily.github.io/spider
# place as data/spider_data.zip and re-run with --zip data/spider_data.zip
python scripts/prepare_data.py --source spider --out-dir data/processed
```

2. SFT then GRPO:

```bash
python scripts/train_sft.py --config configs/full_sft.yaml
python scripts/train_grpo.py --config configs/full_grpo.yaml
```

`configs/full_grpo.yaml` targets **~3000 prompts × 4 generations** with
validity / execution / correctness rewards (`0.2 / 0.3 / 0.5`).

3. Eval comparisons:

```bash
python scripts/eval.py --data data/processed/dev.jsonl --model Qwen/Qwen2.5-3B-Instruct \
  --out outputs/eval_base.json
python scripts/eval.py --data data/processed/dev.jsonl --model Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/full_sft/adapter --out outputs/eval_sft.json
python scripts/eval.py --data data/processed/dev.jsonl --model Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/full_grpo/adapter --out outputs/eval_grpo.json
```

## Reward design

For each completion:

1. **Validity** — parseable SQLite SQL (`sqlglot`)
2. **Execution** — runs read-only on the example DB
3. **Correctness** — result bag equals gold query result

```text
total = 0.2 * validity + 0.3 * execution + 0.5 * correctness
```

## Repo layout

```text
app.py                 Gradio demo (HF Spaces entrypoint)
configs/               smoke_* (free) and full_* (paid GPU) YAML
data/sample/           Mini Spider-like SQLite DBs + JSONL
scripts/               download / prepare / train_sft / train_grpo / eval
src/text_to_sql_grpo/  prompts, spider exec, rewards
tests/                 unit tests for exec + rewards
```

## Deploy to Hugging Face Spaces

**Note:** Gradio on `cpu-basic` needs [HF PRO](https://huggingface.co/pro) (quota 0 otherwise). Free ZeroGPU uses `hardware: zero-a10g` plus `@spaces.GPU` on the Gradio generation handler (`app.py`).

1. Create a Space (Gradio SDK) or reuse `abhisheksutaria/text-to-sql-grpo`.
2. Upload `app.py`, `src/`, `data/sample/`, and `requirements-spaces.txt` as Space `requirements.txt`.
3. Use `README_spaces.md` YAML as the Space README (`app_file: app.py`, `hardware: zero-a10g`).
4. Space vars: `MODEL_ID=Qwen/Qwen2.5-3B-Instruct` (default on ZeroGPU). Secret `ADAPTER_ID` for GRPO LoRA. See `docs/PORTFOLIO_ALIGNMENT.md`.

```bash
# from a machine with `huggingface-cli` logged in:
huggingface-cli upload <user>/text-to-sql-grpo . --repo-type=space
```

## Repository

Public repo: https://github.com/abhishek-sutaria/text-to-sql-grpo

Hugging Face Space (ZeroGPU stub demo): https://huggingface.co/spaces/abhisheksutaria/text-to-sql-grpo

## Honest results (free track)

| Check | Status |
|-------|--------|
| SQL execution env + unit tests | Pass on CPU |
| Reward scoring (valid / exec / correct) | Pass on CPU |
| Oracle eval on sample gold SQL | ~100% exec acc (sanity) |
| Heuristic baseline on sample | Low (expected) |
| Full Qwen-3B SFT + GRPO metrics | **Not run** without paid GPU |
| Resume “+X% execution accuracy” claims | **Do not cite** until full recipe completes |

## License

MIT. Spider data is subject to its own academic license — download separately for full runs.
