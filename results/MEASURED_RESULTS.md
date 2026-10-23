# Measured results (Spider dev, n=1034)

All figures from `scripts/eval.py` JSON on Spider **dev**. Raw artifacts live in `results/eval/`.

## Recipe

| Stage | Config | Notes |
|-------|--------|-------|
| SFT | `configs/lean_full_sft.yaml` | Full Spider train (8659), 1 epoch, batch 1 / accum 16 / seq 1024 / grad checkpoint |
| GRPO | `configs/lean_full_grpo.yaml` | **3000 prompts × 4 generations**, batch 4 / accum 4, completion 160, prompt 896 |
| Base | `Qwen/Qwen2.5-3B-Instruct` | QLoRA r=16, α=32, 4-bit |

## Eval table

| Checkpoint | Exec accuracy | Exec success | Mean reward | Validity |
|------------|---------------|--------------|-------------|----------|
| Base | **60.83%** | **88.78%** | **0.770** | 99.81% |
| SFT | **61.70%** | 81.33% | 0.752 | 99.71% |
| GRPO | **62.67%** | 82.40% | 0.761 | **100%** |

Raw summary fields:

| | base | SFT | GRPO |
|--|------|-----|------|
| `exec_acc` | 0.6083 | 0.6170 | 0.6267 |
| `exec_success` | 0.8878 | 0.8133 | 0.8240 |
| `mean_reward` | 0.7701 | 0.7519 | 0.7605 |

**Interpretation:** GRPO beats base and SFT on execution accuracy (+1.8 pp vs base, +1.0 pp vs SFT). Base leads execution success and mean reward on this harness.

## Links

- Hub adapter: https://huggingface.co/abhisheksutaria/qwen2.5-3b-grpo-text-to-sql
- Live Space: https://huggingface.co/spaces/abhisheksutaria/text-to-sql-grpo
- GitHub: https://github.com/abhishek-sutaria/text-to-sql-grpo
- Eval JSON: `results/eval/eval_{base,sft,grpo}.json` and `*_summary.json`
