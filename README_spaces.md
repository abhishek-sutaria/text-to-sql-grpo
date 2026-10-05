---
title: Text-to-SQL GRPO Demo
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
hardware: zero-a10g
app_file: app.py
pinned: false
license: mit
short_description: Text-to-SQL GRPO demo with execution rewards
variables:
  MODEL_ID: Qwen/Qwen2.5-3B-Instruct
---

# Text-to-SQL GRPO

Portfolio demo for post-training text-to-SQL with GRPO-style execution rewards.

ZeroGPU Space defaults to **Qwen/Qwen2.5-3B-Instruct** (`MODEL_ID` above). Add Space secret
`ADAPTER_ID` (HF model repo with GRPO LoRA) for the post-trained checkpoint. Set
`USE_STUB_GENERATOR=1` only for offline CPU demos without a GPU.
