---
title: Text-to-SQL GRPO Demo
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
hardware: cpu-basic
app_file: app.py
pinned: false
license: mit
short_description: Spider-style text-to-SQL with execution rewards (free-track demo)
---

# Text-to-SQL GRPO

Portfolio demo for post-training text-to-SQL with GRPO-style execution rewards.

**Free track:** ships sample Spider-like DBs + reward harness. Set `MODEL_ID` to enable
a real Instruct model; otherwise a stub generator runs so the Space stays free/CPU-friendly.
