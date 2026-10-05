#!/usr/bin/env bash
# Convenience targets — prefer documented python3 commands in README.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

case "${1:-help}" in
  test)
    pytest -q
    ;;
  smoke)
    python3 scripts/smoke_check.py
    python3 scripts/eval.py --data data/sample/dev.jsonl --mode oracle --out outputs/oracle.json
    ;;
  demo)
    python3 app.py
    ;;
  prepare)
    python3 scripts/prepare_data.py --source sample
    ;;
  help|*)
    echo "Usage: ./Makefile.sh [test|smoke|demo|prepare]"
    ;;
esac
