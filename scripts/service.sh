#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
case "${1:-}" in
  benchmark) exec .venv/bin/python scripts/run_public_suite.py \
    --cache-budget-gib "${ARENA_CACHE_BUDGET_GIB:-45}" \
    --min-free-gib "${ARENA_MIN_FREE_GIB:-32}" ;;
  web) exec .venv/bin/voicehub-arena serve --runs runs --port 7860 ;;
  *) echo 'Usage: service.sh benchmark|web' >&2; exit 2 ;;
esac
