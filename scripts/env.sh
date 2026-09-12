#!/usr/bin/env bash
# Source this file in every shell used for setup, generation or scoring.
arena_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HOME="${HF_HOME:-$arena_root/.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$arena_root/.cache/nltk}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$arena_root${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$arena_root/.venv/bin:$PATH"
