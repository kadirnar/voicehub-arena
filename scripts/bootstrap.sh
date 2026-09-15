#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Full GPU setup targets Linux. Use the lightweight tests for development on other systems.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv' >&2; exit 1; }
command -v git >/dev/null || { echo 'Install git first.' >&2; exit 1; }
command -v espeak-ng >/dev/null || { echo 'Install espeak-ng, libsndfile1 and ffmpeg using your system package manager.' >&2; exit 1; }
native="$arena_root/.deps/voicehub"
revision=d67853dcfdf4385ce504dde66d6f513a446ca294
mkdir -p "$arena_root/.deps"
if [[ ! -d "$native/.git" ]]; then
  git clone --no-checkout https://github.com/kadirnar/voicehub.git "$native"
  git -C "$native" checkout --detach "$revision"
fi
[[ "$(git -C "$native" rev-parse HEAD)" == "$revision" ]] || { echo 'Existing native checkout has a different revision; use a fresh Arena checkout.' >&2; exit 1; }
patch_file="$arena_root/patches/voicehub-runtime.patch"
if git -C "$native" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo 'Validated runtime patch is already applied.'
else
  git -C "$native" diff --exit-code --quiet
  git -C "$native" apply --check "$patch_file"
  git -C "$native" apply "$patch_file"
fi
duration_patch="$arena_root/patches/conversation-duration-budget.patch"
if git -C "$native" apply --reverse --check "$duration_patch" 2>/dev/null; then
  echo 'ConversationTTS duration correction is already applied.'
else
  git -C "$native" apply --check "$duration_patch"
  git -C "$native" apply "$duration_patch"
fi
[[ -x .venv/bin/python ]] || uv venv --python 3.12 .venv
# Keep the validated Torch API version. The wheel index is configurable for the host.
uv pip install --python .venv/bin/python torch==2.8.0 torchaudio==2.8.0 \
  --index-url "${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
uv pip install --python .venv/bin/python -r requirements/runtime.txt -e "$native" -e .
uv pip check --python .venv/bin/python
mkdir -p "$NLTK_DATA" "$HF_HOME"
.venv/bin/python -m nltk.downloader -d "$NLTK_DATA" \
  cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
.venv/bin/python scripts/run_public_suite.py --plan-only
echo 'Installation complete. Authenticate with hf auth login, then run scripts/prepare_all.py.'
echo 'No benchmark or web service has been started.'
