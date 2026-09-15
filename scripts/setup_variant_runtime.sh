#!/usr/bin/env bash
# Run from the repository root after scripts/bootstrap.sh has installed the base runtime.
set -euo pipefail
revision=8687268f4ed3ed20704638fd353b51491de3b476
if [ ! -d .deps/dia2 ]; then
  git clone https://github.com/nari-labs/dia2.git .deps/dia2
  git -C .deps/dia2 checkout --detach "$revision"
fi
if [ -d .deps/dia2/.git ]; then
  test "$(git -C .deps/dia2 rev-parse HEAD)" = "$revision"
  if git -C .deps/dia2 apply --check ../../patches/dia2-torch28-cudnn.patch; then
    git -C .deps/dia2 apply ../../patches/dia2-torch28-cudnn.patch
  else
    git -C .deps/dia2 apply --reverse --check ../../patches/dia2-torch28-cudnn.patch
  fi
fi
.venv/bin/python - <<'VERIFY'
from pathlib import Path
import hashlib,json
m=json.loads(Path('configs/dia2-source-files.json').read_text())
for name,sha in m['files'].items():
    p=Path('.deps/dia2')/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sha, 'Dia2 source differs: '+name
print('Pinned Dia2 source and Torch 2.8 compatibility patch verified')
VERIFY
uv pip install --python .venv/bin/python --no-deps -e .deps/dia2
uv pip install --python .venv/bin/python sphn==0.2.0
.venv/bin/python -c 'import dia2,sphn,torch,transformers; assert torch.__version__.startswith("2.8."); assert transformers.__version__ == "4.57.6"; print("Variant runtime ready")'
