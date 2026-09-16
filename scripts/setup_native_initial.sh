#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="$PWD/.cache/huggingface"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p .venvs .deps/native artifacts/native-metrics runs
test -x .venvs/native-core/bin/python || uv venv .venvs/native-core --python .venv/bin/python --seed
.venvs/native-core/bin/python - <<'PY'
import site,subprocess
from pathlib import Path
p=Path(site.getsitepackages()[0])/'arena-base.pth'
base=subprocess.check_output(['.venv/bin/python','-c','import site;print(site.getsitepackages()[0])'],text=True).strip()
p.write_text(base+'\n')
PY
uv pip install --python .venvs/native-core/bin/python --no-deps 'kokoro @ git+https://github.com/hexgrad/kokoro.git@dfb907a02bba8152ca444717ca5d78747ccb4bec'
uv pip install --python .venvs/native-core/bin/python 'loguru==0.7.3' 'resampy==0.4.3' 'pandas==2.2.3'
.venvs/native-core/bin/python -c 'import kokoro,torch;print("native Kokoro",kokoro.__file__,torch.__version__)'
test -x .venvs/quality-legacy/bin/python || uv venv .venvs/quality-legacy --python 3.10 --seed
uv pip install --python .venvs/quality-legacy/bin/python 'torch==1.13.1+cu117' 'torchaudio==0.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install --python .venvs/quality-legacy/bin/python 'numpy==1.23.5' 'Cython==0.29.36' 'setuptools==59.5.0' 'wheel' 'six==1.17.0' 'pip<24.1' 'soundfile==0.13.1' 'librosa==0.10.2.post1' 'huggingface-hub==0.36.0' 'gdown==5.2.0' 'fire==0.7.1' 'einops==0.8.1'
env -u CUDA_HOME MAX_JOBS=4 .venvs/quality-legacy/bin/python -m pip install --no-build-isolation 'transformers==4.28.1' 'timm==0.6.13' 'torchvision==0.14.1' 'fairseq @ git+https://github.com/pytorch/fairseq.git@d03f4e771484a433f025f47744017c2eb6e9c6bc' 'pytorch-lightning==1.5.10' 'torchmetrics==0.7.2' 'protobuf==3.20.3'
.venvs/quality-legacy/bin/python -m pip install --no-deps 's3prl @ git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3'
.venvs/quality-legacy/bin/python -c 'import torch,fairseq,pytorch_lightning;print("legacy quality runtime",torch.__version__,torch.cuda.is_available())'
.venv/bin/python scripts/prepare_native_metrics.py
printf '%s\n' 'Native environments and metric assets prepared. No benchmark was started.'
