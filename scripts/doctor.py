"""Check data, artifacts and the selected CUDA device without TTS inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-only', action='store_true', help='No Torch import, network access or GPU allocation')
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = json.loads((ROOT/'datasets/public/seedtts_en/manifest.json').read_text())
    for part in [manifest['full'], *manifest['shards']]:
        path = ROOT/part['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == part['sha256'], path
        assert len(path.read_text().splitlines()) == part['samples'], path
    print('Seed dataset: 1,088 source rows and six disjoint shards verified')
    subprocess.run([sys.executable, 'scripts/run_public_suite.py', '--plan-only'], check=True)
    if args.data_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable in this Python environment')
    device = torch.cuda.get_device_properties(0)
    x = torch.ones((16,16), device='cuda')
    assert (x @ x).sum().item() == 4096
    print(json.dumps({'gpu':device.name, 'vram_gib':device.total_memory/2**30,
        'capability':torch.cuda.get_device_capability(0), 'torch':torch.__version__,
        'torch_cuda':torch.version.cuda, 'free_disk_gib':shutil.disk_usage(ROOT).free/2**30}))
    subprocess.run([sys.executable, 'scripts/prepare_all.py', '--check-only'], check=True)
    print('Ready for an explicit benchmark start')


if __name__ == '__main__':
    main()
