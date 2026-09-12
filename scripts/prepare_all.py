"""Rebuild required local artifacts on CPU; never launch benchmark inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREPARATIONS = {
    'styletts2': ['scripts/prepare_styletts2.py'],
    'cosyvoice': ['scripts/prepare_cosyvoice_checkpoint.py'],
    'conversationtts': ['scripts/prepare_conversationtts.py'],
    'voxcpm': ['scripts/prepare_remaining_codecs.py', 'voxcpm'],
    'xtts': ['scripts/prepare_remaining_codecs.py', 'xtts'],
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def required_artifacts(override):
    """Immutable hashes are taken from the frozen campaign, not conversion output."""
    provenance = override.get('artifact_provenance', {})
    expected = {}
    checkpoint = override.get('checkpoint', '')
    if checkpoint.startswith('artifacts/'):
        if provenance.get('files_sha256'):
            expected.update({str(Path(checkpoint)/name): sha for name, sha in provenance['files_sha256'].items()})
        elif provenance.get('sha256'):
            expected[checkpoint] = provenance['sha256']
        else:
            raise ValueError('Local checkpoint has no frozen digest: ' + checkpoint)
    for key, sha in provenance.get('config_files_sha256', {}).items():
        expected[override['config'][key]] = sha
    if 'typed_config_sha256' in provenance:
        expected[override['config']['config_path']] = provenance['typed_config_sha256']
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Report missing/mismatched artifacts without downloads')
    args = parser.parse_args()
    os.chdir(ROOT)
    lock_path = ROOT/'configs/public-models-lock.json'
    lock_bytes = lock_path.read_bytes()
    models_path = ROOT/'configs/models.json'
    models_bytes = models_path.read_bytes()
    overrides = json.loads(lock_bytes)['overrides']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'HF_HOME':os.environ.get('HF_HOME',str(ROOT/'.cache/huggingface')),
           'NLTK_DATA':os.environ.get('NLTK_DATA',str(ROOT/'.cache/nltk')), 'PYTHONPATH':str(ROOT)}
    failures = []
    try:
        for name, command in PREPARATIONS.items():
            expected = required_artifacts(overrides[name])
            invalid = [f for f, sha in expected.items() if not (ROOT/f).is_file() or digest(ROOT/f) != sha]
            if invalid and not args.check_only:
                print('PREPARE', name, flush=True)
                subprocess.run([sys.executable, *command], env=env, check=True)
                invalid = [f for f, sha in expected.items() if not (ROOT/f).is_file() or digest(ROOT/f) != sha]
            if invalid:
                failures.extend(invalid)
            print(name, 'MISSING_OR_CHANGED: '+', '.join(invalid) if invalid else 'VERIFIED', flush=True)
    finally:
        # Standalone legacy converters also update their model overrides using
        # absolute paths. The portable campaign already has frozen relative
        # paths and settings; retain them even if a conversion fails.
        if models_path.read_bytes() != models_bytes:
            models_path.write_bytes(models_bytes)
    # Reference voices/vectors are already bundled and hash-verified by the runner.
    # Per-shard MeloTTS / GPT-SoVITS inputs are built by run_public_suite.py.
    if lock_path.read_bytes() != lock_bytes:
        raise ValueError('Preparation changed the frozen model configuration')
    if failures:
        raise SystemExit('Required artifacts are missing or differ from frozen digests; inspect the preparation logs.')
    print('All required local artifacts verified. Benchmark remains stopped.')


if __name__ == '__main__':
    main()
