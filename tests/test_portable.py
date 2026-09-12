import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def test_fresh_checkout_plan_is_all_english_without_creating_state(tmp_path):
    checkout = tmp_path/'checkout'
    for directory in ('scripts', 'configs', 'voicehub_arena', 'datasets/public'):
        shutil.copytree(ROOT/directory, checkout/directory, ignore=shutil.ignore_patterns('__pycache__'))
    env = {**os.environ, 'PYTHONPATH':str(checkout)}
    process = subprocess.run([sys.executable, str(checkout/'scripts/run_public_suite.py'), '--plan-only'],
                             cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    plan = json.loads(process.stdout)
    assert len(plan['models']) == len(set(plan['models'])) == 33
    assert {'dia','mosstts','zonos2','qwen3tts','orpheustts','openvoice','styletts2'} <= set(plan['models'])
    assert 'irodoritts' not in plan['models']
    assert plan['texts_per_model'] == 1088
    assert plan['jobs'] == 198
    assert plan['planned_audio'] == 35904
    assert not (checkout/'runs').exists()
    assert not (checkout/'artifacts').exists()


def test_all_local_checkpoint_paths_have_frozen_preparation_hashes():
    spec = importlib.util.spec_from_file_location('prepare_all', ROOT/'scripts/prepare_all.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    overrides = json.loads((ROOT/'configs/public-models-lock.json').read_text())['overrides']
    paths = {}
    for name, override in overrides.items():
        if override.get('checkpoint','').startswith('artifacts/'):
            assert name in module.PREPARATIONS
        if name in module.PREPARATIONS:
            files = module.required_artifacts(override)
            assert files
            paths.update(files)
    assert 'artifacts/voxcpm2/audiovae.safetensors' in paths
    assert 'artifacts/styletts2/config.json' in paths
    assert all(len(sha)==64 for sha in paths.values())
    for file in ['configs/models.json','configs/public-models-lock.json','requirements/runtime.txt']:
        assert '/workspace/' not in (ROOT/file).read_text()
        assert '/Users/' not in (ROOT/file).read_text()


def test_bundled_reference_files_match_their_published_provenance():
    reference = ROOT/'datasets/reference'
    for filename in ['speecht5_slt.json','vibevoice_emma.json']:
        metadata = json.loads((reference/filename).read_text())
        audio = reference/('speecht5_slt.npy' if filename.startswith('speecht5') else 'vibevoice_emma.safetensors')
        assert hashlib.sha256(audio.read_bytes()).hexdigest() == metadata['sha256']
