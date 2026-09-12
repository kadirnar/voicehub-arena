import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

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


@pytest.mark.parametrize('conversion_fails', [False, True])
def test_preparation_preserves_portable_settings(tmp_path, monkeypatch, conversion_fails):
    spec = importlib.util.spec_from_file_location('prepare_all', ROOT/'scripts/prepare_all.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'configs').mkdir()
    models = tmp_path/'configs/models.json'
    original = b'{"demo": {"checkpoint": "artifacts/demo"}}\n'
    models.write_bytes(original)
    payload = b'converted artifact'
    lock = tmp_path/'configs/public-models-lock.json'
    lock.write_text(json.dumps({'overrides': {'demo': {'checkpoint': 'artifacts/demo',
        'artifact_provenance': {'sha256': hashlib.sha256(payload).hexdigest()}}}}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'PREPARATIONS', {'demo': ['converter.py']})
    monkeypatch.setattr(sys, 'argv', ['prepare_all.py'])
    monkeypatch.chdir(tmp_path)

    def convert(*args, **kwargs):
        models.write_text('{"demo": {"checkpoint": "/workspace/artifacts/demo"}}')
        if conversion_fails:
            raise subprocess.CalledProcessError(1, 'converter')
        (tmp_path/'artifacts').mkdir()
        (tmp_path/'artifacts/demo').write_bytes(payload)

    monkeypatch.setattr(module.subprocess, 'run', convert)
    if conversion_fails:
        with pytest.raises(subprocess.CalledProcessError):
            module.main()
    else:
        module.main()
    assert models.read_bytes() == original
