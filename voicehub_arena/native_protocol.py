"""Contracts shared by isolated upstream runtimes; no VoiceHub model imports."""
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import sys


class RejectVoiceHub(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'voicehub' or fullname.startswith('voicehub.'):
            raise ImportError('Native campaign forbids VoiceHub inference: ' + fullname)
        return None


def forbid_voicehub():
    if any(k == 'voicehub' or k.startswith('voicehub.') for k in sys.modules):
        raise RuntimeError('VoiceHub was imported before native runtime isolation')
    sys.meta_path.insert(0, RejectVoiceHub())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def load_experiment(manifest, identifier, phase):
    cfg = json.loads(Path(manifest).read_text())
    if digest(cfg['dataset']) != cfg['dataset_sha256']:
        raise ValueError('Native dataset digest mismatch')
    dataset = [json.loads(s) for s in Path(cfg['dataset']).read_text().splitlines()]
    if len(dataset) != cfg['expected_samples'] or len({r['id'] for r in dataset}) != len(dataset):
        raise ValueError('Native dataset coverage mismatch')
    spec = next(s for s in cfg['experiments'] if s['id'] == identifier)
    if not spec.get('verified_api') or not spec.get('revision'):
        raise ValueError('Upstream API and checkpoint must be verified before execution')
    rows = dataset if phase == 'full' else [dataset[i] for i in cfg['pilot_indices']]
    contract = {k: cfg[k] for k in ('campaign','dataset_sha256','expected_samples','seed','asr','metrics')}
    contract.update(experiment=spec, phase=phase, selected_ids=[r['id'] for r in rows])
    directory = Path('runs') / cfg['campaign'] / phase / identifier
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'contract.json'
    if lock.exists():
        if canonical_hash(json.loads(lock.read_text())) != canonical_hash(contract):
            raise ValueError('Frozen experiment changed; use a new experiment ID')
    else:
        write_json(lock, contract)
    return cfg, spec, rows, directory, canonical_hash(contract)


def applicable_metrics(uses_reference):
    return ['asr', 'dnsmos', 'utmos22'] + (['wavlm_sim'] if uses_reference else [])


def verify_rows(rows, expected, directory):
    expected = {r['id']: r for r in expected}
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate native sample IDs')
    for row in rows:
        item = expected[row['id']]
        if row['reference'] != item['reference'] or row['text'] != item['text']:
            raise ValueError('Saved target text changed')
        if row['generation_status'] == 'ok':
            audio = directory / row['audio']
            if audio.parent != directory / 'audio' or digest(audio) != row['audio_sha256']:
                raise ValueError('Saved native audio changed')
        elif row['generation_status'] != 'no_audio' or row.get('audio'):
            raise ValueError('Infrastructure failures cannot become evaluated samples')


def metric_summary(rows, metric, field):
    import numpy as np
    values = [r['scores'][metric][field] for r in rows
              if r.get('scores', {}).get(metric, {}).get('status') == 'ok']
    if not values:
        return {'value': None, 'n': 0, 'ci95': None}
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite quality score')
    a = np.asarray(values)
    rng = np.random.default_rng(42)
    means = np.mean(a[rng.integers(0, len(a), size=(1000, len(a)))], axis=1)
    return {'value': float(a.mean()), 'n': len(a),
            'ci95': [float(v) for v in np.percentile(means, [2.5, 97.5])]}
