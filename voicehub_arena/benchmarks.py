"""Dataset-specific public benchmark cohorts; never pool unrelated corpora."""
import hashlib
import copy
import json
from pathlib import Path
from .storage import read_json, read_rows
from .metrics import summarize

PHASES = {'pilot': {'pilot'}, 'panel': {'pilot', 'panel'}, 'full': {'pilot', 'panel', 'expansion'}}


def public_overrides(root, catalog, prepared):
    """Pin primary model revisions once for the whole multi-shard campaign."""
    from .storage import write_json
    path = Path(root)/'configs/public-models-lock.json'
    if not path.exists():
        from huggingface_hub import HfApi
        from .catalog import discover
        frozen = read_json(Path(root)/'configs/models.json')
        api = HfApi()
        for spec in discover():
            override = frozen.setdefault(spec['model_type'], {})
            languages = spec.get('declared_languages')
            if languages and not any(lang.lower().startswith('en') for lang in languages):
                continue
            checkpoint = override.get('checkpoint', spec['checkpoint'])
            if checkpoint and '/' in checkpoint and not Path(checkpoint).exists():
                config = override.setdefault('config', {})
                config['revision'] = api.model_info(checkpoint, revision=config.get('revision')).sha
        write_json(path, {'overrides':frozen, 'policy':'One primary checkpoint revision and generation configuration per public campaign'})
    frozen = copy.deepcopy(read_json(path)['overrides'])
    # Offline frontends depend on the shard. Their verified tensors and timing
    # replace only the prepared-input location, not generation settings.
    for spec in catalog:
        name = spec['model_type']
        for key in ('prepared_inputs', 'timing_scope'):
            if key in prepared.get(name, {}):
                frozen[name][key] = prepared[name][key]
    return frozen, hashlib.sha256(path.read_bytes()).hexdigest()


def build_jobs(suite, catalog):
    jobs = []
    order = ['seedtts_en', 'librispeech_test_clean', 'librispeech_test_other', 'libritts_test_clean', 'emergenttts']
    datasets = sorted(suite['datasets'], key=lambda d: order.index(d['id']))
    for phase in ('pilot', 'panel', 'expansion'):
        # Visit every provider in each coverage stage before expansion.
        for spec in catalog:
            for dataset in datasets:
                for shard in dataset['shards']:
                    if shard['phase'] != phase:
                        continue
                    name = f"pub-v2-{spec['model_type']}-{dataset['id']}-{shard['index']:03d}"
                    jobs.append(dict(run=name, model=spec['model_type'], dataset=dataset['id'],
                                     shard=shard, status='pending'))
    return jobs


def public_report(runs, dataset_id, phase='pilot'):
    if phase not in PHASES:
        raise ValueError('Unknown coverage phase')
    runs = Path(runs)
    plan = read_json(runs/'public-english-v2/suite.json')
    dataset = next(d for d in plan['datasets'] if d['id'] == dataset_id)
    selected_shards = [s for s in dataset['shards'] if s['phase'] in PHASES[phase]]
    dataset_rows = []
    for shard in selected_shards:
        path = runs.parent/shard['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError('Frozen dataset changed')
        dataset_rows.extend(read_rows(path))
    results = []
    active = []
    for spec in plan['catalog']:
        jobs = [j for j in plan['jobs'] if j['dataset'] == dataset_id and j['model'] == spec['model_type']
                and j['shard']['phase'] in PHASES[phase]]
        rows, errors, origins, flags = [], [], [], []
        details = {}
        synthesis_protocols = set()
        for job in jobs:
            path = runs/job['run']/spec['model_type']/'result.json'
            if job.get('error'):
                errors.append(job['error'])
            if job['status'] in ('failed','partial','waiting_for_storage'):
                flags.append(job['status'])
            if path.exists():
                result = read_json(path)
                cfg = read_json(path.parent.parent/'config.json')
                if cfg['asr'] != plan['asr'] or cfg.get('normalization_id') != plan['normalization_id']:
                    raise ValueError('Run scorer differs from frozen public suite')
                if cfg.get('dataset_part', {}).get('sha256') != job['shard']['sha256']:
                    raise ValueError('Run data differs from the public suite')
                override = cfg.get('overrides', {}).get(spec['model_type'], {})
                synthesis_protocols.add(json.dumps({
                    'checkpoint':result.get('checkpoint', spec['checkpoint']),
                    'revision':result.get('revision', override.get('config', {}).get('revision')),
                    'config':override.get('config'), 'generation':override.get('generation'),
                    'text_prefix':override.get('text_prefix'), 'runtime_adapter':override.get('runtime_adapter'),
                }, sort_keys=True))
                rows.extend({**r, 'source_run': job['run']} for r in result.get('rows', []))
                if result.get('error'):
                    errors.append(result['error'])
                flags.append(result['status'])
                origins.append(job['run'])
                details = {k: result[k] for k in ('checkpoint', 'timing_scope', 'runtime_adapter') if k in result}
            if job['status'] == 'running':
                active.append(dict(run=job['run'], model=job['model'], phase=job['shard']['phase']))
        if len({(r['id'], r['repeat']) for r in rows}) != len(rows):
            raise ValueError('Overlapping benchmark shards')
        if len(synthesis_protocols) > 1:
            raise ValueError('Mixed synthesis settings for ' + spec['model_type'] + '; rerun a consistent cohort')
        summary = summarize(rows)
        planned = len(dataset_rows) * plan['repeats']
        complete = summary['scored'] == planned
        status = ('completed' if complete else 'running' if any(j['status'] == 'running' for j in jobs)
                  else 'partial' if rows or flags else 'pending')
        results.append({**spec, **details, 'status':status, 'rows':rows, 'summary':summary,
                        'planned_samples':planned, 'ranking_eligible':complete,
                        'source_runs':origins, 'error':'\n'.join(dict.fromkeys(errors)) or None})
    return {'run':'public-' + dataset_id + '-' + phase, 'config':{
        'dataset':dataset_rows, 'repeats':plan['repeats'], 'asr':plan['asr'],
        'normalization_id':plan['normalization_id'], 'protocol':plan['protocol_id'],
        'gpu':plan.get('gpu'), 'voicehub_commit':plan.get('voicehub_commit'),
        'dataset_manifest':dataset}, 'results':results, 'active_runs':active,
        'coverage_phase':phase, 'total_dataset_samples':dataset['total_samples'],
        'scope':'Published test texts; fixed provider voices/references; intelligibility only. Not official zero-shot SIM or Emergent model-as-judge scores.'}
