"""Serial, checkpointed public benchmark campaign for the existing GPU.

One bounded provider/shard per CLI run. Pilot -> 256-prompt panel -> full splits.
All rows are kept; failures count against coverage and never become zero WER.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from voicehub_arena.storage import read_json, write_json
from voicehub_arena.benchmarks import build_jobs, restrict_public_scope, public_scope_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', default='datasets/public/suite.json')
    parser.add_argument('--after-service', default='voicehub-arena-followup')
    parser.add_argument('--after-run', default='runs/repairs-en-13')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    state_root = root/'runs/public-english-v2'
    state_root.mkdir(exist_ok=True)
    guard = (state_root/'.controller.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = state_root/'suite.json'
    scope = read_json(root/'configs/public-scope.json')
    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        from voicehub_arena.catalog import discover, declared_languages
        from voicehub_arena.inputs import frontend_protocol
        import voicehub
        catalog = [s for s in discover() if not declared_languages(s['model_type']) or
                   any(lang.lower().startswith('en') for lang in declared_languages(s['model_type']))]
        priority = ['kokoro', 'supertonic', 'vits', 'vui', 'inflecttts', 'styletts2', 'melotts', 'speecht5']
        catalog.sort(key=lambda s: (priority.index(s['model_type']) if s['model_type'] in priority else 100, s['model_type']))
        plan = restrict_public_scope({**read_json(args.suite), 'catalog':catalog}, scope)
        catalog = plan['catalog']
        plan.update(catalog=catalog, jobs=build_jobs(plan, catalog), normalization_id='whisper_english',
                    frontend_protocols={s['model_type']:frontend_protocol(s['model_type']) for s in catalog},
                    started_at=time.time(), status='waiting_for_repairs',
                    voicehub_commit=subprocess.check_output(['git','-C',str(Path(voicehub.__file__).parent),'rev-parse','HEAD'],text=True).strip(),
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip())
        write_json(plan_path, plan)
    if not public_scope_matches(plan, scope):
        raise ValueError('Active plan differs from configured scope; stop the controller and migrate the plan first')
    while args.after_service:
        state = subprocess.run(['supervisorctl','status',args.after_service], capture_output=True, text=True)
        finished = root/args.after_run/'state.json'
        if any(value in state.stdout.split() for value in ('EXITED','STOPPED')) and finished.exists():
            if read_json(finished).get('status') in {'finished', 'scoring_failed', 'failed', 'superseded'}:
                break
        print('Waiting for current repair job to release the GPU', flush=True)
        time.sleep(30)
    # Multiple controllers may queue work, but the shared GPU lock permits only
    # one generation/recognition run at a time.
    for job in plan['jobs']:
        if (state_root/'pause.request').exists():
            plan['status'] = 'paused_at_job_boundary'
            write_json(plan_path, plan)
            return
        if job['status'] in ('completed', 'partial', 'failed'):
            continue
        with (root/'runs/.gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
        dataset = next(d for d in plan['datasets'] if d['id'] == job['dataset'])
        text_transform = dataset.get('input_text_transform', 'identity')
        if hashlib.sha256(Path(job['shard']['path']).read_bytes()).hexdigest() != job['shard']['sha256']:
            raise ValueError('Dataset digest changed: ' + job['dataset'])
        run = root/'runs'/job['run']
        run.mkdir(exist_ok=True)
        overrides_path = run/'overrides.json'
        if not (run/'config.json').exists():
            write_json(overrides_path, read_json(root/'configs/models.json'))
            if job['model'] in ('melotts', 'gptsovits'):
                prepared = root/'datasets/prepared/public'/job['dataset']/str(job['shard']['index'])/job['model']
                job.update(status='running', phase='preparing_inputs')
                plan.update(status='running', current_job=job['run'])
                write_json(plan_path, plan)
                code = subprocess.call([sys.executable, 'scripts/prepare_linguistic.py', job['model'],
                    '--dataset', job['shard']['path'], '--output', str(prepared), '--overrides', str(overrides_path),
                    '--input-text-transform', text_transform])
                if code:
                    job.update(status='failed', error='Linguistic input preparation failed; see public-suite.log')
                    write_json(plan_path, plan)
                    continue
        job.update(status='running', phase='benchmark', started_at=time.time())
        plan.update(status='running', current_job=job['run'])
        write_json(plan_path, plan)
        command = [sys.executable, '-m', 'voicehub_arena.cli', 'run', '--models', job['model'],
            '--dataset', job['shard']['path'], '--dataset-manifest', f"datasets/public/{job['dataset']}/manifest.json",
            '--overrides', str(overrides_path), '--output', str(run), '--protocol-id', plan['protocol_id'],
            '--repeats', str(plan['repeats']), '--seed', str(plan['seed']), '--timeout', '86400',
            '--scoring-timeout', '14400', '--cache-budget-gib', '45', '--min-free-gib', '32',
            '--asr', plan['asr']['checkpoint'], '--asr-revision', plan['asr']['revision'],
            '--asr-device', 'cuda', '--asr-compute-type', 'float16', '--normalization', plan['normalization_id'],
            '--input-text-transform', text_transform,
            '--resume', '--resume-samples', '--score-each-model']
        code = subprocess.call(command)
        result_path = run/job['model']/'result.json'
        result = read_json(result_path) if result_path.exists() else {'status':'failed'}
        if result['status'] == 'disk_limit':
            job.update(status='waiting_for_storage', error=result.get('error'))
            plan['status'] = 'waiting_for_storage'
            write_json(plan_path, plan)
            # Stop without launching more downloads. Supervisor and heartbeat
            # can resume after off-box archival has been verified.
            raise SystemExit(75)
        job.update(status='completed' if code == 0 and result['status'] == 'completed' else 'partial',
                   finished_at=time.time(), worker_exit=code)
        write_json(plan_path, plan)
    plan['status'] = 'completed' if all(j['status'] == 'completed' for j in plan['jobs']) else 'partial'
    plan.pop('current_job', None)
    write_json(plan_path, plan)


if __name__ == '__main__':
    main()
