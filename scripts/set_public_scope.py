"""Apply the reviewed scope configuration while the campaign is stopped."""
from pathlib import Path
import datetime
import fcntl
import json

from voicehub_arena.benchmarks import restrict_public_scope, public_scope_matches
from voicehub_arena.storage import read_json, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    state = root/'runs/public-english-v2'
    scope = read_json(root/'configs/public-scope.json')
    with (state/'.controller.lock').open('a') as controller, (root/'runs/.gpu.lock').open('a') as gpu:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = state/'suite.json'
        old = read_json(path)
        if public_scope_matches(old, scope):
            print('Scope already applied')
            return
        now = datetime.datetime.now(datetime.timezone.utc)
        backup = state/f"suite-before-scope-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        backup.write_bytes(path.read_bytes())
        plan = restrict_public_scope(old, scope)
        for job in plan.get('deferred_jobs', []):
            if job['status'] != 'running':
                continue
            job.update(status='deferred', previous_status='running',
                       reason='Model or dataset excluded from the first release at user request')
            run = root/'runs'/job['run']
            state_path = run/'state.json'
            previous = read_json(state_path) if state_path.exists() else {}
            write_json(state_path, {'status':'deferred', 'phase':'scope_change',
                                    'previous_state':previous, 'reason':job['reason']})
            result_path = run/job['model']/'result.json'
            if result_path.exists():
                result = read_json(result_path)
                result.update(previous_status=result['status'], status='deferred', deferred_reason=job['reason'])
                write_json(result_path, result)
        plan.update(status='ready')
        plan.setdefault('scope_changes', []).append({
            'changed_at':now.isoformat(), 'scope':scope, 'backup':backup.name,
            'reason':'User narrowed the first-release benchmark scope',
            'previous_dataset_ids':[d['id'] for d in old['datasets']],
            'previous_model_types':[s['model_type'] for s in old['catalog']],
            'preserved_seed_runs':[j['run'] for j in plan['jobs'] if (root/'runs'/j['run']/'config.json').exists()]})
        write_json(path, plan)
        (state/'pause.request').unlink(missing_ok=True)
        print(json.dumps({'scope':scope, 'active_jobs':len(plan['jobs']),
                          'active_models':[s['model_type'] for s in plan['catalog']],
                          'deferred_jobs':len(plan.get('deferred_jobs', [])),
                          'planned_audio':sum(d['total_samples'] for d in plan['datasets'])*len(plan['catalog'])*plan['repeats']}))


if __name__ == '__main__':
    main()
