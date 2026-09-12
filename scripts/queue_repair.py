"""Wait for a managed repair run to exit before starting the next GPU batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from voicehub_arena.storage import write_json


def run_finished(root, service_state):
    path = root / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    stopped = any(word in service_state.split() for word in ('EXITED', 'STOPPED'))
    return state.get('status') == 'finished' and stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-run', required=True)
    parser.add_argument('--after-service', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous = root / args.after_run
    output = root / args.output
    plan = {'models':args.models.split(','), 'after_run':args.after_run,
            'after_service':args.after_service, 'status':'waiting'}
    if output.exists():
        existing = output/'queue.json'
        if set(output.iterdir()) != {existing} or json.loads(existing.read_text()) != plan:
            raise RuntimeError('Queued validation requires a fresh output directory or the same unstarted plan')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'queue.json', plan)
    deadline = time.monotonic() + 8 * 3600
    print(f'Waiting for {args.after_run} and {args.after_service}; next: {args.models}', flush=True)
    while time.monotonic() < deadline:
        status = subprocess.run(['supervisorctl', 'status', args.after_service],
                                capture_output=True, text=True)
        if run_finished(previous, status.stdout):
            break
        time.sleep(10)
    else:
        raise TimeoutError('Previous repair did not finish within eight hours')
    write_json(output/'queue.json', {**plan, 'status':'starting'})
    code = subprocess.call([
        sys.executable, str(root / 'scripts/interleave_repair.py'),
        '--after-model', 'mosstts', '--models', args.models, '--output', args.output,
    ], cwd=root)
    write_json(output/'queue.json', {**plan, 'status':'finished' if code == 0 else 'failed', 'exit_code':code})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
