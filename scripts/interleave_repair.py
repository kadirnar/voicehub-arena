"""Complete the current model, validate repairs, and resume the full benchmark."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from voicehub_arena.storage import write_json

parser = argparse.ArgumentParser()
parser.add_argument('--after-model',required=True)
parser.add_argument('--models',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--preflight', help='Optional GPU validation script to run before this repair set')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
full = root/'runs/english-extended'
terminal = {'completed','partial','blocked','failed','timeout','disk_limit','unsupported_language'}
deadline = time.monotonic()+7200
while time.monotonic()<deadline:
    path = full/args.after_model/'result.json'
    if path.exists() and json.loads(path.read_text())['status'] in terminal:
        break
    time.sleep(1)
else:
    raise SystemExit('The current model did not finish in the repair wait window')

service = subprocess.run(['supervisorctl','status','voicehub-arena-extended'],
                         capture_output=True,text=True)
if any(state in service.stdout.split() for state in ('RUNNING', 'STARTING', 'BACKOFF')):
    subprocess.run(['supervisorctl','stop','voicehub-arena-extended'],check=True)
elif not any(state in service.stdout.split() for state in ('STOPPED', 'EXITED', 'FATAL')):
    raise RuntimeError('Could not determine the full benchmark service state')
try:
    # The CLI's signal handler joins its isolated worker before releasing this
    # lock. Supervisor can report the wrapper stopped slightly before that.
    with (root/'runs/.gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_json(full/'state.json',{'status':'paused','phase':'repair','models':args.models})
        if args.preflight:
            subprocess.run([sys.executable, args.preflight], cwd=root, check=True, timeout=1200)
    command=[sys.executable,'-m','voicehub_arena.cli','run','--models',args.models,
             '--output',args.output,'--repeats','3','--timeout','3600',
             '--cache-budget-gib','45','--score-each-model','--resume']
    result=subprocess.call(command,cwd=root)
finally:
    subprocess.run(['supervisorctl','start','voicehub-arena-extended'],check=True)
raise SystemExit(result)
