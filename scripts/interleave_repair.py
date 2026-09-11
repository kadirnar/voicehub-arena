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

subprocess.run(['supervisorctl','stop','voicehub-arena-extended'],check=True)
try:
    # The CLI's signal handler joins its isolated worker before releasing this
    # lock. Supervisor can report the wrapper stopped slightly before that.
    with (root/'runs/.gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write_json(full/'state.json',{'status':'paused','phase':'repair','models':args.models})
    command=[sys.executable,'-m','voicehub_arena.cli','run','--models',args.models,
             '--output',args.output,'--repeats','3','--timeout','3600',
             '--cache-budget-gib','45','--score-each-model','--resume']
    result=subprocess.call(command,cwd=root)
finally:
    subprocess.run(['supervisorctl','start','voicehub-arena-extended'],check=True)
raise SystemExit(result)
