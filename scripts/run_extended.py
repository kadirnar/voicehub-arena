"""After coverage finishes, evaluate successful/repaired providers on all prompts."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
probe=root/'runs/all-models-en'
while True:
    state=probe/'state.json'
    if state.exists():
        status=json.loads(state.read_text()).get('status')
        if status in ('finished','scoring_failed'):
            break
        if status=='failed':
            raise SystemExit('Coverage run failed; inspect its logs before continuing')
    time.sleep(5)
models=[]
for path in probe.glob('*/result.json'):
    result=json.loads(path.read_text())
    if any(row.get('audio') for row in result.get('rows',[])):
        models.append(result['model_type'])
# These two have concrete configuration repairs after their initial probe.
models=sorted(set(models)|{'kokoro','bark'})
print('Extended English evaluation:', ', '.join(models), flush=True)
command=[sys.executable,'-m','voicehub_arena.cli','run','--models',','.join(models),
         '--output','runs/english-extended','--repeats','3','--timeout','1200','--resume']
raise SystemExit(subprocess.call(command,cwd=root))
