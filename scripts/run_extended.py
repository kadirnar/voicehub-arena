"""After coverage finishes, evaluate every registered provider on all prompts."""
import json
import os
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
# Revisit the full registry with the current reviewed overrides and protected
# Hub credentials. Japanese-only providers remain explicitly out of scope.
print('Extended English evaluation: full VoiceHub registry', flush=True)
command=[sys.executable,'-m','voicehub_arena.cli','run','--models','all',
         '--output','runs/english-extended','--repeats','3','--timeout','3600','--cache-budget-gib','45','--score-each-model','--resume']
environment = os.environ.copy()
environment.setdefault('NLTK_DATA',str(root.parent/'.nltk_data'))
raise SystemExit(subprocess.call(command,cwd=root,env=environment))
