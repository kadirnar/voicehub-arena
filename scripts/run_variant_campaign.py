"""Resume the frozen variant campaign. A failed pilot blocks only that model's full run."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from voicehub_arena.storage import write_json
from voicehub_arena.variant_eval import load_campaign, digest, SCORED_STATUSES

cfg,_=load_campaign('configs/variant-campaign.json')
root=Path.cwd();run=root/'runs'/cfg['campaign'];run.mkdir(parents=True,exist_ok=True)
lock=(root/'runs/.gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
pause_requested=False
def pause_after_phase(signum,frame):
    global pause_requested
    pause_requested=True
signal.signal(signal.SIGUSR1,pause_after_phase)
progress=run/'campaign-status.json'
state=json.loads(progress.read_text()) if progress.exists() else {'campaign':cfg['campaign'],'started_at':time.time(),'events':[]}

def save(**values):
    state.update(updated_at=time.time(),**values);write_json(progress,state)

def run_action(action, model=None, phase=None):
    cmd=[sys.executable,'-u','-m','voicehub_arena.variant_eval',action]
    if model:cmd+=['--model',model,'--phase',phase]
    save(status='running',action=action,model=model,phase=phase)
    code=subprocess.call(cmd)
    state['events'].append({'action':action,'model':model,'phase':phase,'exit_code':code,'time':time.time()});save()
    return code==0

def complete(model,phase):
    p=run/phase/model/'result.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    return r['status']=='completed' and len(r['rows'])==r['expected_samples'] and all(x['status'] in SCORED_STATUSES for x in r['rows'])

def remove_staging(model):
    # Only files downloaded into this campaign's explicit staging root are evicted.
    cache=root/'.cache/variant-staging'/model
    assert cache.parent==root/'.cache/variant-staging' and '/' not in model
    if cache.exists():shutil.rmtree(cache)

if not (run/'reference-words.json').exists() and not run_action('align'):
    save(status='failed',error='Reference alignment failed');sys.exit(1)
for phase in ['pilot','full']:
    for spec in cfg['models']:
        model=spec['id']
        if complete(model,phase):continue
        if pause_requested:
            save(status='paused_after_phase');sys.exit(0)
        if phase=='full' and not complete(model,'pilot'):
            state['events'].append({'model':model,'phase':phase,'status':'blocked_by_pilot'});save();continue
        if not run_action('generate',model,phase):
            remove_staging(model);continue
        run_action('score',model,phase)
        remove_staging(model)
        # Publisher is deliberately separate; local scoring succeeds even during a Hub outage.
        publisher=root/'scripts/publish_variant_progress.py'
        if publisher.exists():
            code=subprocess.call([sys.executable,str(publisher),'--model',model,'--phase',phase])
            state['events'].append({'action':'publish','model':model,'phase':phase,'exit_code':code,'time':time.time()});save()
missing=[s['id'] for s in cfg['models'] if not complete(s['id'],'full')]
save(status='completed' if not missing else 'needs_attention',incomplete=missing)
print(json.dumps({'campaign':cfg['campaign'],'status':state['status'],'incomplete':missing}),flush=True)
sys.exit(bool(missing))
