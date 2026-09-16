"""Supervised native campaign. Unreviewed methods remain explicit pending work."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from voicehub_arena.native_protocol import write_json,applicable_metrics


def main():
    os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    cfg=json.loads(Path('configs/native-methods.json').read_text())
    run=Path('runs')/cfg['campaign'];run.mkdir(parents=True,exist_ok=True)
    controller_lock=(run/'controller.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=run/'campaign-status.json'
    state=json.loads(status_path.read_text()) if status_path.exists() else {'campaign':cfg['campaign'],'started_at':time.time(),'events':[]}
    pause=[False]
    signal.signal(signal.SIGUSR1,lambda s,f:pause.__setitem__(0,True))
    def save(**values):
        state.update(updated_at=time.time(),**values);write_json(status_path,state)
    def action(command,experiment,phase,step):
        save(status='running',experiment=experiment,phase=phase,action=step)
        with (run/(experiment+'--'+phase+'--'+step+'.log')).open('a') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
        state['events'].append(dict(experiment=experiment,phase=phase,action=step,exit_code=code,time=time.time()));save()
        return code==0
    def completed(identifier,phase):
        path=run/phase/identifier/'result.json'
        if not path.exists():return False
        result=json.loads(path.read_text())
        return result.get('status')=='completed' and len(result['rows'])==result['expected_samples']
    from scripts.setup_native_model import setup
    active=[s for s in cfg['experiments'] if s.get('verified_api')]
    # Fast independent engines establish end-to-end correctness first.
    active.sort(key=lambda s:0 if s['family']=='vits' else 1 if s['family']=='kokoro'
                else 3 if s.get('settings',{}).get('inference_device')=='cpu' else 2)
    save(inventory_candidates=len(cfg['experiments']),review_pending=[s['id'] for s in cfg['experiments'] if not s.get('verified_api')])
    groups={}
    for spec in active:groups.setdefault((spec['repo'],spec['revision']),[]).append(spec)
    schedule=[(phase,spec) for group in groups.values() for phase in ('pilot','full') for spec in group]
    for phase,spec in schedule:
            identifier=spec['id']
            if completed(identifier,phase):
                receipt=run/'publication'/(phase+'--'+identifier+'.json')
                if not receipt.exists():
                    action([sys.executable,'-m','voicehub_arena.native_eval','summarize','--experiment',identifier,'--phase',phase],identifier,phase,'summarize')
                    action([sys.executable,'scripts/publish_native_progress.py','--experiment',identifier,'--phase',phase],identifier,phase,'publish')
                if phase=='full' and receipt.exists():
                    action([sys.executable,'scripts/offload_native_audio.py','--experiment',identifier,'--phase',phase],identifier,phase,'offload')
                continue
            if pause[0]:save(status='paused_after_phase');return
            if phase=='full' and not completed(identifier,'pilot'):continue
            try:python=setup(spec)
            except Exception as exc:
                state['events'].append(dict(experiment=identifier,phase=phase,action='setup',error=str(exc),time=time.time()));save();continue
            base=['-u','-m','voicehub_arena.native_eval']
            suffix=['--experiment',identifier,'--phase',phase]
            if not action([python,*base,'generate',*suffix],identifier,phase,'generate'):continue
            # The setup supervisor may still be installing the independent legacy metrics.
            while not Path('artifacts/native-metrics/manifest.json').exists():
                save(status='waiting_for_metric_setup',experiment=identifier,phase=phase)
                check=subprocess.run(['supervisorctl','status','voicehub-native-setup'],capture_output=True,text=True)
                if 'RUNNING' not in check.stdout:
                    save(status='needs_attention',error='Native metric setup did not finish');return
                time.sleep(30)
            for metric in applicable_metrics(spec['uses_reference']):
                python='.venvs/quality-legacy/bin/python' if metric in ('utmos22','wavlm_sim') else '.venvs/native-core/bin/python'
                if not action([python,*base,metric,*suffix],identifier,phase,metric):break
            action(['.venv/bin/python',*base,'summarize',*suffix],identifier,phase,'summarize')
            if completed(identifier,phase):
                published=action([sys.executable,'scripts/publish_native_progress.py','--experiment',identifier,'--phase',phase],identifier,phase,'publish')
                if published and phase=='full':
                    action([sys.executable,'scripts/offload_native_audio.py','--experiment',identifier,'--phase',phase],identifier,phase,'offload')
    missing=[s['id'] for s in cfg['experiments'] if s.get('availability')!='not_supported_upstream' and not completed(s['id'],'full')]
    save(status='completed' if not missing else 'needs_attention',incomplete=missing,experiment=None,action=None)


if __name__=='__main__':main()
