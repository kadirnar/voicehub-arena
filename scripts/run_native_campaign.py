"""Run native methods with measured VRAM reservations and independent CPU workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys

from voicehub_arena.native_protocol import write_json,applicable_metrics,canonical_hash,digest
from voicehub_arena.native_resources import read_gpus,select_gpus
from voicehub_arena.native_scheduler import Job,Scheduler


def is_completed(path):
    if not path.exists():return False
    r=json.loads(path.read_text())
    return r.get('status')=='completed' and len(r.get('rows',[]))==r['expected_samples']


def build_jobs(cfg,manifest,publish=False):
    run=Path('runs')/cfg['campaign'];jobs=[]
    active=[s for s in cfg['experiments'] if s.get('verified_api')]
    # Changes to inference/metric code invalidate earlier memory profiles.
    code=canonical_hash({n:digest(Path('voicehub_arena')/n) for n in
        ('native_adapters.py','native_extra.py','native_quality.py','native_eval.py')})
    setups={}
    controls='quality-controls'
    if any(s['uses_reference'] for s in active):
        jobs.append(Job(controls,['.venvs/quality-legacy/bin/python','scripts/validate_native_quality.py',
            '--manifest',str(manifest),'--device','cpu'],'cpu',action='quality_controls'))
    for spec in active:
        backend=spec['backend']
        if backend not in setups:
            key='setup:'+backend;setups[backend]=key
            jobs.append(Job(key,[sys.executable,'scripts/setup_native_model.py','--manifest',str(manifest),
                '--experiment',spec['id']],'setup',action='setup',experiment=spec['id']))
    for spec in active:
        identifier=spec['id'];pilot_end=None
        for phase in ('pilot','full'):
            result=run/phase/identifier/'result.json';previous=None
            if not is_completed(result):
                actions=['generate',*applicable_metrics(spec['uses_reference']),'summarize']
                for action in actions:
                    gpu=action not in ('dnsmos','summarize') and not (
                        action=='generate' and spec.get('settings',{}).get('inference_device')=='cpu')
                    if action=='generate':
                        core=spec['backend'] in ('kokoro','transformers_vits','transformers_speecht5')
                        python='.venvs/native-'+('core' if core else spec['backend'])+'/bin/python'
                    else:python='.venvs/quality-legacy/bin/python' if action in ('utmos22','wavlm_sim') else '.venvs/native-core/bin/python'
                    if action=='summarize':python=sys.executable
                    key=f'{identifier}:{phase}:{action}'
                    dependencies=[previous] if previous else [setups[spec['backend']]]+([pilot_end] if phase=='full' and pilot_end else [])
                    if action=='wavlm_sim':dependencies.append(controls)
                    profile=canonical_hash(dict(action=action,code=code,
                        model=spec if action=='generate' else cfg['asr'] if action=='asr' else action))
                    jobs.append(Job(key,[python,'-u','-m','voicehub_arena.native_eval',action,
                        '--manifest',str(manifest),'--experiment',identifier,'--phase',phase],
                        'gpu' if gpu else 'cpu',identifier,phase,action,dependencies,profile,str(result)))
                    previous=key
                if phase=='pilot':pilot_end=previous
            receipt=run/'publication'/(phase+'--'+identifier+'.json')
            already_published=receipt.exists() and result.exists() and json.loads(receipt.read_text()).get('result_sha256')==digest(result)
            if publish and not already_published:
                key=f'{identifier}:{phase}:publish'
                jobs.append(Job(key,[sys.executable,'scripts/publish_native_progress.py','--manifest',str(manifest),
                    '--experiment',identifier,'--phase',phase],'publish',identifier,phase,'publish',
                    [previous] if previous else []));previous=key
            if publish and phase=='full':
                key=f'{identifier}:{phase}:offload'
                jobs.append(Job(key,[sys.executable,'scripts/offload_native_audio.py','--manifest',str(manifest),
                    '--experiment',identifier,'--phase',phase],'publish',identifier,phase,'offload',
                    [previous] if previous else []))
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='configs/native-methods.json')
    parser.add_argument('--campaign',help='New run name; required when moving to another GPU or execution mode')
    parser.add_argument('--gpus',default='auto',help='Physical indices or GPU UUIDs, limited by CUDA_VISIBLE_DEVICES')
    parser.add_argument('--max-gpu-jobs',type=int,default=4,help='Maximum concurrent workers per GPU; VRAM can impose a lower limit')
    parser.add_argument('--max-cpu-jobs',type=int,default=2)
    parser.add_argument('--headroom-mib',type=int,default=2048,help='Reserve at least this much VRAM and at least 10%% of device capacity')
    parser.add_argument('--mode',choices=['parallel','isolated'],default='parallel')
    parser.add_argument('--publish',action='store_true',help='Publish verified artifacts to the configured HF repos and offload audio')
    parser.add_argument('--dry-run',action='store_true',help='Print the plan; no setup, downloads, generation or state writes')
    args=parser.parse_args()
    if min(args.max_gpu_jobs,args.max_cpu_jobs)<1 or args.headroom_mib<1024:parser.error('Worker limits must be positive; VRAM headroom must be >=1024 MiB')
    cfg=json.loads(Path(args.manifest).read_text())
    if args.campaign:cfg['campaign']=args.campaign
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',cfg['campaign']):parser.error('Invalid campaign name')
    run=Path('runs')/cfg['campaign'];manifest=run/'manifest.json'
    if manifest.exists() and canonical_hash(json.loads(manifest.read_text()))!=canonical_hash(cfg):
        parser.error('Run manifest changed; use its existing --manifest or a new --campaign')
    jobs=build_jobs(cfg,manifest,args.publish)
    try:gpus=select_gpus(read_gpus(),args.gpus)
    except (OSError,ValueError,RuntimeError) as exc:
        if not args.dry_run:raise
        gpus=[];print('GPU telemetry unavailable in plan-only mode: '+str(exc),file=sys.stderr)
    if args.dry_run:
        print(json.dumps(dict(campaign=cfg['campaign'],mode=args.mode,publish=args.publish,
            gpu_devices=[dict(uuid=g.uuid,name=g.name,total_mib=g.total_mib) for g in gpus],
            jobs=[dict(id=j.id,resource=j.resource,dependencies=j.dependencies) for j in jobs]),indent=2));return
    run.mkdir(parents=True,exist_ok=True)
    controller=(run/'controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Held by the controller and inherited by its workers; legacy/manual GPU
    # processes cannot enter concurrently and bypass the memory reservations.
    gpu_lock=Path('runs/.gpu.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    execution=dict(schema=1,mode=args.mode,max_gpu_jobs=args.max_gpu_jobs,max_cpu_jobs=args.max_cpu_jobs,
        devices=[dict(uuid=g.uuid,name=g.name,total_mib=g.total_mib) for g in gpus])
    execution_path=run/'execution-config.json'
    if execution_path.exists():
        if json.loads(execution_path.read_text())!=execution:parser.error('Hardware/concurrency changed; start a new --campaign to keep timing results separate')
    elif any(run.glob('*/**/result.json')):
        parser.error('Existing results have no parallel hardware contract; choose a new --campaign')
    write_json(execution_path,execution);write_json(manifest,cfg)
    scheduler=Scheduler(jobs,run,gpus,gpu_lock.fileno(),args.max_gpu_jobs,args.max_cpu_jobs,args.headroom_mib,mode=args.mode)
    scheduler.state.update(inventory_candidates=len(cfg['experiments']),review_pending=[s['id'] for s in cfg['experiments']
        if not s.get('verified_api') and s.get('availability')!='not_supported_upstream'])
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,scheduler.stop)
    signal.signal(signal.SIGUSR1,scheduler.drain)
    done=scheduler.run_all()
    if done and scheduler.state['review_pending']:scheduler.save('needs_attention')
    if args.publish:
        import subprocess
        subprocess.run([sys.executable,'scripts/publish_native_progress.py','--manifest',str(manifest)],check=True)
    if not done:sys.exit(2)


if __name__=='__main__':main()
