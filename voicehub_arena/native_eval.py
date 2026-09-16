"""Resumable native workers; experiment locks and scheduler-owned GPU reservations."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

from .native_protocol import (forbid_voicehub, digest, load_experiment, write_json,
                              verify_rows, applicable_metrics, metric_summary, canonical_hash)


def memory_usage(result,action):
    """Report allocator peaks as well as the scheduler's non-PyTorch telemetry."""
    import torch
    if torch.cuda.is_available():
        usage=result.setdefault('resource_usage',{}).setdefault(action,{})
        usage['torch_reserved_peak_mib']=max(usage.get('torch_reserved_peak_mib',0),
                                           int(torch.cuda.max_memory_reserved()/2**20))


def execution_context():
    path=os.environ.get('VOICEHUB_EXECUTION_CONTEXT')
    return json.loads(Path(path).read_text()) if path else {'mode':'isolated'}


def lock_worker(directory,gpu):
    import fcntl
    lock=None
    if gpu:
        inherited=os.environ.get('VOICEHUB_SCHEDULER_LOCK_FD')
        if inherited is not None:
            expected=Path('runs/.gpu.lock').stat();actual=os.fstat(int(inherited))
            if (expected.st_dev,expected.st_ino)!=(actual.st_dev,actual.st_ino):
                raise ValueError('Invalid scheduler GPU lock descriptor')
        else:
            lock=Path('runs/.gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    experiment=(directory/'worker.lock').open('a')
    try:fcntl.flock(experiment,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        experiment.close()
        if lock:lock.close()
        raise
    return experiment,lock


def generation(spec, cfg, selected, directory, result):
    import numpy as np
    import soundfile as sf
    import torch
    from .native_adapters import load
    from .metrics import signal_metrics
    complete = {r['id'] for r in result['rows']}
    if len(complete) == len(selected):
        return
    cuda=spec.get('settings',{}).get('inference_device')!='cpu'
    torch.set_num_threads(4)
    def seed():
        random.seed(cfg['seed']); np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        if cuda:torch.cuda.manual_seed_all(cfg['seed'])
    seed()
    packages = {p: importlib.metadata.version(p) for p in
                          ('torch','transformers','numpy','soundfile','huggingface-hub')}
    provenance = dict(packages=packages,execution=execution_context(), implementation_sha256={name:digest(Path(__file__).with_name(name))
        for name in ('native_adapters.py','native_extra.py','native_eval.py')})
    runtime=Path(os.sys.executable).parent.parent/'arena-runtime.json'
    if runtime.exists():provenance['runtime']=json.loads(runtime.read_text())
    provenance_id=canonical_hash(provenance)
    result.setdefault('generation_provenance',{})[provenance_id]=provenance
    result.setdefault('packages',packages)
    result['status'] = 'loading'; write_json(directory/'result.json',result)
    generate = load(spec)
    # A separate warmup never becomes a measured sample or a quality-based retry.
    with torch.inference_mode():
        for _ in generate(selected[0]):
            pass
    if cuda:memory_usage(result,'generate')
    result['status'] = 'generating'
    for item in selected:
        if item['id'] in complete:
            continue
        if shutil.disk_usage(directory).free < 2 * 2**30:
            raise RuntimeError('2 GiB output reserve reached')
        if spec['uses_reference'] and digest(item['reference_audio']) != item['reference_audio_sha256']:
            raise ValueError('Conditioning reference digest mismatch')
        seed()
        if cuda:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start = time.perf_counter(); chunks=[]; first=None; sr=None
        with torch.inference_mode():
            for chunk, rate in generate(item):
                if hasattr(chunk, 'detach'):
                    chunk = chunk.detach().float().cpu().numpy()
                chunk = np.asarray(chunk, dtype=np.float32).squeeze()
                if chunk.size == 0:
                    continue
                if chunk.ndim != 1 or not np.isfinite(chunk).all() or rate <= 0:
                    raise ValueError('Upstream adapter returned invalid audio')
                if sr is not None and rate != sr:
                    raise ValueError('Streaming sample rate changed')
                sr = int(rate)
                if cuda:torch.cuda.synchronize()
                if first is None:
                    first = time.perf_counter()-start
                chunks.append(chunk)
        if cuda:torch.cuda.synchronize()
        latency = time.perf_counter()-start
        row = {**item, 'generation_status':'ok' if chunks else 'no_audio',
               'seed':cfg['seed'], 'generation_provenance_id':provenance_id, 'latency_s':latency,
               'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20 if cuda else 0,
               'timing_scope':execution_context()['mode'],
               'ttfa_s':first if spec['streaming'] else None,
               'chunks':len(chunks), 'scores':{}}
        if chunks:
            wave = np.concatenate(chunks)
            row.update(signal_metrics(wave,sr))
            row['audio']='audio/'+item['id']+'.wav'
            target=directory/row['audio'];target.parent.mkdir(exist_ok=True)
            sf.write(target,wave,sr,subtype='FLOAT')
            row['audio_sha256']=digest(target);row['rtf']=latency/row['duration_s']
        else:
            row['failure_policy']='empty-output-deletions-v1'
        if not spec['uses_reference']:
            row['scores']['wavlm_sim']={'status':'not_applicable','reason':'No conditioning speaker audio in this generation method'}
        result['rows'].append(row);result['updated_at']=time.time()
        if cuda:memory_usage(result,'generate')
        write_json(directory/'result.json',result)
        print(json.dumps({'experiment':spec['id'],'action':'generate','done':len(result['rows']),
                          'total':len(selected),'latency_s':round(latency,3)}),flush=True)


def scoring(action,spec,cfg,directory,result):
    from .native_quality import make_scorer
    pending=[r for r in result['rows'] if action not in r['scores']]
    if not pending:
        return
    if action=='wavlm_sim':
        control=json.loads((Path('runs')/cfg['campaign']/'quality-controls.json').read_text())
        manifest=json.loads(Path('artifacts/native-metrics/manifest.json').read_text())['wavlm_sim']
        if control.get('status')!='passed' or any(control['provenance'][k]!=manifest[k] for k in ('source_revision','files_sha256')):
            raise ValueError('Run matching WavLM identity controls before scoring TTS')
    scorer,provenance=make_scorer(action,cfg)
    provenance={**provenance,'execution':execution_context()}
    result.setdefault('metric_provenance',{})[action]=provenance
    result['status']='scoring_'+action
    for row in pending:
        if row['generation_status']=='no_audio':
            if action=='asr':
                from .metrics import errors
                row['scores']['asr']={'status':'ok','transcript':'',
                    **errors([row['reference']],[''],normalization='whisper_english'),
                    'failure_policy':'empty-output-deletions-v1'}
            else:
                row['scores'][action]={'status':'not_applicable','reason':'No generated audio; generation failure retained'}
        else:
            audio=directory/row['audio']
            if digest(audio)!=row['audio_sha256']:
                raise ValueError('Audio digest changed before '+action)
            row['scores'][action]={'status':'ok',**scorer(audio,row)}
        result['updated_at']=time.time();write_json(directory/'result.json',result)
        print(json.dumps({'experiment':spec['id'],'action':action,
                          'done':sum(action in r['scores'] for r in result['rows']),
                          'total':result['expected_samples']}),flush=True)


def summarize_native(result,spec):
    import numpy as np
    from .metrics import errors,summarize
    rows=result['rows'];good=[r for r in rows if r['generation_status']=='ok']
    summary={'evaluated':len(rows),'generated':len(good),'generation_failures':len(rows)-len(good),
             'expected_samples':result['expected_samples']}
    # Preserve old immutable summaries when historical rows have no timing label.
    if any('timing_scope' in r for r in rows):summary['timing_scopes']=sorted({r.get('timing_scope','isolated') for r in rows})
    asr=[r for r in rows if r['scores'].get('asr',{}).get('status')=='ok']
    if asr:
        summary.update(summarize([{**r,'status':'ok' if r['generation_status']=='ok' else 'generation_failed_scored',
            'normalization_id':'whisper_english','category':r.get('category','read_speech'),
            'transcript':r['scores']['asr']['transcript']} for r in asr],
            scored_statuses=('ok','generation_failed_scored')))
        summary['asr_scored']=len(asr)
    for metric,fields in {'dnsmos':['sig','bak','ovrl','p808'],
                          'utmos22':['mos'],'wavlm_sim':['similarity']}.items():
        for field in fields:
            summary[metric+'_'+field]=metric_summary(rows,metric,field)
    if good:
        summary.update(rtf=sum(r['latency_s'] for r in good)/sum(r['duration_s'] for r in good),
                       latency_p50_s=float(np.percentile([r['latency_s'] for r in good],50)),
                       latency_p95_s=float(np.percentile([r['latency_s'] for r in good],95)),
                       peak_vram_mib=max(r['peak_vram_mib'] for r in good))
        for metric in ('silence_ratio','clipping_ratio','rms_dbfs'):
            summary[metric]=float(np.mean([r[metric] for r in good]))
        ttfa=[r['ttfa_s'] for r in good if r.get('ttfa_s') is not None]
        if ttfa:summary['ttfa_p50_s']=float(np.median(ttfa))
    result['summary']=summary
    required=applicable_metrics(spec['uses_reference'])
    def complete_row(row):
        for metric in required:
            expected='not_applicable' if row['generation_status']=='no_audio' and metric!='asr' else 'ok'
            if row['scores'].get(metric,{}).get('status')!=expected:return False
        return True
    result['status']='completed' if len(rows)==result['expected_samples'] and all(
        complete_row(r) for r in rows) else 'partial'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','asr','dnsmos','utmos22','wavlm_sim','summarize'])
    p.add_argument('--manifest',default='configs/native-methods.json');p.add_argument('--experiment',required=True)
    p.add_argument('--phase',choices=['pilot','full'],default='pilot');args=p.parse_args()
    forbid_voicehub()
    Path('runs').mkdir(exist_ok=True)
    os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    cfg,spec,selected,directory,contract_hash=load_experiment(args.manifest,args.experiment,args.phase)
    gpu=args.action not in ('dnsmos','summarize') and not (args.action=='generate' and spec.get('settings',{}).get('inference_device')=='cpu')
    worker_lock,gpu_lock=lock_worker(directory,gpu)
    if args.action!='summarize':
        import torch
        torch.set_num_threads(4)
    path=directory/'result.json'
    result=json.loads(path.read_text()) if path.exists() else dict(schema_version=1,
        experiment=spec['id'],family=spec['family'],method=spec['method'],streaming=spec['streaming'],
        uses_reference=spec['uses_reference'],repo=spec['repo'],revision=spec['revision'],
        phase=args.phase,contract_sha256=contract_hash,expected_samples=len(selected),rows=[],started_at=time.time())
    if result['contract_sha256']!=contract_hash:raise ValueError('Experiment contract changed')
    verify_rows(result['rows'],selected,directory)
    if result.get('error'):
        result.setdefault('previous_errors',[]).append(result.pop('error'))
    result.pop('error_kind',None)
    try:
        if args.action=='generate':generation(spec,cfg,selected,directory,result)
        elif args.action=='summarize':
            summarize_native(result,spec)
            if result['status']!='completed':sys.exit(2)
        else:scoring(args.action,spec,cfg,directory,result)
    except Exception as exc:
        message=type(exc).__name__+': '+str(exc)
        if 'cuda' in message.lower() and any(x in message.lower() for x in ('out of memory','outofmemory','memory allocation','memoryallocation')):
            result['error_kind']='cuda_oom'
        result.update(status='needs_attention',error=message);traceback.print_exc();raise
    finally:
        if gpu:memory_usage(result,args.action)
        result['updated_at']=time.time();write_json(path,result)


if __name__=='__main__':main()
