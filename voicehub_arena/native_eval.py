"""Resumable native generation and independent scoring, one GPU process at a time."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from .native_protocol import (forbid_voicehub, digest, load_experiment, write_json,
                              verify_rows, applicable_metrics, metric_summary, canonical_hash)


def generation(spec, cfg, selected, directory, result):
    import numpy as np
    import soundfile as sf
    import torch
    from .native_adapters import load
    from .metrics import signal_metrics
    complete = {r['id'] for r in result['rows']}
    if len(complete) == len(selected):
        return
    torch.set_num_threads(4)
    def seed():
        random.seed(cfg['seed']); np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    seed()
    packages = {p: importlib.metadata.version(p) for p in
                          ('torch','transformers','numpy','soundfile','huggingface-hub')}
    provenance = dict(packages=packages, implementation_sha256={name:digest(Path(__file__).with_name(name))
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
    result['status'] = 'generating'
    for item in selected:
        if item['id'] in complete:
            continue
        if shutil.disk_usage(directory).free < 2 * 2**30:
            raise RuntimeError('2 GiB output reserve reached')
        if spec['uses_reference'] and digest(item['reference_audio']) != item['reference_audio_sha256']:
            raise ValueError('Conditioning reference digest mismatch')
        seed(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
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
                torch.cuda.synchronize()
                if first is None:
                    first = time.perf_counter()-start
                chunks.append(chunk)
        torch.cuda.synchronize(); latency = time.perf_counter()-start
        row = {**item, 'generation_status':'ok' if chunks else 'no_audio',
               'seed':cfg['seed'], 'generation_provenance_id':provenance_id, 'latency_s':latency,
               'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,
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
    import fcntl
    Path('runs').mkdir(exist_ok=True)
    gpu_lock=Path('runs/.gpu.lock').open('a')
    if args.action!='summarize':fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    cfg,spec,selected,directory,contract_hash=load_experiment(args.manifest,args.experiment,args.phase)
    path=directory/'result.json'
    result=json.loads(path.read_text()) if path.exists() else dict(schema_version=1,
        experiment=spec['id'],family=spec['family'],method=spec['method'],streaming=spec['streaming'],
        uses_reference=spec['uses_reference'],repo=spec['repo'],revision=spec['revision'],
        phase=args.phase,contract_sha256=contract_hash,expected_samples=len(selected),rows=[],started_at=time.time())
    if result['contract_sha256']!=contract_hash:raise ValueError('Experiment contract changed')
    verify_rows(result['rows'],selected,directory)
    if result.get('error'):
        result.setdefault('previous_errors',[]).append(result.pop('error'))
    try:
        if args.action=='generate':generation(spec,cfg,selected,directory,result)
        elif args.action=='summarize':summarize_native(result,spec)
        else:scoring(args.action,spec,cfg,directory,result)
    except Exception as exc:
        result.update(status='needs_attention',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise
    finally:
        result['updated_at']=time.time();write_json(path,result)


if __name__=='__main__':main()
