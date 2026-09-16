"""Verified native artifacts and separate Space progress; historical tables are untouched."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tarfile
import time

from voicehub_arena.native_protocol import digest,load_experiment,verify_rows,write_json
from voicehub_arena.native_eval import summarize_native

DATASET='kadirnar/voicehub-arena-seed-tts-eval'
SPACE='kadirnar/voicehub-arena'


def verify_remote(api,repo,kind,revision,files):
    items={i.path:i for i in api.get_paths_info(repo,repo_type=kind,revision=revision,paths=list(files))}
    for name,path in files.items():
        item=items[name];path=Path(path)
        assert item.size==path.stat().st_size,name
        if item.lfs:assert item.lfs.sha256==digest(path),name
        else:
            raw=path.read_bytes()
            assert item.blob_id==hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),name


def commit(api,repo,kind,files,message):
    from huggingface_hub import CommitOperationAdd
    for name,path in files.items():
        if Path(path).suffix in ('.json','.csv','.py','.html','.js','.md'):
            assert not re.search(rb'(?:hf_|gh[pousr]_)[A-Za-z0-9]{15,}',Path(path).read_bytes()),name
    response=api.create_commit(repo,repo_type=kind,parent_commit=api.repo_info(repo,repo_type=kind).sha,
        commit_message=message,operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=p) for n,p in files.items()])
    verify_remote(api,repo,kind,response.oid,files)
    return response.oid


def publish_artifact(api,identifier,phase):
    import requests
    cfg,spec,selected,directory,contract=load_experiment('configs/native-methods.json',identifier,phase)
    source=directory/'result.json';result=json.loads(source.read_text())
    verify_rows(result['rows'],selected,directory)
    assert {r['id'] for r in result['rows']}=={r['id'] for r in selected}
    assert result['contract_sha256']==contract
    check=summarize_native(copy.deepcopy(result),spec)
    assert check['status']=='completed' and check['summary']==result['summary']
    run=Path('runs')/cfg['campaign'];receipt=run/'publication'/(phase+'--'+identifier+'.json')
    if receipt.exists() and json.loads(receipt.read_text())['result_sha256']==digest(source):return
    out=run/'exports'/phase/identifier;out.mkdir(parents=True,exist_ok=True)
    archive=out/'audio.tar';records=copy.deepcopy(result)
    prefix=f'experiments/{cfg["campaign"]}/{phase}/{identifier}'
    with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT) as tar:
        for row in records['rows']:
            if row['generation_status']!='ok':continue
            audio=directory/row['audio'];info=tarfile.TarInfo(audio.name)
            info.size=audio.stat().st_size;info.mode=0o644
            offset=tar.offset+len(info.tobuf(format=tarfile.PAX_FORMAT))
            with audio.open('rb') as f:tar.addfile(info,f)
            row.update(audio_archive=prefix+'/audio.tar',audio_offset=offset,audio_bytes=info.size)
    with archive.open('rb') as f:
        for row in records['rows']:
            if row['generation_status']!='ok':continue
            f.seek(row['audio_offset']);assert hashlib.sha256(f.read(row['audio_bytes'])).hexdigest()==row['audio_sha256']
    verification=dict(expected=len(selected),generated=sum(r['generation_status']=='ok' for r in records['rows']),
        exact_target_coverage=True,all_audio_and_offsets_verified=True,metrics_recomputed=True,
        contract_sha256=contract,result_sha256=digest(source),archive_sha256=digest(archive))
    write_json(out/'records.json',records);write_json(out/'verification.json',verification)
    files={prefix+'/'+n:p for n,p in [('audio.tar',archive),('result.json',source),
        ('records.json',out/'records.json'),('verification.json',out/'verification.json'),
        ('contract.json',directory/'contract.json')]}
    revision=commit(api,DATASET,'dataset',files,f'Native {phase}: {identifier}, {len(selected)} verified targets')
    first=next((r for r in records['rows'] if r['generation_status']=='ok'),None)
    if first:
        response=requests.get(f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{first["audio_archive"]}',
            headers={'Range':f'bytes={first["audio_offset"]}-{first["audio_offset"]+first["audio_bytes"]-1}'},timeout=90)
        assert response.status_code==206 and hashlib.sha256(response.content).hexdigest()==first['audio_sha256']
    for row in records['rows']:row.update(audio_dataset_id=DATASET,audio_dataset_revision=revision)
    browser=out/'browser-records.json';write_json(browser,records)
    location=f'data/native/{phase}/{identifier}.json'
    space_revision=commit(api,SPACE,'space',{location:browser},f'Native {phase} audio and metrics: {identifier}')
    write_json(receipt,dict(experiment=identifier,phase=phase,result_sha256=digest(source),
                           dataset_revision=revision,space_revision=space_revision,
                           records_path=location,verification=verification))


def publish_progress(api):
    cfg=json.loads(Path('configs/native-methods.json').read_text());run=Path('runs')/cfg['campaign']
    controller=run/'campaign-status.json'
    events=json.loads(controller.read_text()).get('events',[]) if controller.exists() else []
    progress=dict(campaign=cfg['campaign'],updated_at=time.time(),scope=cfg['scope'],
        dataset='Seed-TTS-Eval English',expected_samples=cfg['expected_samples'],asr=cfg['asr'],
        metrics=cfg['metrics'],pilot_indices=cfg['pilot_indices'],experiments=[])
    for spec in cfg['experiments']:
        entry={k:spec[k] for k in ('id','family','method','streaming','repo','revision','source','uses_reference','implementation_status')}
        entry['availability']=spec.get('availability','supported' if spec.get('verified_api') else 'under_review')
        entry['method_note']=spec.get('method_note')
        for phase in ('pilot','full'):
            p=run/phase/spec['id']/'result.json';r=json.loads(p.read_text()) if p.exists() else {}
            receipt=run/'publication'/(phase+'--'+spec['id']+'.json')
            saved=json.loads(receipt.read_text()) if receipt.exists() else None
            published=bool(saved and p.exists() and saved['result_sha256']==digest(p))
            entry[phase]=dict(status=r.get('status','pending'),attempted=len(r.get('rows',[])),
                generated=sum(x['generation_status']=='ok' for x in r.get('rows',[])),
                expected=1088 if phase=='full' else 8,published=published,
                metric_counts={m:sum(x['scores'].get(m,{}).get('status')=='ok' for x in r.get('rows',[]))
                               for m in ('asr','dnsmos','utmos22','wavlm_sim')})
            if published:entry[phase].update(summary=r['summary'],records_path=saved['records_path'],dataset_revision=saved['dataset_revision'])
            if r.get('error'):entry[phase]['error']=r['error']
            if not r:
                setup_events=[e for e in events if e.get('experiment')==spec['id'] and e.get('phase')==phase and e.get('action')=='setup']
                if setup_events and setup_events[-1].get('error'):
                    entry[phase].update(status='needs_attention',error=setup_events[-1]['error'])
        progress['experiments'].append(entry)
    path=run/'public-progress.json';write_json(path,progress)
    csv_path=run/'native-comparison.csv'
    fields=['experiment','family','method','streaming','repo','phase','evaluated','generated','generation_failures',
            'wer','cer','wer_ci95','cer_ci95','raw_wer','raw_cer','mer','wil','wip','exact_match',
            'rtf','latency_p50_s','latency_p95_s','ttfa_p50_s','peak_vram_mib','silence_ratio','clipping_ratio','rms_dbfs',
            'dnsmos_sig','dnsmos_bak','dnsmos_ovrl','dnsmos_p808','utmos22_mos','wavlm_sim_similarity']
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader()
        for entry in progress['experiments']:
            for phase in ('pilot','full'):
                if not entry[phase]['published']:continue
                row=dict(entry,experiment=entry['id'],phase=phase,**entry[phase]['summary'])
                for k,v in list(row.items()):
                    if isinstance(v,dict) and 'value' in v:row[k]=v['value']
                writer.writerow(row)
    dataset_files={f'experiments/{cfg["campaign"]}/progress.json':path,
        f'experiments/{cfg["campaign"]}/native-comparison.csv':csv_path,
        f'experiments/{cfg["campaign"]}/inventory.json':Path('configs/native-methods.json')}
    space_files={'data/native-progress.json':path,'data/native-comparison.csv':csv_path}
    for name in ('quality-controls.json','dnsmos-thread-controls.json'):
        control=run/name
        if control.exists() and json.loads(control.read_text()).get('status')=='passed':
            dataset_files[f'experiments/{cfg["campaign"]}/'+name]=control
            space_files['data/native-'+name]=control
    commit(api,DATASET,'dataset',dataset_files,'Native method campaign progress and coverage')
    commit(api,SPACE,'space',space_files,'Native method campaign progress and metrics')
    print(json.dumps({'experiments':len(progress['experiments']),'full_published':sum(s['full']['published'] for s in progress['experiments'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment');p.add_argument('--phase',default='pilot',choices=['pilot','full']);a=p.parse_args()
    os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    from huggingface_hub import HfApi
    api=HfApi();assert api.whoami()['name']=='kadirnar'
    if a.experiment:publish_artifact(api,a.experiment,a.phase)
    publish_progress(api)
