"""Publish only verified complete variant artifacts; keep pilots separate from full scores."""
from pathlib import Path
import argparse
import os
import copy
import csv
import hashlib
import json
import math
import re
import tarfile
import time

os.environ['HF_HOME'] = str(Path.cwd()/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import requests
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.metrics import errors
from voicehub_arena.storage import write_json
from voicehub_arena.variant_scope import load_selection,require_active
from voicehub_arena.variant_eval import load_campaign,digest,SCORED_STATUSES,validate_scored_row,NO_AUDIO_POLICY

DATASET='kadirnar/voicehub-arena-seed-tts-eval';SPACE='kadirnar/voicehub-arena'


def count_recordings(table):
    # Frozen original rows were independently verified complete but predate the generated field.
    return sum(row.get('generated', row['scored']) for row in table)


def publish(manifest='configs/variant-campaign.json',model=None,phase='pilot'):
    configure_hub_auth();cfg,dataset=load_campaign(manifest);run=Path('runs')/cfg['campaign'];run.mkdir(parents=True,exist_ok=True)
    selection,models=load_selection(cfg,manifest)
    if model:require_active(model.removesuffix('-unconditioned'),cfg,manifest)
    api=HfApi();assert api.whoami()['name']=='kadirnar'
    receipts=run/'publication';receipts.mkdir(exist_ok=True)
    if model:
        source=run/phase/model/'result.json';result=json.loads(source.read_text())
        selected=dataset if phase=='full' else [dataset[i] for i in cfg['pilot_indices']]
        assert result['status']=='completed' and len(result['rows'])==len(selected) and all(r['status'] in SCORED_STATUSES for r in result['rows'])
        for row in result['rows']:validate_scored_row(row)
        audio_rows=[r for r in result['rows'] if r['status']=='ok']
        failures=len(selected)-len(audio_rows)
        assert result['summary']['generated']==len(audio_rows)
        by_id={r['id']:r for r in result['rows']};assert len(by_id)==len(selected)
        assert set(by_id)=={r['id'] for r in selected}
        for item in selected:assert by_id[item['id']]['text']==item['text'] and by_id[item['id']]['reference']==item['reference']
        actual=errors([r['reference'] for r in result['rows']],[r['transcript'] for r in result['rows']],normalization='whisper_english')
        for key,value in actual.items():assert math.isclose(value,result['summary'][key],abs_tol=1e-12),key
        receipt=receipts/(phase+'--'+model+'.json')
        if not receipt.exists() or json.loads(receipt.read_text())['result_sha256']!=digest(source):
            out=run/'exports'/phase/model;out.mkdir(parents=True,exist_ok=True)
            archive=out/'audio.tar';records=copy.deepcopy(result)
            prefix=f'experiments/{cfg["campaign"]}/{phase}/{model}'
            with tarfile.open(archive,'w',format=tarfile.PAX_FORMAT) as tar:
                for row in records['rows']:
                    if row['status']!='ok':continue
                    audio=run/phase/row['audio'];assert digest(audio)==row['audio_sha256']
                    info=tarfile.TarInfo(audio.name);info.size=audio.stat().st_size;info.mode=0o644
                    offset=tar.offset+len(info.tobuf(format=tarfile.PAX_FORMAT))
                    with audio.open('rb') as f:tar.addfile(info,f)
                    row.update(audio_archive=prefix+'/audio.tar',audio_offset=offset,audio_bytes=info.size,audio_path=audio.name)
            with archive.open('rb') as f:
                for row in records['rows']:
                    if row['status']!='ok':continue
                    f.seek(row['audio_offset']);assert hashlib.sha256(f.read(row['audio_bytes'])).hexdigest()==row['audio_sha256']
            with tarfile.open(archive) as tar:assert len(tar.getmembers())==len(audio_rows)
            verification=dict(expected=len(selected),generated=len(audio_rows),generation_failures=failures,failure_policy=NO_AUDIO_POLICY,scored=len(selected),unique_ids=len(selected),all_targets_match=True,all_audio_sha256_verified=True,all_tar_offsets_verified=True,corpus_metrics_recomputed=actual,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,result_sha256=digest(source),config_sha256=digest(manifest))
            write_json(out/'records.json',records);write_json(out/'verification.json',verification)
            # Public artifact metadata is reviewed structured run data, never environment dumps.
            for path in [source,Path(manifest),out/'records.json',out/'verification.json']:
                assert not re.search(r'(?:hf_|gh[pousr]_)[A-Za-z0-9]{15,}',path.read_text())
            parent=api.repo_info(DATASET,repo_type='dataset').sha
            commit=api.create_commit(DATASET,repo_type='dataset',parent_commit=parent,commit_message=f'Verified {phase} {model}: {len(selected)} Seed-TTS-Eval texts',operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=file) for name,file in [('audio.tar',archive),('result.json',source),('records.json',out/'records.json'),('verification.json',out/'verification.json'),('config.json',Path(manifest))]])
            # Verify a real pinned byte-range before publishing browser playback metadata.
            row=next((r for r in records['rows'] if r['status']=='ok'),None)
            if row:
                response=requests.get(f'https://huggingface.co/datasets/{DATASET}/resolve/{commit.oid}/{row["audio_archive"]}',headers={'Range':f'bytes={row["audio_offset"]}-{row["audio_offset"]+row["audio_bytes"]-1}'},timeout=90)
                assert response.status_code==206 and hashlib.sha256(response.content).hexdigest()==row['audio_sha256']
            for row in records['rows']:row['audio_dataset_revision']=commit.oid;row['audio_dataset_id']=DATASET
            browser_file=out/'browser-records.json';write_json(browser_file,records)
            info=api.create_commit(SPACE,repo_type='space',parent_commit=api.repo_info(SPACE,repo_type='space').sha,commit_message=f'Add verified {phase} samples for {model}',operations=[CommitOperationAdd(path_in_repo=f'data/variants/{phase}/{model}.json',path_or_fileobj=browser_file)])
            write_json(receipt,dict(model=model,phase=phase,result_sha256=digest(source),dataset_revision=commit.oid,space_revision=info.oid,records_path=f'data/variants/{phase}/{model}.json',artifact_prefix=prefix,verification=verification))
            print(json.dumps({'published':model,'phase':phase,'samples':len(selected),'dataset_revision':commit.oid}),flush=True)
    progress={'campaign':cfg['campaign'],'updated_at':time.time(),'scope':selection['scope'],'active_model_ids':selection['active_model_ids'],'excluded_base_model_ids':selection['excluded_base_model_ids'],'expected_samples':cfg['expected_samples'],'pilot_indices':cfg['pilot_indices'],'pilot_selection':cfg['pilot_selection'],'protocol':'Fixed English Emily reference. Seed 42. All full runs target 1,088 exact texts. Whisper-large-v3, no VAD, corpus WER/CER. These conditioning experiments do not replace the original unconditioned Dia/Llasa scores.','models':[]}
    for spec in models:
        entry={k:spec[k] for k in ('id','name','repo','revision','family')}
        for ph in ['pilot','full']:
            p=run/ph/spec['id']/'result.json';r=json.loads(p.read_text()) if p.exists() else {};receipt=receipts/(ph+'--'+spec['id']+'.json');published=json.loads(receipt.read_text()) if receipt.exists() else None
            valid=bool(published and p.exists() and published['result_sha256']==digest(p))
            entry[ph]={'status':r.get('status','queued'),'generated':sum(x.get('status') in ['generated','ok'] for x in r.get('rows',[])),'scored':sum(x.get('status') in SCORED_STATUSES for x in r.get('rows',[])),'expected':1088 if ph=='full' else len(cfg['pilot_indices']),'published':valid,'generation_failures':sum(x.get('status') in ['generation_failed','generation_failed_scored'] for x in r.get('rows',[]))}
            if r.get('error'):entry[ph]['error']=r['error']
            if valid:entry[ph].update(metrics=r['summary'],records_path=published['records_path'],dataset_revision=published['dataset_revision'],artifact_prefix=published['artifact_prefix'])
        p=run/'pilot'/(spec['id']+'-unconditioned')/'result.json';receipt=receipts/('pilot--'+spec['id']+'-unconditioned.json')
        if p.exists() and receipt.exists():
            r=json.loads(p.read_text());rec=json.loads(receipt.read_text())
            if rec['result_sha256']==digest(p):entry['unconditioned_control']={'metrics':r['summary'],'records_path':rec['records_path'],'samples':len(r['rows'])}
        progress['models'].append(entry)
    base_path=hf_hub_download(DATASET,'leaderboard.json',repo_type='dataset',revision='5c3ff83af71a0de21b8fc00c2953157070afa9a5')
    assert digest(base_path)=='0de7dd4c3af30aead1f97e51c00cd7970a81b62f2a1d52fe65cef35609690dcc'
    combined=json.loads(Path(base_path).read_text())
    combined['table']=[r for r in combined['table'] if r['model'] not in selection['excluded_base_model_ids']]
    for row in combined['table']:row['protocol']='Original provider configuration'
    for spec in progress['models']:
        full=spec['full']
        if full['published'] and full['scored']==1088:
            combined['table'].append({**full['metrics'],'model':spec['id'],'name':spec['name']+' · ref','checkpoint':spec['repo'],'revision':spec['revision'],'protocol':'Fixed reference · independent implementation','experiment':'reference-conditioned','records_path':full['records_path']})
    assert len({r['model'] for r in combined['table']})==len(combined['table'])
    combined.update(models=len(combined['table']),scored_audio=count_recordings(combined['table']),base_snapshot_sha256=digest(base_path),variant_campaign=cfg['campaign'],active_model_ids=selection['active_model_ids'],excluded_base_model_ids=selection['excluded_base_model_ids'])
    comparison=run/'current-comparison.json';write_json(comparison,combined)
    csv_path=run/'current-comparison.csv';fields=['model','name','protocol','checkpoint','revision','scored','generated','generation_failures','generation_failure_rate','wer','cer','rtf','latency_p50_s','latency_p95_s','peak_vram_mib','mer','wil','wip','exact_match_rate']
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(combined['table'])
    p=run/'public-progress.json';write_json(p,progress)
    api.create_commit(DATASET,repo_type='dataset',parent_commit=api.repo_info(DATASET,repo_type='dataset').sha,commit_message='Update variant experiment progress',operations=[CommitOperationAdd(path_in_repo=f'experiments/{cfg["campaign"]}/selection.json',path_or_fileobj=Path(manifest).parent/'variant-selection.json'),CommitOperationAdd(path_in_repo=f'experiments/{cfg["campaign"]}/progress.json',path_or_fileobj=p),CommitOperationAdd(path_in_repo=f'experiments/{cfg["campaign"]}/current-comparison.json',path_or_fileobj=comparison),CommitOperationAdd(path_in_repo=f'experiments/{cfg["campaign"]}/current-comparison.csv',path_or_fileobj=csv_path)])
    api.create_commit(SPACE,repo_type='space',parent_commit=api.repo_info(SPACE,repo_type='space').sha,commit_message='Update Dia2 and Llasa variant progress',operations=[CommitOperationAdd(path_in_repo='data/variant-selection.json',path_or_fileobj=Path(manifest).parent/'variant-selection.json'),CommitOperationAdd(path_in_repo='data/variant-progress.json',path_or_fileobj=p),CommitOperationAdd(path_in_repo='data/current-comparison.json',path_or_fileobj=comparison),CommitOperationAdd(path_in_repo='data/current-comparison.csv',path_or_fileobj=csv_path)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--phase',choices=['pilot','full'],default='pilot');args=p.parse_args();publish(model=args.model,phase=args.phase)
