"""Reclaim only native audio whose immutable HF archive has been verified.

The original per-row hashes and remote receipt remain locally. Historical results,
incomplete experiments, model weights and user files are never removed here.
"""
import argparse
import json
from pathlib import Path
import hashlib
import shutil
import tarfile
import tempfile
import time

from voicehub_arena.native_protocol import digest,write_json


def release_completed_staging(cfg,identifier):
    spec=next(s for s in cfg['experiments'] if s['id']==identifier)
    group=[s for s in cfg['experiments'] if s.get('verified_api') and (s['repo'],s['revision'])==(spec['repo'],spec['revision'])]
    run=Path('runs')/cfg['campaign']
    for sibling in group:
        result=run/'full'/sibling['id']/'result.json'
        receipt=run/'publication'/('full--'+sibling['id']+'.json')
        if not result.exists() or not receipt.exists():return
        if json.loads(result.read_text()).get('status')!='completed':return
        if digest(result)!=json.loads(receipt.read_text())['result_sha256']:return
    key=hashlib.sha256((spec['repo']+'@'+spec['revision']).encode()).hexdigest()[:20]
    cache=Path('.cache/native-staging')/key
    if not cache.exists():return
    if cache.is_symlink():raise ValueError('Refusing unexpected staging symlink')
    # Only this campaign's reproducible download cache. Shared/historical caches
    # and codec caches are outside this operation.
    receipt=dict(repo=spec['repo'],revision=spec['revision'],path=str(cache),time=time.time(),reason='All audited methods for checkpoint are fully published')
    with (run/'native-staging-evictions.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
    shutil.rmtree(cache)


def offload(identifier,phase,manifest='configs/native-methods.json'):
    from huggingface_hub import HfApi
    from scripts.publish_native_progress import DATASET,verify_remote
    cfg=json.loads(Path(manifest).read_text())
    if identifier not in {s['id'] for s in cfg['experiments']}:raise ValueError('Unknown experiment')
    run=Path('runs')/cfg['campaign'];directory=run/phase/identifier
    receipt=json.loads((run/'publication'/(phase+'--'+identifier+'.json')).read_text())
    result_path=directory/'result.json';result=json.loads(result_path.read_text())
    if result['status']!='completed' or digest(result_path)!=receipt['result_sha256']:
        raise ValueError('Only unchanged, fully published native results can be offloaded')
    archive=run/'exports'/phase/identifier/'audio.tar'
    marker=directory/'audio-storage.json'
    if marker.exists() and not archive.exists():
        if phase=='full':release_completed_staging(cfg,identifier)
        return
    if digest(archive)!=receipt['verification']['archive_sha256']:raise ValueError('Archive changed')
    prefix=f'experiments/{cfg["campaign"]}/{phase}/{identifier}'
    verify_remote(HfApi(),DATASET,'dataset',receipt['dataset_revision'],
                  {prefix+'/audio.tar':archive,prefix+'/result.json':result_path})
    # Complete all checks before removing any recoverable local copy.
    paths=[]
    for row in result['rows']:
        if row['generation_status']!='ok':continue
        path=directory/row['audio']
        if path.parent!=directory/'audio' or (path.exists() and digest(path)!=row['audio_sha256']):
            raise ValueError('Local audio digest/path mismatch')
        if path.exists():paths.append(path)
    write_json(marker,dict(storage='immutable_hf_archive',dataset=DATASET,
        revision=receipt['dataset_revision'],archive=prefix+'/audio.tar',
        archive_sha256=receipt['verification']['archive_sha256'],result_sha256=digest(result_path),
        restore='Download this pinned archive and extract only the WAV basenames in result.json; verify every audio_sha256.'))
    freed=sum(p.stat().st_size for p in paths)+archive.stat().st_size
    for path in paths:path.unlink()
    archive.unlink()
    if phase=='full':release_completed_staging(cfg,identifier)
    print(json.dumps(dict(experiment=identifier,phase=phase,verified_offload_bytes=freed)),flush=True)


def restore(identifier,phase,manifest='configs/native-methods.json'):
    import requests
    from huggingface_hub import hf_hub_url
    cfg=json.loads(Path(manifest).read_text())
    if identifier not in {s['id'] for s in cfg['experiments']}:raise ValueError('Unknown experiment')
    directory=Path('runs')/cfg['campaign']/phase/identifier
    marker=json.loads((directory/'audio-storage.json').read_text())
    if digest(directory/'result.json')!=marker['result_sha256']:raise ValueError('Stored result changed')
    result=json.loads((directory/'result.json').read_text())
    wanted={Path(r['audio']).name:r for r in result['rows'] if r['generation_status']=='ok'}
    if all((directory/r['audio']).exists() and digest(directory/r['audio'])==r['audio_sha256'] for r in wanted.values()):return
    url=hf_hub_url(marker['dataset'],filename=marker['archive'],revision=marker['revision'],repo_type='dataset')
    with requests.get(url,stream=True,timeout=90) as response:
        response.raise_for_status()
        size=int(response.headers.get('Content-Length',0))
        if not size or shutil.disk_usage(directory).free<2*size+2*2**30:raise RuntimeError('Insufficient restore space')
        with tempfile.TemporaryFile(dir=directory) as temp:
            sha=hashlib.sha256()
            for chunk in response.iter_content(8*2**20):temp.write(chunk);sha.update(chunk)
            if sha.hexdigest()!=marker['archive_sha256']:raise ValueError('Downloaded archive digest mismatch')
            temp.seek(0)
            with tarfile.open(fileobj=temp,mode='r:') as archive:
                members=archive.getmembers()
                if len(members)!=len(wanted) or {m.name for m in members}!=set(wanted):raise ValueError('Unexpected archive members')
                for member in members:
                    if not member.isfile():raise ValueError('Only plain audio files can be restored')
                    row=wanted[member.name];target=directory/'audio'/member.name
                    target.parent.mkdir(exist_ok=True)
                    with archive.extractfile(member) as source,target.with_suffix('.restoring').open('wb') as out:
                        shutil.copyfileobj(source,out)
                    if digest(target.with_suffix('.restoring'))!=row['audio_sha256']:raise ValueError('Restored audio mismatch')
                    target.with_suffix('.restoring').replace(target)
    print(json.dumps(dict(experiment=identifier,phase=phase,restored_audio=len(wanted))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--phase',choices=['pilot','full'],required=True);p.add_argument('--restore',action='store_true');p.add_argument('--manifest',default='configs/native-methods.json');a=p.parse_args()
    (restore if a.restore else offload)(a.experiment,a.phase,a.manifest)
