"""Publish or verify the prepared audio dataset; never include loose audio caches."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi


def verify(api,repo_id,folder,revision=None):
    expected=json.loads((folder/'audio-shards.json').read_text())
    rows=json.loads((folder/'audio-manifest.json').read_text())
    assert len(expected)==33 and sum(s['samples'] for s in expected)==len(rows)==35904
    info=api.repo_info(repo_id,repo_type='dataset',revision=revision,files_metadata=True)
    remote={f.rfilename:f for f in info.siblings}
    for shard in expected:
        stored=remote[shard['path']]
        assert stored.size==shard['size'],shard['path']
        assert stored.lfs and stored.lfs.sha256==shard['sha256'],shard['path']
    return {'repo_id':repo_id,'revision':info.sha,'verified_archives':33,
            'verified_audio_records':35904,'archive_sha256_match':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--repo-id',default='kadirnar/voicehub-arena-seed-tts-eval')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--revision')
    a=p.parse_args();api=HfApi()
    if not a.verify_only:
        for shard in json.loads((a.dataset_dir/'audio-shards.json').read_text()):
            path=a.dataset_dir/shard['path'];assert path.stat().st_size==shard['size']
            h=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(8*1024*1024):h.update(block)
            assert h.hexdigest()==shard['sha256']
        api.create_repo(repo_id=a.repo_id,repo_type='dataset',private=False,exist_ok=True)
        result=api.upload_folder(repo_id=a.repo_id,repo_type='dataset',folder_path=a.dataset_dir,
                                 ignore_patterns=['audio/**','.cache/**','.git/**'],
                                 commit_message='Publish complete English Seed-TTS benchmark audio and records')
        a.revision=result.oid
    print(json.dumps(verify(api,a.repo_id,a.dataset_dir,a.revision)))


if __name__=='__main__':
    main()
