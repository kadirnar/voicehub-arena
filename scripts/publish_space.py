"""Publish the static viewer against an immutable, complete HF dataset revision."""
import argparse
import json
from pathlib import Path
import re
from huggingface_hub import HfApi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--space-id',default='kadirnar/voicehub-arena')
    p.add_argument('--space-dir',type=Path,default=Path(__file__).resolve().parents[1]/'hf-space')
    p.add_argument('--dataset-revision',required=True)
    args=p.parse_args()
    if not re.fullmatch(r'[a-f0-9]{40}',args.dataset_revision):
        p.error('--dataset-revision must be an immutable 40-character commit SHA')
    path=args.space_dir/'data/leaderboard.json'
    data=json.loads(path.read_text());api=HfApi()
    repo=api.repo_info(data['dataset_id'],repo_type='dataset',revision=args.dataset_revision)
    assert repo.sha==args.dataset_revision
    assert sum(f.rfilename.startswith('audio_shards/') and f.rfilename.endswith('.tar') for f in repo.siblings)==33
    assert data['models']==len(data['table'])==33 and data['scored_audio']==35904
    data['dataset_revision']=args.dataset_revision
    path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    api.create_repo(repo_id=args.space_id,repo_type='space',space_sdk='static',private=False,exist_ok=True)
    result=api.upload_folder(repo_id=args.space_id,repo_type='space',folder_path=args.space_dir,
                             commit_message='Publish full 33-model Seed-TTS benchmark arena')
    print(json.dumps({'space':args.space_id,'space_revision':result.oid,
                      'dataset_revision':args.dataset_revision,
                      'url':f'https://huggingface.co/spaces/{args.space_id}'}))


if __name__=='__main__':
    main()
