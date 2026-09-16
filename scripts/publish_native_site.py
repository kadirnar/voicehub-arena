"""Publish the native dashboard without rebuilding historical arena data."""
import json
from pathlib import Path

from huggingface_hub import HfApi
from scripts.publish_native_progress import SPACE,commit
from voicehub_arena.native_protocol import write_json


def main():
    root=Path(__file__).resolve().parents[1];site=root/'hf-space';api=HfApi()
    assert api.whoami()['name']=='kadirnar'
    files={name:site/name for name in ('index.html','native.html','native.js')}
    files['reports/NATIVE_METHODS_2026-09-16.md']=root/'docs/NATIVE_METHODS_2026-09-16.md'
    files['reports/NATIVE_PARALLEL_TR.md']=root/'docs/NATIVE_PARALLEL_TR.md'
    revision=commit(api,SPACE,'space',files,'Update native method dashboard and protocol')
    receipt=dict(revision=revision,verified_files=list(files))
    write_json(root/'runs/native-methods-20260916/site-publication.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
