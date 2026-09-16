"""Speaker metric identity controls, separate from all TTS benchmark results."""
import fcntl
import argparse
import json
import math
import os
from pathlib import Path

from voicehub_arena.native_protocol import forbid_voicehub,write_json
from voicehub_arena.native_quality import make_scorer


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    forbid_voicehub();os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    if a.device=='cuda':
        lock=Path('runs/.gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    import torch
    torch.set_num_threads(4)
    cfg=json.loads(Path('configs/native-methods.json').read_text())
    cfg['quality_control_device']=a.device
    rows=[json.loads(line) for line in Path(cfg['dataset']).read_text().splitlines()]
    first=rows[0];second=next(r for r in rows[1:] if r['reference_audio_sha256']!=first['reference_audio_sha256'])
    score,provenance=make_scorer('wavlm_sim',cfg)
    same=score(Path(first['reference_audio']),first)['similarity']
    different=score(Path(second['reference_audio']),first)['similarity']
    assert math.isclose(same,1.,abs_tol=1e-5)
    assert math.isfinite(different) and -1.00001<=different<same-1e-5
    result=dict(status='passed',device=a.device,same_recording_sim=same,different_recording_sim=different,
                reference_sha256=first['reference_audio_sha256'],different_reference_sha256=second['reference_audio_sha256'],
                provenance=provenance,scope='Metric integrity controls, not generated TTS scores')
    write_json(Path('runs')/cfg['campaign']/'quality-controls.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))


if __name__=='__main__':main()
