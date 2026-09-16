"""Download immutable author metric implementations and record every artifact hash."""
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from voicehub_arena.native_protocol import digest, write_json


def main():
    os.environ['HF_HOME']=str(Path.cwd()/'.cache/huggingface')
    from huggingface_hub import snapshot_download
    root=Path('artifacts/native-metrics');root.mkdir(parents=True,exist_ok=True)
    dnsmos_revision='591184a9fcb2cbdec02520fed81a32bbbf9d73ff'
    dns=root/'dnsmos';dns.mkdir(exist_ok=True)
    for name in ['dnsmos_local.py','DNSMOS/sig_bak_ovr.onnx','DNSMOS/model_v8.onnx']:
        dest=dns/Path(name).name
        if not dest.exists():
            url=f'https://raw.githubusercontent.com/microsoft/DNS-Challenge/{dnsmos_revision}/DNSMOS/{name}'
            temporary=dest.with_suffix('.partial')
            with urllib.request.urlopen(url,timeout=180) as source,temporary.open('wb') as target:
                shutil.copyfileobj(source,target)
            temporary.replace(dest)
    utmos_revision='47212055c2ecfb02d40cec2395233b83295d3d30'
    utmos=snapshot_download('sarulab-speech/UTMOS-demo',repo_type='space',revision=utmos_revision,
        allow_patterns=['*.py','*.pt','*.ckpt','LICENSE'],local_dir=root/'utmos22')
    unispeech_revision='6112826ac13a4327f4c9a7afa2a505e35b763514'
    source=Path('.deps/native/unispeech')
    if not source.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/microsoft/UniSpeech.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',unispeech_revision],check=True)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual!=unispeech_revision:raise ValueError('UniSpeech source revision changed')
    sv_revision='e876de7154845cd668b599bd4866f1d354c723df'
    sv=snapshot_download('k2-fsa/TTS_eval_models',revision=sv_revision,
        allow_patterns=['speaker_similarity/wavlm_large_finetune.pth',
                        'speaker_similarity/wavlm_large/wavlm_large.pt'],local_dir=root/'wavlm')
    manifest=dict(
        dnsmos={'source':'https://github.com/microsoft/DNS-Challenge','revision':dnsmos_revision,
                'path':str(dns.resolve()),'personalized':False},
        utmos22={'source':'https://huggingface.co/spaces/sarulab-speech/UTMOS-demo','revision':utmos_revision,
                 'path':str(Path(utmos).resolve()),'variant':'UTMOS22 strong learner, epoch=3-step=7459; not full ensemble'},
        wavlm_sim={'source':'https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification',
                   'source_revision':unispeech_revision,'source_path':str((source/'downstreams/speaker_verification').resolve()),
                   'checkpoint_mirror':'k2-fsa/TTS_eval_models','checkpoint_revision':sv_revision,
                   'path':str(Path(sv).resolve()),
                   's3prl_revision':'7ab62aaf2606d83da6c71ee74e7d16e0979edbc3',
                   'variant':'WavLM-large + fine-tuned ECAPA-TDNN speaker verification cosine SIM-o'}
    )
    for key,item in manifest.items():
        path=Path(item['path'])
        item['files_sha256']={str(p.relative_to(path)):digest(p) for p in path.rglob('*')
                              if p.is_file() and '.cache' not in p.parts}
    write_json(root/'manifest.json',manifest)
    print('Pinned native metric sources and checkpoints prepared',flush=True)


if __name__=='__main__':main()
