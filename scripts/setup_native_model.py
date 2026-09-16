"""Install one audited publisher runtime without changing the historical environment."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from voicehub_arena.native_protocol import write_json


DEPENDENCIES={
    'zonos':['transformers==4.57.6','huggingface-hub==0.36.0','kanjize==1.6.1','sudachipy==0.6.11','sudachidict-full==20260723','inflect==7.5.0','phonemizer==3.3.0','numpy==2.2.6'],
    'neutts':['transformers==5.1.0','huggingface-hub==1.7.2','neucodec==0.0.6','resemble-perth==1.0.1','librosa==0.11.0','phonemizer==3.3.0','soundfile==0.13.1','numpy==2.2.6','torchao==0.12.0','torchtune==0.3.1','vector-quantize-pytorch==1.17.8','einops==0.8.1','einx==0.3.0','frozendict==2.4.6','local-attention==1.11.1','hyper-connections==0.1.8','blobfile==3.1.0','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3','tiktoken==0.9.0'],
    'chatterbox':['transformers==5.2.0','huggingface-hub==1.7.2','numpy==1.26.4','librosa==0.11.0','s3tokenizer==0.3.0','onnx==1.17.0','Pillow==11.3.0','resemble-perth==1.0.1','diffusers==0.29.0','conformer==0.3.2','pykakasi==2.3.0','pyloudnorm==0.1.1','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3'],
    'qwen3tts':['transformers==4.57.3','accelerate==1.12.0','sox==1.5.0'],
    'omnivoice':['transformers==5.3.0','huggingface-hub==1.7.2','accelerate==1.12.0','tensorboardX==2.6.4','webdataset==1.0.2','pydub==0.25.1'],
    'voxcpm':['addict==2.4.0','wetext==0.1.2','simplejson==3.20.2','sortedcontainers==2.4.0','argbind==0.3.9'],
    'parlertts':['transformers==4.46.1','tokenizers==0.20.3','sentencepiece==0.2.1','descript-audio-codec==1.0.0','descript-audiotools==0.7.2','flatten-dict==0.4.2','argbind==0.3.9','julius==0.2.7','pyloudnorm==0.1.1','torch-stoi==0.2.3','ffmpy==0.5.0'],
    'supertonic':['onnxruntime==1.23.1','soundfile==0.13.1'],
    'dia':['descript-audio-codec==1.0.0','descript-audiotools==0.7.2','argbind==0.3.9','julius==0.2.7','pyloudnorm==0.1.1','torch-stoi==0.2.3','ffmpy==0.5.0'],
    'llasa':['xcodec2==0.1.5','torchao==0.12.0','torchtune==0.3.1','vector-quantize-pytorch==1.17.8','einops==0.8.1','einx==0.3.0','frozendict==2.4.6','local-attention==1.11.1','hyper-connections==0.1.8','blobfile==3.1.0','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3','datasets==3.6.0','dill==0.3.8','multiprocess==0.70.16','tiktoken==0.9.0'],
    'dia2':['sphn==0.2.1','whisper-timestamped==1.15.9','openai-whisper==20250625','dtw-python==1.5.3','tiktoken==0.9.0'],
    'f5tts':['accelerate==1.12.0','cached-path==1.6.5','vocos==0.1.0','torchdiffeq==0.2.5','ema-pytorch==0.7.9','x-transformers==2.8.2','hydra-core==1.3.2','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3','pydub==0.25.1','rjieba==0.1.13','transformers-stream-generator==0.0.5','einx==0.3.0','frozendict==2.4.6','loguru==0.7.3','encodec==0.1.1'],
}

# Import-time helpers do not pull in a second Torch runtime. Effective versions
# are recorded in arena-runtime.json for every environment.
AUDIO_UTILITIES=['docstring-parser==0.16','importlib-resources==6.5.2','matplotlib==3.10.6','ipython==9.5.0',
    'pystoi==0.4.1','markdown2==2.5.4','randomname==0.2.1','tensorboard==2.20.0',
    'protobuf==3.19.6','numpy==2.2.6']
SUPPORT_DEPENDENCIES={
    'neutts':['transformers==5.1.0','huggingface-hub==1.7.2','datasets==3.6.0','pandas==2.2.3','numpy==2.2.6'],
    'chatterbox':['transformers==5.2.0','huggingface-hub==1.7.2','numpy==1.26.4','scipy==1.15.3','scikit-learn==1.7.2','pandas==2.2.3','spacy-pkuseg==1.0.1','pykakasi==2.3.0'],
    'f5tts':['google-cloud-storage==2.19.0','boto3==1.35.99','matplotlib==3.10.6',
             'wandb==0.19.11','datasets==3.6.0','numpy==2.2.6','protobuf==5.29.5','huggingface-hub==0.36.0'],
    'llasa':['datasets==3.6.0','pandas==2.2.3','numpy==2.2.6','huggingface-hub==0.36.0'],
    'parlertts':AUDIO_UTILITIES,'dia':AUDIO_UTILITIES,
}


def setup(spec):
    key=spec['backend']
    if key in ('kokoro','transformers_vits','transformers_speecht5'):
        target=Path('.venvs/native-core/bin/python')
        if not target.exists():raise RuntimeError('Run setup_native_initial.sh first')
        return str(target)
    if key not in DEPENDENCIES:
        raise RuntimeError('Runtime dependency audit pending: '+key)
    source=spec['source'];checkout=Path('.deps/native')/key
    if not checkout.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',
                        'https://github.com/'+source['repo']+'.git',str(checkout)],check=True,
                       env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'})
        subprocess.run(['git','-C',str(checkout),'checkout','--detach',source['revision']],check=True)
    revision=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if revision!=source['revision']:raise ValueError('Native source revision mismatch')
    environment=Path('.venvs')/('native-'+key);python=environment/'bin/python'
    receipt=environment/'arena-runtime.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['source_revision']!=revision or saved['dependencies']!=DEPENDENCIES[key] or saved.get('support_dependencies',[])!=SUPPORT_DEPENDENCIES.get(key,[]):
            receipt.unlink()
        else:
            return str(python)
    if not python.exists():subprocess.run(['uv','venv',str(environment),'--python','.venv/bin/python','--seed'],check=True)
    site=subprocess.check_output([str(python),'-c','import site;print(site.getsitepackages()[0])'],text=True).strip()
    base=subprocess.check_output(['.venv/bin/python','-c','import site;print(site.getsitepackages()[0])'],text=True).strip()
    (Path(site)/'arena-base.pth').write_text(base+'\n')
    if key not in ('supertonic','llasa'):
        subprocess.run(['uv','pip','install','--python',str(python),'--no-deps','-e',str(checkout)],check=True)
    subprocess.run(['uv','pip','install','--python',str(python),'--no-deps',*DEPENDENCIES[key]],check=True)
    if key in SUPPORT_DEPENDENCIES:
        subprocess.run(['uv','pip','install','--python',str(python),*SUPPORT_DEPENDENCIES[key]],check=True)
    packages=json.loads(subprocess.check_output([str(python),'-c',
        'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:m.version(d.metadata["Name"]) for d in m.distributions()}))'],text=True))
    write_json(receipt,dict(source_revision=revision,dependencies=DEPENDENCIES[key],support_dependencies=SUPPORT_DEPENDENCIES.get(key,[]),packages=packages))
    return str(python)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    cfg=json.loads(Path('configs/native-methods.json').read_text())
    print(setup(next(s for s in cfg['experiments'] if s['id']==args.experiment)))
