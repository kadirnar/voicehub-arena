"""Check audited native imports without loading model weights or producing scores."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from scripts.setup_native_model import setup
from voicehub_arena.native_protocol import write_json

IMPORTS={
    'kokoro':'from kokoro import KModel,KPipeline',
    'transformers_vits':'from transformers import VitsModel',
    'transformers_speecht5':'from transformers import SpeechT5ForTextToSpeech',
    'qwen3tts':'from qwen_tts import Qwen3TTSModel',
    'omnivoice':'from omnivoice import OmniVoice',
    'voxcpm':'from voxcpm import VoxCPM',
    'parlertts':'from parler_tts import ParlerTTSForConditionalGeneration',
    'dia':'from dia.model import Dia',
    'dia2':'from dia2 import Dia2,GenerationConfig',
    'llasa':'from xcodec2.modeling_xcodec2 import XCodec2Model',
    'f5tts':'from f5_tts.api import F5TTS',
    'zonos':'from zonos.model import Zonos;from zonos.conditioning import make_cond_dict',
    'neutts':'from neutts import NeuTTS,NeuTTS2E',
    'chatterbox':'from chatterbox.tts import ChatterboxTTS',
    'supertonic':"import sys;sys.path.insert(0,'.deps/native/supertonic/py');from helper import load_text_to_speech",
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',action='append',choices=IMPORTS)
    args=parser.parse_args()
    cfg=json.loads(Path('configs/native-methods.json').read_text());checks={};visited=set()
    output=Path('runs')/cfg['campaign']/'runtime-import-checks.json'
    if args.backend and output.exists():checks=json.loads(output.read_text()).get('checks',{})
    for spec in cfg['experiments']:
        backend=spec['backend']
        if not spec.get('verified_api') or backend in visited or (args.backend and backend not in args.backend):continue
        visited.add(backend)
        try:
            python=setup(spec)
            command='from voicehub_arena.native_protocol import forbid_voicehub;forbid_voicehub();'+IMPORTS[backend]
            run=subprocess.run([python,'-c',command],capture_output=True,text=True,timeout=180)
            checks[backend]=dict(status='passed' if run.returncode==0 else 'failed',exit_code=run.returncode,output=(run.stdout+run.stderr)[-6000:])
        except Exception as exc:checks[backend]=dict(status='failed',error=str(exc))
        write_json(output,dict(updated_at=time.time(),checks=checks));print(backend,checks[backend]['status'],flush=True)


if __name__=='__main__':main()
