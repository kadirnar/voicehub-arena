"""Compare the actual MMS checkpoint and full tokenizer against Transformers on CPU."""
import os
from pathlib import Path
ROOT=Path(os.environ.get('VOICEHUB_ARENA_ROOT', Path.cwd())).resolve()
OUT=Path(os.environ.get('VOICEHUB_AUDIT_DIR', ROOT/'runs/high-wer-diagnostics')).resolve()
os.chdir(ROOT)
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import json, hashlib
import numpy as np
import torch
import soundfile as sf
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse
configure_hub_auth(); enable_verified_cache_reuse(); torch.set_num_threads(4)
from voicehub import AutoModelForTextToSpeech,TTSGenerationConfig
from transformers import VitsModel, AutoTokenizer

revision='c71de0fe7204c83f1c10820a7d696d0b450048ba'
model=AutoModelForTextToSpeech.from_pretrained('facebook/mms-tts-eng',model_type='vits',revision=revision,device='cpu',torch_dtype='float32')
model.load()
hf=VitsModel.from_pretrained('facebook/mms-tts-eng',revision=revision,torch_dtype=torch.float32).eval()
tokenizer=AutoTokenizer.from_pretrained('facebook/mms-tts-eng',revision=revision)
rows=[]
for p in sorted((ROOT/'runs').glob('pub-v2-vits-seedtts_en-*/vits/result.json')): rows+=json.loads(p.read_text())['rows']
assert len(rows)==1088 and len({r['id'] for r in rows})==1088
token_mismatch=[]
for row in rows:
 n=model._tokenize(row['text'],normalize=None)['input_ids'].tolist()
 h=tokenizer(row['text'],return_tensors='pt')['input_ids'].tolist()
 if n!=h: token_mismatch.append(row['id'])
selection=json.loads((OUT/'diagnostic-selection.json').read_text())['vits']
results=[]
for i,row in enumerate(selection):
 with torch.inference_mode():
  native=model.generate(row['text'],generation_config=TTSGenerationConfig(seed=42))
  torch.manual_seed(42)
  official=hf(**tokenizer(row['text'],return_tensors='pt')).waveform
 a=native.audio.float().cpu().numpy().reshape(-1);b=official.float().cpu().numpy().reshape(-1)
 for variant,wave in [('native_cpu_fp32',a),('transformers_cpu_fp32',b)]:
  sf.write(OUT/f'vits-{i}-{variant}.wav',wave,16000,subtype='FLOAT')
 entry={'id':row['id'],'reference':row['reference'],'sample':i,'shape_native':len(a),'shape_transformers':len(b),'native_dtype':str(next(model.model.parameters()).dtype)}
 if len(a)==len(b): entry.update(max_abs_diff=float(np.max(np.abs(a-b))),mean_abs_diff=float(np.mean(np.abs(a-b))),correlation=float(np.corrcoef(a,b)[0,1]))
 results.append(entry)
 (OUT/'vits-cpu-parity.json').write_text(json.dumps({'checkpoint_revision':revision,'all1088_tokenizer_mismatches':token_mismatch,'samples':results},indent=2)+'\n')
 print(json.dumps(entry),flush=True)
