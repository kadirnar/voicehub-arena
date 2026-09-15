"""Bounded diagnostic ablations. These selected cases must never replace full scores."""
import os
from pathlib import Path
ROOT=Path(os.environ.get('VOICEHUB_ARENA_ROOT', Path.cwd())).resolve()
OUT=Path(os.environ.get('VOICEHUB_AUDIT_DIR', ROOT/'runs/high-wer-diagnostics')).resolve()
os.chdir(ROOT);os.environ['HF_HOME']=str(ROOT/'.cache/huggingface');os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
import fcntl,gc,hashlib,inspect,json,time,traceback
lock=open(ROOT/'runs/.gpu.lock','a+')
print('Waiting for the current benchmark GPU lock',flush=True)
fcntl.flock(lock,fcntl.LOCK_EX)
print('GPU lock acquired',flush=True)
import numpy as np,soundfile as sf,torch
torch.set_num_threads(4)
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse
from voicehub_arena.inputs import prepare_request
from voicehub_arena.metrics import errors,signal_metrics
from voicehub import AutoModelForTextToSpeech,TTSGenerationConfig
configure_hub_auth();enable_verified_cache_reuse()
selected=json.loads((OUT/'diagnostic-selection.json').read_text())
records=[];events=[]

def persist():
 (OUT/'diagnostic-records.json').write_text(json.dumps(records,indent=2)+'\n')
 (OUT/'diagnostic-events.json').write_text(json.dumps(events,indent=2)+'\n')

def add_audio(model,i,row,variant,audio,sr,**extra):
 if hasattr(audio,'detach'): audio=audio.detach().float().cpu().numpy()
 audio=np.asarray(audio).reshape(-1)
 name=f'{model}-{i}-{variant}.wav';p=OUT/name;sf.write(p,audio,sr,subtype='FLOAT')
 rec=dict(model=model,sample=i,id=row['id'],reference=row['reference'],variant=variant,audio=name,
          audio_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**signal_metrics(audio,sr),**extra)
 records.append(rec);persist();print(json.dumps({'phase':'generated','model':model,'sample':i,'variant':variant,'duration_s':rec['duration_s']}),flush=True)

def attempt(model,i,row,variant,fn):
 try:
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode(): result=fn()
  torch.cuda.synchronize()
  add_audio(model,i,row,variant,result.audio,result.sample_rate,latency_s=time.perf_counter()-start)
 except Exception as e:
  events.append({'model':model,'sample':i,'variant':variant,'error':f'{type(e).__name__}: {e}'});persist();traceback.print_exc()

# Preserve the historical WAV and exact archived transcript for every selected case.
for model,rows in selected.items():
 for i,row in enumerate(rows):
  p=ROOT/'runs'/row['source_run']/row['audio']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['audio_sha256']
  records.append(dict(model=model,sample=i,id=row['id'],reference=row['reference'],variant='archived',audio=str(p),
     audio_sha256=row['audio_sha256'],archived_transcript=row['transcript'],archived_metrics=row['metrics'],duration_s=row['duration_s']))
persist()

for model in ['vits','vui','conversationtts','bark','openvoice','voxcpm']:
 native=None
 try:
  cfg=json.loads((ROOT/'runs'/f'pub-v2-{model}-seedtts_en-000'/'config.json').read_text())
  override=cfg['overrides'].get(model,{})
  spec=next(r for r in cfg['catalog'] if r['model_type']==model)
  checkpoint=override.get('checkpoint',spec['checkpoint'])
  native=AutoModelForTextToSpeech.from_pretrained(checkpoint,model_type=model,device='cuda',**override.get('config',{}));native.load()
  events.append({'model':model,'checkpoint':checkpoint,'dtype':str(next(native.model.parameters()).dtype),'phase':'loaded'});persist()
  def generate(row,**options):
   row_cfg=json.loads((ROOT/'runs'/row['source_run']/'config.json').read_text())
   prepared_dir=row_cfg['overrides'].get(model,{}).get('prepared_inputs')
   text,opts=prepare_request(model,override.get('text_prefix','')+row['reference'],{**override.get('generation',{}),**options},prepared_inputs=prepared_dir,model=native)
   return native.generate(text,generation_config=TTSGenerationConfig(seed=42),**opts)
  for i,row in enumerate(selected[model]): attempt(model,i,row,'baseline',lambda row=row:generate(row))
  if model=='vits':
   native=None;gc.collect();torch.cuda.empty_cache()
   native=AutoModelForTextToSpeech.from_pretrained(checkpoint,model_type=model,device='cuda',**{**override.get('config',{}),'torch_dtype':'float32'});native.load()
   for i,row in enumerate(selected[model]): attempt(model,i,row,'fp32',lambda row=row:generate(row))
  elif model=='vui':
   from voicehub.models.vui import tts
   original=tts.generate
   source=inspect.getsource(original)
   assert 'if offset < 24.53 * 4:' in source
   context=dict(tts.__dict__)
   exec(compile(source.replace('if offset < 24.53 * 4:','if offset < self.codec.hz:'),'<diagnostic-minimum-one-second>','exec'),context)
   tts.generate=context['generate']
   try:
    for i,row in enumerate(selected[model]): attempt(model,i,row,'minimum_1_second',lambda row=row:generate(row))
   finally: tts.generate=original
  elif model=='conversationtts':
   codec=native._inference_generator()._audio_tokenizer
   tokens=torch.zeros(32,20,dtype=torch.long,device='cuda')
   with torch.inference_mode(): probe=codec.detokenize(tokens)
   events.append({'model':model,'codec_probe_frames':20,'codec_probe_samples':probe.numel(),'sample_rate':native.sample_rate,'actual_ms_per_frame':probe.numel()/native.sample_rate/20*1000,'generator_ms_per_frame':40});persist()
   for i,row in enumerate(selected[model]): attempt(model,i,row,'speaker_1',lambda row=row:generate(row,speaker=1))
  elif model=='bark':
   native=None;gc.collect();torch.cuda.empty_cache()
   from transformers import BarkModel,BarkProcessor
   hf=BarkModel.from_pretrained(checkpoint,revision=override['config']['revision'],torch_dtype=torch.float32).eval().cuda()
   tok=BarkProcessor.from_pretrained(checkpoint,revision=override['config']['revision'])
   for i,row in enumerate(selected[model]):
    try:
     torch.manual_seed(42);torch.cuda.manual_seed_all(42)
     with torch.inference_mode(): wave=hf.generate(**tok(row['reference'],return_tensors='pt').to('cuda'))
     add_audio(model,i,row,'transformers',wave,24000)
    except Exception as e:
     events.append({'model':model,'sample':i,'variant':'transformers','error':f'{type(e).__name__}: {e}'});persist();traceback.print_exc()
   del hf,tok
  elif model=='openvoice':
   original_base=native._native_base_audio
   # Save the exact intermediate signal from the conversion call, with matching RNG.
   for i,row in enumerate(selected[model]):
    def capture(*args,**kwargs):
     result=original_base(*args,**kwargs)
     add_audio(model,i,row,'melo_before_conversion',result.audio,result.sample_rate)
     return result
    native._native_base_audio=capture
    attempt(model,i,row,'captured_conversion',lambda row=row:generate(row))
   native._native_base_audio=original_base
   del original_base,capture
   from huggingface_hub import hf_hub_download
   source_path=hf_hub_download(checkpoint,'base_speakers/ses/en-us.pth',revision=override['config']['revision'])
   for i,row in enumerate(selected[model]): attempt(model,i,row,'publisher_source_embedding',lambda row=row:generate(row,source_embedding=source_path))
  elif model=='voxcpm':
   for i,row in enumerate(selected[model]): attempt(model,i,row,'cfg_3',lambda row=row:generate(row,cfg_value=3.0))
 except Exception as e:
  events.append({'model':model,'phase':'model_failed','error':f'{type(e).__name__}: {e}'});persist();traceback.print_exc()
 finally:
  native=None;gc.collect();torch.cuda.empty_cache()

# Include independent VITS FP32 CPU outputs already computed while the GPU was busy.
for i,row in enumerate(selected['vits']):
 for variant in ['native_cpu_fp32','transformers_cpu_fp32']:
  p=OUT/f'vits-{i}-{variant}.wav'
  if p.exists(): records.append(dict(model='vits',sample=i,id=row['id'],reference=row['reference'],variant=variant,audio=p.name,audio_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
persist()
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
asr=WhisperModel(snapshot_download('Systran/faster-whisper-large-v3',revision='edaa852ec7e145841d8ffdb056a99866b5f0a478',local_files_only=True),device='cuda',compute_type='float16',cpu_threads=4)
for row in records:
 path=Path(row['audio']);path=path if path.is_absolute() else OUT/path
 try:
  modes=[False,True] if row['variant']=='archived' else [False]
  for vad in modes:
   segments,info=asr.transcribe(str(path),language='en',task='transcribe',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=vad)
   transcript=' '.join(s.text.strip() for s in segments).strip()
   key='vad_diagnostic' if vad else 'rescore'
   row[key]={'transcript':transcript,'metrics':errors([row['reference']],[transcript],normalization='whisper_english')}
  persist();print(json.dumps({'phase':'scored','model':row['model'],'sample':row['sample'],'variant':row['variant'],'wer':row['rescore']['metrics']['wer']}),flush=True)
 except Exception as e:
  row['scoring_error']=f'{type(e).__name__}: {e}';persist();traceback.print_exc()
assert len(records)==132 and all('rescore' in r for r in records), 'All 132 controls must finish; run the VITS CPU script first'
summary=[]
for model in selected:
 for variant in sorted({r['variant'] for r in records if r['model']==model}):
  group=[r for r in records if r['model']==model and r['variant']==variant and 'rescore' in r]
  if group: summary.append({'model':model,'variant':variant,'samples':len(group),**errors([r['reference'] for r in group],[r['rescore']['transcript'] for r in group],normalization='whisper_english')})
(OUT/'diagnostic-summary.json').write_text(json.dumps({'selection':'Top three by word-edit count, first two corpus records, longest text. Diagnostic only; not representative full WER.','variants':summary,'events':events},indent=2)+'\n')
(OUT/'DONE.json').write_text(json.dumps({'finished_at':time.time(),'record_count':len(records),'scored':sum('rescore' in r for r in records),'events':len(events)},indent=2)+'\n')
print('DIAGNOSTIC DONE',flush=True)
