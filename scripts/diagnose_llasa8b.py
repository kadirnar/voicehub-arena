"""Isolated early-EOS precision diagnostic. Never included in benchmark rankings."""
from pathlib import Path
import json,torch,time,fcntl
from transformers import LlamaForCausalLM
from voicehub_arena.variant_eval import load_campaign,prepare_llasa
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse
lock=Path('runs/.gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
configure_hub_auth();enable_verified_cache_reuse()
cfg,data=load_campaign('configs/variant-campaign.json');spec=next(s for s in cfg['models'] if s['id']=='llasa-8b')
out=Path('runs')/cfg['campaign']/'diagnostics/llasa-8b';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
evidence={'kind':'diagnostic only; never included in benchmark rankings','source_index':931,'text':data[931]['text'],'attempts':[]}
base=LlamaForCausalLM.generate
model_ref=[]
def capture(self,*args,**kwargs):
 model_ref[:]=[self]
 result=base(self,*args,**kwargs)
 generated=result[0,args[0].shape[1]:].tolist()
 evidence['attempts'].append({'dtype':str(next(self.parameters()).dtype),'generated_ids':generated,'eos':kwargs.get('eos_token_id'),'input_length':args[0].shape[1],'seed':42})
 return result
LlamaForCausalLM.generate=capture
synth,meta=prepare_llasa(spec,cfg,out);evidence['effective']=meta
with torch.inference_mode():
 # Match the campaign warmup and per-prompt seed reset, without ASR selection.
 synth(data[0]['text']);evidence['attempts']=[]
 for dtype in ['bfloat16','float32']:
  if dtype=='float32':model_ref[0].float();torch.cuda.empty_cache()
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  try:
   audio,sr,details=synth(data[931]['text']);evidence['attempts'][-1].update(details=details,samples=audio.numel())
  except Exception as e:evidence['attempts'][-1]['error']=str(e)
  (out/'early-eos-diagnostic.json').write_text(json.dumps(evidence,indent=2))
  print(json.dumps({k:v for k,v in evidence['attempts'][-1].items() if k!='generated_ids'}),flush=True)
print('DONE',flush=True)
