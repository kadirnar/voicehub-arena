"""Compare cached and full-prefix Dia on the pinned release and actual GPU.

Run only while holding Arena's GPU lock (interleave_repair --preflight does so).
This checks numerical/token agreement, not independent acoustic quality.
"""
import json
from pathlib import Path
import time

import torch
from voicehub.architectures.dia.runtime import load_dia_runtime
from voicehub.architectures.dia.metadata import NARI_DIA_CHECKPOINT_REVISION
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse, resolved_artifacts

configure_hub_auth()
enable_verified_cache_reuse()
torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
runtime = load_dia_runtime('nari-labs/Dia-1.6B-0626', revision=NARI_DIA_CHECKPOINT_REVISION,
                          compute_dtype='float32', device='cuda')
model = runtime.model.eval()
batch = runtime.processor(text=['[S1] The morning train arrived at the station.']).to('cuda')
errors = []
with torch.inference_mode():
    encoded = model.model.encoder(batch['input_ids'], batch['attention_mask']).last_hidden_state
    torch.manual_seed(71)
    codes = torch.randint(0, 1024, (1, 12, model.num_channels), device='cuda')
    cache = [{} for _ in model.model.decoder.layers]
    start = 0
    for end in (4, 5, 8, 12):
        full = model.model.decoder(codes[:, :end], encoder_hidden_states=encoded,
                                  encoder_attention_mask=batch['attention_mask'])
        cached = model.model.decoder(codes[:, start:end], encoder_hidden_states=encoded,
                                    encoder_attention_mask=batch['attention_mask'], cache=cache)
        expected = model.logits_dense(full[:, start:end]).float()
        actual = model.logits_dense(cached).float()
        errors.append(float((expected-actual).abs().max()))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-5)
        start = end
    timings = {}
    tokens = []
    for cached in (False, True):
        torch.cuda.synchronize()
        began = time.perf_counter()
        tokens.append(model.generate(**batch, max_new_tokens=12, use_cache=cached, do_sample=False))
        torch.cuda.synchronize()
        timings[str(cached)] = time.perf_counter()-began
    if not torch.equal(*tokens):
        raise ValueError('Dia cached greedy tokens differ from full-prefix evaluation')
record = {'model': 'dia', 'revision': NARI_DIA_CHECKPOINT_REVISION, 'compute_dtype': 'float32',
          'gpu': torch.cuda.get_device_name(), 'max_absolute_logit_errors': errors,
          'greedy_token_match': True, 'generation_seconds': timings,
          'scope': '12-token GPU comparison; acoustic benchmark is a separate repair run',
          'artifacts': resolved_artifacts()}
path = root/'runs/validations/dia-cache-gpu.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(record, indent=2))
print('Dia actual-checkpoint GPU cache validation passed:', json.dumps(record), flush=True)
