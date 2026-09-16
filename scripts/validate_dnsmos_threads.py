import json,time
from pathlib import Path
from voicehub_arena.native_quality import make_scorer
from voicehub_arena.native_protocol import write_json
cfg=json.loads(Path('configs/native-methods.json').read_text());run=Path('runs')/cfg['campaign'];directory=run/'pilot/kokoro--preset_voice';r=json.loads((directory/'result.json').read_text());score,p=make_scorer('dnsmos',cfg);before=time.perf_counter();errors=[]
for row in r['rows']:
 actual=score(directory/row['audio'],row)
 errors.extend(abs(actual[k]-row['scores']['dnsmos'][k]) for k in actual)
assert max(errors)<1e-5,max(errors)
result=dict(status='passed',samples=len(r['rows']),max_absolute_difference=max(errors),total_scoring_s=time.perf_counter()-before,provenance=p)
write_json(run/'dnsmos-thread-controls.json',result);print(json.dumps({k:v for k,v in result.items() if k!='provenance'}))
