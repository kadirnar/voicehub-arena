import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from voicehub_arena.native_protocol import write_json
from voicehub_arena.native_resources import GPU,admission,select_gpus
from voicehub_arena.native_scheduler import Job,Scheduler
from voicehub_arena.native_eval import lock_worker
from scripts.run_native_campaign import build_jobs


def test_reservations_cover_workers_loading_before_cuda_allocation():
    gpu=GPU('0','GPU-a','Test',24000,20000)
    # 6 GB has been promised, only 1 GB allocated; keep the remaining 5 GB free.
    assert admission(gpu,[(6000,1000)],13000) is None
    assert admission(gpu,[(6000,1000)],12000)==12000
    # Fully resident reservations are already subtracted in memory.free.
    assert admission(gpu,[(6000,6000)],16000)==16000
    # Other applications' memory is also part of the live free figure.
    gpu.free_mib=7000
    assert admission(gpu,[],5000) is None


def test_unprofiled_and_exclusive_jobs_wait_for_other_gpu_workers():
    gpu=GPU('0','GPU-a','Test',24000,22000)
    assert admission(gpu,[(3000,2000)],None) is None
    assert admission(gpu,[(3000,2000)],2000,exclusive=True) is None
    assert admission(gpu,[],None)==19600
    assert admission(gpu,[(1000,1000)]*4,1000,max_jobs=4) is None


def test_gpu_selection_honors_visible_devices_and_rejects_ambiguous_uuid():
    devices=[GPU('0','GPU-aa','A',24000,24000),GPU('1','GPU-ab','B',48000,48000)]
    assert select_gpus(devices,visible='1')==devices[1:]
    with pytest.raises(ValueError):select_gpus(devices,'0',visible='1')
    with pytest.raises(ValueError):select_gpus(devices,visible='GPU-a')
    with pytest.raises(ValueError):select_gpus(devices,visible='')


WORKER='''import json,os,sys,time,subprocess
from pathlib import Path
out=Path(sys.argv[1]);kind=sys.argv[2]
start=time.time()
if kind=='tree':
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
 out.write_text(json.dumps({'child':child.pid,'pid':os.getpid()}));time.sleep(30)
else:
 counter=out.with_suffix('.attempts');attempt=int(counter.read_text())+1 if counter.exists() else 1;counter.write_text(str(attempt))
 time.sleep(.16)
 fail=kind=='fail' or kind=='oom' and attempt==1
 out.write_text(json.dumps({'start':start,'end':time.time(),'attempt':attempt,
  'resource_usage':{'generate':{'torch_reserved_peak_mib':2000}},
  'error_kind':'cuda_oom' if fail else None,'status':'failed' if fail else 'completed'}))
 sys.exit(1 if fail else 0)
'''


def scheduler_fixture(tmp_path,jobspecs,profiled=True):
    worker=tmp_path/'worker.py';worker.write_text(WORKER)
    jobs=[]
    for name,resource,deps,kind in jobspecs:
        out=tmp_path/(name+'.json')
        jobs.append(Job(name,[sys.executable,str(worker),str(out),kind],resource,
                        name,'pilot','generate',deps,name,str(out)))
    gpu=GPU('0','GPU-test','Fake GPU',24000,24000)
    lock=(tmp_path/'gpu.lock').open('a')
    scheduler=Scheduler(jobs,tmp_path/'run',[gpu],lock.fileno(),max_gpu_jobs=2,
                        max_cpu_jobs=2,poll_seconds=.01,telemetry=lambda:[gpu])
    if profiled:scheduler.profiles={'GPU-test|'+j.id:{'peak_mib':2000} for j in jobs if j.resource=='gpu'}
    return scheduler,lock


def test_real_subprocesses_overlap_cpu_and_gpu_but_dependencies_wait(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('a','gpu',[],'ok'),('b','gpu',[],'ok'),
        ('cpu','cpu',[],'ok'),('after','gpu',['a','b'],'ok')])
    assert s.run_all()
    a,b,c,d=[json.loads((tmp_path/(name+'.json')).read_text()) for name in ('a','b','cpu','after')]
    assert max(a['start'],b['start'],c['start'])<min(a['end'],b['end'],c['end'])
    assert d['start']>=max(a['end'],b['end'])
    assert all(j.status=='completed' for j in s.jobs.values())
    lock.close()


def test_unknown_memory_profiles_run_exclusively_then_are_saved(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('a','gpu',[],'ok'),('b','gpu',[],'ok')],profiled=False)
    assert s.run_all()
    a,b=[json.loads((tmp_path/(n+'.json')).read_text()) for n in ('a','b')]
    assert b['start']>=a['end']
    assert s.profiles['GPU-test|a']['peak_mib']>=2000
    lock.close()


def test_two_gpus_can_calibrate_different_jobs_at_the_same_time(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('a','gpu',[],'ok'),('b','gpu',[],'ok')],profiled=False)
    devices=[GPU('0','GPU-test','A',24000,24000),GPU('1','GPU-second','B',24000,24000)]
    s.gpu_ids={g.uuid for g in devices};s.telemetry=lambda:devices
    assert s.run_all()
    a,b=[json.loads((tmp_path/(n+'.json')).read_text()) for n in ('a','b')]
    assert max(a['start'],b['start'])<min(a['end'],b['end'])
    contexts=[json.loads((s.run/(n+'--execution.json')).read_text()) for n in ('a','b')]
    assert {c['gpu_uuid'] for c in contexts}==s.gpu_ids
    lock.close()


def test_oom_retries_once_exclusively_and_does_not_unblock_failed_dependencies(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('oom','gpu',[],'oom'),('peer','gpu',[],'ok'),
        ('bad','cpu',[],'fail'),('blocked','gpu',['bad'],'ok')])
    assert not s.run_all()
    assert s.jobs['oom'].status=='completed' and s.jobs['oom'].attempts==2
    assert s.profiles['GPU-test|oom']['exclusive']
    assert s.jobs['blocked'].status=='blocked'
    assert not (tmp_path/'blocked.json').exists()
    events=s.state['events'];assert sum(e.get('retry_exclusive',False) for e in events)==1
    # Retry finishes after the concurrent peer; the next attempt is exclusive.
    assert json.loads((tmp_path/'oom.json').read_text())['start']>=json.loads((tmp_path/'peer.json').read_text())['end']
    lock.close()


def test_profile_larger_than_shared_budget_still_runs_alone(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('a','gpu',[],'ok')])
    s.profiles['GPU-test|a']['peak_mib']=21000
    assert s.run_all() and s.jobs['a'].exclusive
    lock.close()


def test_stop_terminates_worker_process_group_and_records_stopped(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('tree','gpu',[],'tree')])
    gpu=GPU('0','GPU-test','Fake GPU',24000,24000)
    def telemetry():
        if (tmp_path/'tree.json').exists():s.stop()
        return [gpu]
    s.telemetry=telemetry
    assert not s.run_all()
    assert s.state['status']=='stopped_by_user' and not s.running
    row=json.loads((tmp_path/'tree.json').read_text())
    with pytest.raises(ProcessLookupError):os.kill(row['pid'],0)
    # A terminated orphan may briefly be a zombie until the host init reaps it.
    proc=Path('/proc')/str(row['child'])/'stat'
    if proc.exists():assert proc.read_text().split()[2]=='Z'
    child_status=subprocess.run(['ps','-o','stat=','-p',str(row['child'])],capture_output=True,text=True).stdout.strip()
    assert not child_status or child_status.startswith('Z')
    lock.close()


def test_telemetry_failure_stops_active_jobs_instead_of_admitting_more(tmp_path):
    s,lock=scheduler_fixture(tmp_path,[('a','gpu',[],'ok')])
    calls=0
    def telemetry():
        nonlocal calls
        calls+=1
        if calls>1:raise RuntimeError('telemetry unavailable')
        return [GPU('0','GPU-test','Fake GPU',24000,24000)]
    s.telemetry=telemetry
    with pytest.raises(RuntimeError,match='telemetry unavailable'):s.run_all()
    assert not s.running and s.state['status']=='needs_attention'
    lock.close()


def test_cpu_generation_never_initializes_cuda(tmp_path,monkeypatch):
    import contextlib
    import types
    import numpy as np
    from voicehub_arena import native_eval,native_adapters
    class CudaForbidden:
        def __getattr__(self,key):raise AssertionError('CPU worker accessed CUDA: '+key)
    torch=types.SimpleNamespace(set_num_threads=lambda n:None,manual_seed=lambda n:None,
        inference_mode=contextlib.nullcontext,cuda=CudaForbidden())
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setattr(native_eval.importlib.metadata,'version',lambda name:'test')
    monkeypatch.setattr(native_adapters,'load',lambda spec:lambda row:iter([(np.ones(1600,dtype=np.float32)*.1,16000)]))
    spec=dict(id='cpu-test',settings={'inference_device':'cpu'},uses_reference=False,streaming=False)
    result={'rows':[]}
    native_eval.generation(spec,{'seed':42},[dict(id='one',text='one',reference='one')],tmp_path,result)
    assert len(result['rows'])==1 and result['rows'][0]['peak_vram_mib']==0


def test_same_experiment_rejects_two_workers_even_with_gpu_sharing(tmp_path):
    lock,_=lock_worker(tmp_path,False)
    command='from pathlib import Path;from voicehub_arena.native_eval import lock_worker;lock_worker(Path('+repr(str(tmp_path))+'),False)'
    p=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True)
    assert p.returncode!=0 and 'BlockingIOError' in p.stderr
    lock.close()


def test_campaign_dag_uses_native_envs_cpu_dnsmos_and_full_pilot_gate():
    cfg=json.loads(Path('configs/native-methods.json').read_text());cfg['campaign']='test-new-parallel-no-results'
    cfg['experiments']=[s for s in cfg['experiments'] if s['id'] in ('kokoro--preset_voice','supertonic--preset_voice')]
    jobs=build_jobs(cfg,'unused-manifest.json');by_id={j.id:j for j in jobs}
    assert by_id['kokoro--preset_voice:full:generate'].dependencies==['setup:kokoro','kokoro--preset_voice:pilot:summarize']
    assert by_id['supertonic--preset_voice:pilot:generate'].resource=='cpu'
    assert by_id['kokoro--preset_voice:pilot:dnsmos'].resource=='cpu'
    assert by_id['kokoro--preset_voice:pilot:asr'].resource=='gpu'
    assert all('--manifest' in j.command for j in jobs)
    assert not any(j.action in ('publish','offload') for j in jobs)
