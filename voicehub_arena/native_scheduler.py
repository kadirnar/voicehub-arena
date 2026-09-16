"""Dependency-aware subprocess scheduler; one writer per experiment, bounded VRAM."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .native_protocol import write_json
from .native_resources import read_gpus,process_memory,reservation,admission


@dataclass
class Job:
    id: str
    command: list
    resource: str
    experiment: str = ''
    phase: str = ''
    action: str = ''
    dependencies: list = field(default_factory=list)
    profile_key: str = ''
    result_path: str = ''
    status: str = 'pending'
    attempts: int = 0
    exclusive: bool = False
    peak_mib: int = 0


class Scheduler:
    def __init__(self,jobs,run,gpus,lock_fd,max_gpu_jobs=4,max_cpu_jobs=2,
                 headroom_mib=2048,poll_seconds=1,telemetry=read_gpus,mode='parallel'):
        self.jobs={j.id:j for j in jobs};self.run=Path(run);self.run.mkdir(parents=True,exist_ok=True)
        self.gpu_ids={g.uuid for g in gpus};self.telemetry=telemetry;self.lock_fd=lock_fd
        self.max_gpu_jobs=max_gpu_jobs;self.max_cpu_jobs=max_cpu_jobs
        self.headroom=headroom_mib;self.interval=poll_seconds;self.mode=mode
        self.running={};self.stopping=False;self.draining=False;self.reason=None
        self.profile_path=self.run/'gpu-profiles.json'
        self.profiles=json.loads(self.profile_path.read_text()) if self.profile_path.exists() else {}
        old=self.run/'campaign-status.json'
        self.state=json.loads(old.read_text()) if old.exists() else {'events':[],'started_at':time.time()}
        self.state.update(campaign=self.run.name,scheduler='vram-aware-v1',execution_mode=mode)

    def save(self,status='running'):
        self.state.update(status=status,updated_at=time.time(),experiment=None,phase=None,action=None,
            active_jobs=[dict(id=j.id,experiment=j.experiment,phase=j.phase,action=j.action,
                resource=j.resource,pid=p.pid,gpu_uuid=g,reserved_mib=reserved)
                for j,p,log,g,reserved in self.running.values()],
            jobs={k:dict(status=j.status,attempts=j.attempts) for k,j in self.jobs.items()},
            scheduler_error=self.reason)
        write_json(self.run/'campaign-status.json',self.state)

    def stop(self,*_):self.stopping=True
    def drain(self,*_):self.draining=True

    def profile(self,job,gpu):return self.profiles.get(gpu.uuid+'|'+job.profile_key,{})

    def start(self,job,gpu=None,reserved=0):
        context=dict(mode='parallel_throughput' if self.mode=='parallel' else 'isolated',
                     gpu_uuid=gpu.uuid if gpu else None,gpu_name=gpu.name if gpu else None,
                     gpu_total_mib=gpu.total_mib if gpu else None,max_gpu_jobs=self.max_gpu_jobs)
        context_path=self.run/(job.id.replace(':','--')+'--execution.json');write_json(context_path,context)
        env={**os.environ,'PYTHONPATH':str(Path.cwd()),'HF_HOME':str(Path.cwd()/'.cache/huggingface'),
             'CUDA_VISIBLE_DEVICES':gpu.uuid if gpu else '',
             'VOICEHUB_SCHEDULER_LOCK_FD':str(self.lock_fd),
             'VOICEHUB_EXECUTION_CONTEXT':str(context_path.resolve()),
             'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        log=(self.run/(job.id.replace(':','--')+'.log')).open('a')
        try:p=subprocess.Popen(job.command,env=env,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,pass_fds=(self.lock_fd,))
        except BaseException:log.close();raise
        job.status='running';job.attempts+=1;job.peak_mib=0
        self.running[job.id]=(job,p,log,gpu.uuid if gpu else None,reserved)

    def reap(self,devices):
        by_uuid={g.uuid:g for g in devices}
        for key,(job,p,log,uuid,reserved) in list(self.running.items()):
            if uuid in by_uuid:job.peak_mib=max(job.peak_mib,process_memory(by_uuid[uuid],p.pid))
            code=p.poll()
            if code is None:continue
            log.close();del self.running[key]
            result={}
            if job.result_path and Path(job.result_path).exists():result=json.loads(Path(job.result_path).read_text())
            usage=result.get('resource_usage',{}).get(job.action,{})
            job.peak_mib=max(job.peak_mib,usage.get('torch_reserved_peak_mib',0)+512 if usage else 0)
            profile_key=uuid+'|'+job.profile_key if uuid else None
            old=self.profiles.get(profile_key,{})
            if profile_key and code==0 and job.peak_mib:
                self.profiles[profile_key]={**old,'peak_mib':max(job.peak_mib,old.get('peak_mib',0)),
                    'updated_at':time.time(),'measured_on':job.id}
            oom=code!=0 and result.get('error_kind')=='cuda_oom'
            retry=oom and not job.exclusive and bool(old.get('peak_mib')) and job.attempts<2 and not self.stopping
            if oom and profile_key:
                self.profiles[profile_key]={**old,'exclusive':True,'oom_job':job.id}
            if retry:job.exclusive=True;job.status='pending'
            else:job.status='completed' if code==0 else 'stopped' if self.stopping else 'failed'
            self.state['events'].append(dict(experiment=job.experiment,phase=job.phase,action=job.action,
                exit_code=code,time=time.time(),gpu_uuid=uuid,peak_process_mib=job.peak_mib,
                retry_exclusive=retry,attempt=job.attempts))
        write_json(self.profile_path,self.profiles)

    def terminate(self):
        for _,p,_,_,_ in self.running.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+8
        while any(p.poll() is None for _,p,_,_,_ in self.running.values()) and time.monotonic()<deadline:
            time.sleep(.1)
        for job,p,log,_,_ in self.running.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            p.wait();log.close();job.status='stopped'
        self.running.clear()

    def run_all(self):
        try:
            while True:
                if self.stopping:
                    self.terminate();self.save('stopped_by_user');return False
                # Fail closed on broken telemetry instead of launching unbudgeted work.
                devices=[g for g in self.telemetry() if g.uuid in self.gpu_ids]
                if {g.uuid for g in devices}!=self.gpu_ids:raise RuntimeError('Selected GPU disappeared')
                self.reap(devices)
                for job in self.jobs.values():
                    if self.stopping:break
                    if job.status!='pending':continue
                    deps=[self.jobs[d].status for d in job.dependencies]
                    if any(s in ('failed','blocked','stopped') for s in deps):job.status='blocked';continue
                    if any(s!='completed' for s in deps) or self.draining:continue
                    if job.resource!='gpu':
                        limit=self.max_cpu_jobs if job.resource=='cpu' else 1
                        if sum(j.resource==job.resource for j,_,_,_,_ in self.running.values())<limit:self.start(job)
                        continue
                    for gpu in sorted(devices,key=lambda g:g.free_mib,reverse=True):
                        running=[(j,p,r) for j,p,_,uuid,r in self.running.values() if uuid==gpu.uuid]
                        if any(j.exclusive or not self.profile(j,gpu).get('peak_mib') for j,_,_ in running):continue
                        profile=self.profile(job,gpu)
                        exclusive=self.mode=='isolated' or job.exclusive or profile.get('exclusive',False)
                        estimate=reservation(profile)
                        # A large model that fits alone must not wait forever just
                        # because the conservative sharing margin exceeds capacity.
                        if estimate is not None and estimate>gpu.total_mib-max(self.headroom,int(gpu.total_mib*.10)):
                            exclusive=True
                        reserved=admission(gpu,[(r,process_memory(gpu,p.pid)) for _,p,r in running],
                            estimate,self.headroom,self.max_gpu_jobs,exclusive)
                        if reserved is not None:
                            job.exclusive=exclusive or estimate is None
                            self.start(job,gpu,reserved);break
                pending=any(j.status=='pending' for j in self.jobs.values())
                if not self.running and (not pending or self.draining):
                    success=all(j.status=='completed' for j in self.jobs.values())
                    self.save('paused_after_jobs' if self.draining else 'completed' if success else 'needs_attention')
                    return success
                self.save('running' if self.running else 'waiting_for_vram')
                time.sleep(self.interval)
        except BaseException as exc:
            self.reason=type(exc).__name__+': '+str(exc);self.stopping=True
            self.terminate();self.save('needs_attention');raise
