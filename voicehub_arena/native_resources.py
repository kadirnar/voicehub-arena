"""VRAM admission using live device memory and reservations for starting workers."""
from dataclasses import dataclass, field
import csv
import io
import os
import subprocess


@dataclass
class GPU:
    index: str
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    processes: dict = field(default_factory=dict)


def read_gpus():
    def query(arguments):
        p=subprocess.run(['nvidia-smi',*arguments,'--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True,timeout=15)
        return list(csv.reader(io.StringIO(p.stdout),skipinitialspace=True))
    devices=[GPU(i,u,n,int(t),int(f)) for i,u,n,t,f in query(
        ['--query-gpu=index,uuid,name,memory.total,memory.free'])]
    by_uuid={g.uuid:g for g in devices}
    for uuid,pid,memory in query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory']):
        if uuid in by_uuid:
            # Unknown/MIG process accounting is unsafe for concurrent admission.
            if not memory.isdigit():raise RuntimeError('GPU process memory unavailable: '+memory)
            by_uuid[uuid].processes[int(pid)]=int(memory)
    return devices


def select_gpus(devices,selection='auto',visible=None):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    def resolve(tokens,pool):
        chosen=[]
        for token in tokens:
            matches=[g for g in pool if token==g.index or token==g.uuid or (token.startswith('GPU-') and g.uuid.startswith(token))]
            if len(matches)!=1:raise ValueError('Unknown or ambiguous GPU: '+token)
            if matches[0] not in chosen:chosen.append(matches[0])
        return chosen
    permitted=devices if visible is None else resolve([x.strip() for x in visible.split(',') if x.strip()],devices)
    chosen=permitted if selection=='auto' else resolve(selection.split(','),permitted)
    if not chosen:raise ValueError('No visible CUDA GPU; set CUDA_VISIBLE_DEVICES or --gpus')
    return chosen


def process_memory(gpu,pid):
    """Include native engines that create children in the worker's process group."""
    total=0
    for candidate,memory in gpu.processes.items():
        try:belongs=candidate==pid or os.getpgid(candidate)==pid
        except ProcessLookupError:belongs=False
        if belongs:total+=memory
    return total


def reservation(profile,margin=1.35):
    if not profile or not profile.get('peak_mib'):return None
    return int(profile['peak_mib']*margin+1024)


def admission(gpu,active,estimate,headroom_mib=2048,max_jobs=4,exclusive=False):
    """Return reservation MiB, or None. active contains (reserved, observed) pairs.

    memory.free already subtracts resident processes. Subtract only the unused
    part of every outstanding reservation, including workers still loading.
    """
    headroom=max(headroom_mib,int(gpu.total_mib*.10))
    if len(active)>=max_jobs:return None
    if exclusive or estimate is None:
        if active or gpu.free_mib-headroom<1024:return None
        return gpu.free_mib-headroom
    promised=sum(max(0,reserved-observed) for reserved,observed in active)
    if estimate>gpu.free_mib-headroom-promised:return None
    return estimate
