import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import shutil

from .storage import read_json, write_json, read_rows


def isolated(command, log, timeout):
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Runner stopped")

    with open(log, "w") as f:
        # CTranslate2 uses the CUDA libraries already supplied with PyTorch.
        # Set the loader path before the child imports it; never change drivers.
        env = dict(os.environ)
        library_dirs = list(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/*/lib'))
        if library_dirs:
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, library_dirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
        process = subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            return 124
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)


def report(run):
    from .metrics import summarize
    cfg = read_json(run/"config.json")
    results = []
    for spec in cfg["catalog"]:
        p = run/spec["model_type"]/"result.json"
        r = read_json(p) if p.exists() else {"model_type":spec["model_type"],"checkpoint":spec["checkpoint"],"status":"pending","rows":[]}
        r["summary"] = summarize(r.get("rows",[]))
        results.append(r)
    payload = {"run":run.name,"config":cfg,"results":results,"updated_at":time.time()}
    write_json(run/"report.json", payload)
    columns = ["model_type","status","checkpoint","wer","cer","mer","wil","wip","exact_match_rate",
               "word_hits","word_substitutions","word_deletions","word_insertions","reference_words",
               "scored","generated","attempted","generation_failure_rate","latency_p50_s","latency_p95_s",
               "rtf","peak_vram_mib","ttfa_p50_s","ttfa_p95_s","ttfa_samples","clipping_ratio","silence_ratio",
               "timing_scope","runtime_adapter","error"]
    with (run/"leaderboard.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({**r,**r["summary"]})
    return payload


def execute(args):
    import fcntl
    root = Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    lock = (root.parent/".gpu.lock").open("w")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another Arena run owns the GPU lock")
    config_path = root/"config.json"
    if config_path.exists():
        if not args.resume:
            raise SystemExit("Run already exists: choose a new output or pass --resume")
        cfg = read_json(config_path)
    else:
        from .catalog import discover
        from .inputs import frontend_protocol
        from huggingface_hub import HfApi
        catalog = discover()
        if args.models != "all":
            requested=set(args.models.split(","))
            unknown=requested-{s["model_type"] for s in catalog}
            if unknown:
                raise SystemExit(f"Unknown models: {unknown}")
            catalog=[s for s in catalog if s["model_type"] in requested]
        # Start with small providers, but keep every discovered provider in the run.
        priority=["neutts","vits","supertonic","kokoro","vui","inflecttts","styletts2","melotts","speecht5","gptsovits","openvoice","cosyvoice","vibevoice","outetts"]
        catalog.sort(key=lambda s:priority.index(s["model_type"]) if s["model_type"] in priority else 100)
        dataset=read_rows(args.dataset)
        if args.limit:
            dataset=dataset[:args.limit]
        if not dataset or len({x["id"] for x in dataset}) != len(dataset):
            raise SystemExit("Dataset must contain unique ids and at least one row")
        import re
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+",x["id"]) or not x.get("text","").strip() for x in dataset):
            raise SystemExit("Dataset ids must be filename-safe and text non-empty")
        if args.repeats < 1 or args.warmups < 1:
            raise SystemExit("At least one repeat and one warm-up are required")
        packages={p:importlib.metadata.version(p) for p in ["voicehub","torch","numpy","jiwer","faster-whisper"]}
        try:
            import voicehub
            revision=subprocess.check_output(["git","-C",str(Path(voicehub.__file__).parent),"rev-parse","HEAD"],text=True).strip()
            native_diff=subprocess.check_output(["git","-C",str(Path(voicehub.__file__).parent),"diff","--binary","HEAD"])
            native_diff_sha256=hashlib.sha256(native_diff).hexdigest() if native_diff else None
        except Exception:
            revision=None
            native_diff_sha256=None
        cfg=dict(catalog=catalog,dataset=dataset,dataset_sha256=hashlib.sha256(json.dumps(dataset,sort_keys=True).encode()).hexdigest(),
                 overrides=read_json(args.overrides) if args.overrides else {},
                 device=args.device,seed=args.seed,repeats=args.repeats,warmups=args.warmups,
                 timeout_s=args.timeout,cpu_threads=args.cpu_threads,packages=packages,cache_budget_gib=args.cache_budget_gib,
                 min_free_gib=args.min_free_gib,
                 incremental_scoring=args.score_each_model,
                 voicehub_commit=revision,voicehub_diff_sha256=native_diff_sha256,python=platform.python_version(),started_at=time.time(),
                 asr={"checkpoint":args.asr,"revision":HfApi().model_info(args.asr, revision=(args.asr_revision or
                      ('edaa852ec7e145841d8ffdb056a99866b5f0a478' if args.asr == 'Systran/faster-whisper-large-v3' else None))).sha,
                      'device':args.asr_device, 'compute_type':args.asr_compute_type,
                      'language':'en','beam_size':5,'temperature':0,'condition_on_previous_text':False,'vad_filter':False},
                 protocol=args.protocol_id,
                 normalization_id=args.normalization,
                 input_text_transform=args.input_text_transform,
                 frontend_protocols={s['model_type']:frontend_protocol(s['model_type']) for s in catalog},
                 resume_samples=args.resume_samples,
                 scoring_timeout_s=args.scoring_timeout,
                 normalization=("whisper-normalizer 0.1.12 EnglishTextNormalizer; corpus WER/CER; CER includes spaces"
                                if args.normalization == 'whisper_english' else
                                "NFKC; lowercase; punctuation removed; apostrophes joined; whitespace collapsed; CER includes spaces"))
        if args.dataset_manifest:
            manifest = read_json(args.dataset_manifest)
            file_sha = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
            matches = [part for part in [manifest['full'], *manifest['shards']] if part['sha256'] == file_sha]
            if not matches or len(dataset) != matches[0]['samples']:
                raise ValueError('Dataset differs from its frozen publisher manifest')
            cfg['dataset_manifest'] = manifest
            cfg['dataset_part'] = matches[0]
        cfg["arena_source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob("*.py")}
        if args.protocol_id == 'public-english-v2':
            from .benchmarks import public_overrides
            cfg['overrides'], cfg['public_models_lock_sha256'] = public_overrides(
                Path.cwd(), catalog, cfg['overrides'])
        references = {}
        for override in cfg["overrides"].values():
            for key,value in override.get("generation",{}).items():
                if key.endswith("_path") and isinstance(value,str) and Path(value).is_file():
                    references[value] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
        cfg["reference_sha256"] = references
        cfg["prepared_inputs_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for override in cfg["overrides"].values() if override.get("prepared_inputs")
            for path in Path(override["prepared_inputs"]).glob("*") if path.is_file()
        }
        try:
            cfg["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],text=True).strip()
        except Exception:
            cfg["gpu"]=None
        write_json(config_path,cfg)
    write_json(root/"state.json",{"status":"running","phase":"generation"})
    report(root)
    for spec in cfg["catalog"]:
        name=spec["model_type"]
        directory=root/name
        directory.mkdir(exist_ok=True)
        result_path=directory/"result.json"
        if args.resume and result_path.exists():
            old_status = read_json(result_path)['status']
            if old_status in ("completed","partial","generated","blocked","failed","timeout","disk_limit","unsupported_language"):
                if not (args.resume_samples and old_status in {'partial','failed','timeout','disk_limit'}):
                    continue
        from .cache import cleanup_abandoned_downloads, cleanup_dead_download_locks
        dead_locks = cleanup_dead_download_locks()
        if dead_locks:
            print('Reclaimed dead download locks:', dead_locks, flush=True)
        reclaimed = cleanup_abandoned_downloads()
        if reclaimed["files"]:
            print("Reclaimed abandoned download bytes:", reclaimed["bytes"], flush=True)
        if cfg.get("cache_budget_gib"):
            from .cache import prune_checkpoint_caches
            checkpoint = cfg["overrides"].get(name,{}).get("checkpoint",spec["checkpoint"])
            protected = [cfg['asr']['checkpoint']]
            if checkpoint:
                protected.append(checkpoint)
            pruned = prune_checkpoint_caches(cfg["cache_budget_gib"]*2**30, protected,
                                             cfg.get('min_free_gib', 8)*2**30)
            if pruned['repositories']:
                print("Reclaimed paired checkpoint caches:",json.dumps(pruned),flush=True)
        reserve = cfg.get('min_free_gib', 8)
        if shutil.disk_usage(root).free < reserve*2**30:
            write_json(result_path,{"model_type":name,"checkpoint":spec["checkpoint"],"status":"disk_limit","error":f"Less than {reserve:g} GiB free; checkpoint download was not started","rows":[]})
            report(root)
            continue
        print(f"GENERATE {name}",flush=True)
        write_json(root/"state.json",{"status":"running","phase":"generation","model":name})
        code=isolated([sys.executable,"-m","voicehub_arena.worker","generate",str(config_path),name],directory/"worker.log",cfg["timeout_s"])
        if code:
            r=read_json(result_path) if result_path.exists() else {"model_type":name,"rows":[]}
            r.update(status="timeout" if code==124 else "failed",error=f"Worker exit {code}; see worker.log")
            write_json(result_path,r)
        report(root)
        if cfg.get('incremental_scoring') and result_path.exists():
            produced = read_json(result_path)
            if any(row.get('status')=='generated' for row in produced.get('rows',[])):
                print(f'SCORE {name}',flush=True)
                write_json(root/'state.json',{'status':'running','phase':'scoring','model':name})
                isolated([sys.executable,'-m','voicehub_arena.worker','score',str(config_path),name],
                         directory/'scorer.log',cfg.get('scoring_timeout_s',7200))
                report(root)
    print("SCORING",flush=True)
    write_json(root/"state.json",{"status":"running","phase":"scoring"})
    code=isolated([sys.executable,"-m","voicehub_arena.worker","score",str(config_path)],root/"scorer.log",cfg.get('scoring_timeout_s',7200))
    payload=report(root)
    write_json(root/"state.json",{"status":"finished" if code==0 else "scoring_failed","phase":"done","scorer_exit":code})
    print(json.dumps({r["model_type"]:r["status"] for r in payload["results"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description="English TTS benchmark for the VoiceHub registry")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("catalog")
    run=sub.add_parser("run")
    run.add_argument("--models",default="all")
    run.add_argument("--dataset",default="datasets/english.jsonl")
    run.add_argument("--overrides",default="configs/models.json")
    run.add_argument("--output",required=True)
    run.add_argument("--repeats",type=int,default=3)
    run.add_argument("--warmups",type=int,default=1)
    run.add_argument("--seed",type=int,default=42)
    run.add_argument("--timeout",type=int,default=900)
    run.add_argument("--cpu-threads",type=int,default=4)
    run.add_argument("--cache-budget-gib",type=float,help="Prune paired native/Hub checkpoint caches between models; requires dedicated HF_HOME")
    run.add_argument("--min-free-gib",type=float,default=32,help="Free disk reserve before each model download (default: 32 GiB)")
    run.add_argument('--score-each-model',action='store_true',help='Run the configured ASR after each TTS worker exits')
    run.add_argument("--device",default="cuda")
    run.add_argument("--asr",default="Systran/faster-whisper-large-v3")
    run.add_argument('--asr-revision')
    run.add_argument('--asr-device', choices=['cpu','cuda'], default='cuda')
    run.add_argument('--asr-compute-type', default='float16')
    run.add_argument('--scoring-timeout', type=int, default=14400)
    run.add_argument('--normalization', choices=['orthographic','whisper_english'], default='whisper_english')
    run.add_argument('--input-text-transform', choices=['identity','librispeech_lowercase_v1'], default='identity')
    run.add_argument('--protocol-id', default='English diagnostic v2; Whisper large-v3; no human MOS')
    run.add_argument('--dataset-manifest')
    run.add_argument('--resume-samples', action='store_true')
    run.add_argument("--limit",type=int)
    run.add_argument("--resume",action="store_true")
    rep=sub.add_parser("report")
    rep.add_argument("run")
    score=sub.add_parser("score")
    score.add_argument("run")
    serve=sub.add_parser("serve")
    serve.add_argument("--runs",default="runs")
    serve.add_argument("--port",type=int,default=7860)
    args=parser.parse_args()
    if args.command=="catalog":
        from .catalog import discover
        print(json.dumps(discover(),indent=2))
    elif args.command=="run":
        try:
            execute(args)
        except (Exception, KeyboardInterrupt) as error:
            write_json(Path(args.output)/"state.json",{"status":"failed","error":f"{type(error).__name__}: {error}"})
            raise
    elif args.command=="report":
        report(Path(args.run))
    elif args.command=="score":
        import fcntl
        root = Path(args.run).resolve()
        with (root.parent/'.gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            cfg = read_json(root/'config.json')
            code = isolated([sys.executable,'-m','voicehub_arena.worker','score',str(root/'config.json')],
                            root/'rescorer.log',cfg.get('scoring_timeout_s',14400))
            report(root)
            if code:
                raise SystemExit(code)
    elif args.command=="serve":
        from .server import create_app
        import uvicorn
        uvicorn.run(create_app(Path(args.runs).resolve()),host="127.0.0.1",port=args.port)


if __name__=="__main__":
    main()
