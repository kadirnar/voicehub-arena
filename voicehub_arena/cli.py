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
        process = subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
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
                 asr={"checkpoint":args.asr,"revision":HfApi().model_info(args.asr).sha},
                 protocol="English diagnostic v1; primary checkpoint per registered provider; no human MOS",
                 normalization="NFKC; lowercase; punctuation removed; apostrophes joined; whitespace collapsed; CER includes spaces")
        cfg["arena_source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob("*.py")}
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
        if args.resume and result_path.exists() and read_json(result_path)["status"] in ("completed","partial","generated","blocked","failed","timeout","disk_limit","unsupported_language"):
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
                         directory/'scorer.log',7200)
                report(root)
    print("SCORING",flush=True)
    write_json(root/"state.json",{"status":"running","phase":"scoring"})
    code=isolated([sys.executable,"-m","voicehub_arena.worker","score",str(config_path)],root/"scorer.log",7200)
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
    run.add_argument('--score-each-model',action='store_true',help='Run CPU ASR after each TTS worker exits so scored results appear incrementally')
    run.add_argument("--device",default="cuda")
    run.add_argument("--asr",default="Systran/faster-whisper-small.en")
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
        from .worker import score
        score(Path(args.run)/"config.json")
        report(Path(args.run))
    elif args.command=="serve":
        from .server import create_app
        import uvicorn
        uvicorn.run(create_app(Path(args.runs).resolve()),host="127.0.0.1",port=args.port)


if __name__=="__main__":
    main()
