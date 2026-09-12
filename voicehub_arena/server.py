"""Private, read-only result explorer. Generation is controlled by the CLI."""
from pathlib import Path
import csv
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from .storage import read_json


def create_app(runs):
    runs=Path(runs).resolve()
    app=FastAPI(title="VoiceHub Arena",version="0.1.0")

    def safe_file(relative):
        path=(runs/relative).resolve()
        if not path.is_relative_to(runs) or not path.is_file():
            raise HTTPException(404)
        return path

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent/"static"/"index.html", headers={'Cache-Control':'no-store'})

    @app.get("/api/runs")
    def list_runs():
        rows=[]
        for p in sorted(runs.glob("*/config.json"),key=lambda p:p.stat().st_mtime,reverse=True):
            state=p.parent/"state.json"
            rows.append({"id":p.parent.name,**(read_json(state) if state.exists() else {"status":"unknown"})})
        rows.sort(key=lambda row: row['status']=='running',reverse=True)
        return rows

    @app.get("/api/runs/{run}")
    def get_run(run:str):
        cfg=read_json(safe_file(f"{run}/config.json"))
        results=[]
        for spec in cfg["catalog"]:
            p=runs/run/spec["model_type"]/"result.json"
            result=read_json(p) if p.exists() else {"model_type":spec["model_type"],"checkpoint":spec["checkpoint"],"status":"pending","rows":[]}
            from .metrics import summarize
            result["summary"]=summarize(result.get("rows",[]))
            results.append(result)
        return {"run":run,"config":cfg,"results":results}

    @app.get("/api/overview")
    def overview():
        # Compare one dataset/repeat/ASR protocol; never silently mix the smoke
        # test with extended evaluations or use the best score from old runs.
        configs = [(p.parent.name, read_json(p)) for p in runs.glob('*/config.json')]
        configs = [(name,cfg) for name,cfg in configs if cfg.get('protocol') != 'public-english-v2']
        if not configs:
            return {"run": "overview", "config": {"dataset": [], "repeats": 0}, "results": [], "active_runs": []}
        _, base = max(configs, key=lambda item: (len(item[1].get('dataset', [])) * item[1].get('repeats', 0),
                                                       len(item[1].get('catalog', [])),
                                                       item[1].get('started_at', 0)))
        def protocol(cfg):
            return (cfg.get('dataset'), cfg.get('repeats'), cfg.get('asr'), cfg.get('normalization'))
        matching = sorted(((name, cfg) for name, cfg in configs if protocol(cfg) == protocol(base)),
                          key=lambda item: item[1].get('started_at', 0))
        latest = {}
        active = []
        for name, cfg in matching:
            state_path = runs/name/'state.json'
            state = read_json(state_path) if state_path.exists() else {}
            if state.get('status') == 'running':
                active.append({"run": name, **state})
            for result in get_run(name)['results']:
                result['source_run'] = name
                result['planned_samples'] = len(cfg.get('dataset', [])) * cfg.get('repeats', 0)
                latest[result['model_type']] = result
        for path in runs.glob('*/queue.json'):
            if (path.parent/'config.json').exists():
                continue
            plan = read_json(path)
            if plan.get('status') not in {'waiting', 'starting'}:
                continue
            for model_type in plan.get('models', []):
                if model_type in latest:
                    latest[model_type]['queued_run'] = path.parent.name
                    latest[model_type]['current_status'] = 'queued'
        return {"run": "overview", "config": base, "results": list(latest.values()), "active_runs": active}

    @app.get('/api/overview.csv')
    def overview_csv():
        output = io.StringIO()
        columns = ['model_type', 'source_run', 'status', 'current_status', 'queued_run', 'checkpoint', 'planned_samples',
                   'scored', 'wer', 'cer', 'mer', 'wil', 'wip', 'rtf', 'latency_p50_s',
                   'latency_p95_s', 'peak_vram_mib', 'generation_failure_rate',
                   'generation_limit_rate', 'error']
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for result in overview()['results']:
            writer.writerow({**result, **result['summary']})
        return Response(output.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition': 'attachment; filename=voicehub-latest.csv'})

    @app.get('/api/public-suite')
    def public_suite():
        path = runs/'public-english-v2/suite.json'
        if not path.exists():
            return {'datasets':[], 'status':'not_started'}
        plan = read_json(path)
        return {k:plan.get(k) for k in ('datasets','status','current_job','catalog','asr','normalization_id','protocol_id')}

    @app.get('/api/public-suite/{dataset}')
    def public_dataset(dataset:str, phase:str='panel'):
        from .benchmarks import public_report
        try:
            return public_report(runs, dataset, phase)
        except (FileNotFoundError, StopIteration, ValueError) as error:
            raise HTTPException(404, str(error)) from error

    @app.get('/api/public-suite/{dataset}/leaderboard.csv')
    def public_csv(dataset:str, phase:str='panel'):
        data = public_dataset(dataset, phase)
        output = io.StringIO()
        columns = ['dataset', 'coverage_phase', 'model_type','status','ranking_eligible','planned_samples',
                   'attempted','generated','scored','wer','cer','utterance_mean_wer','utterance_mean_cer',
                   'mer','wil','wip','word_substitutions','word_deletions','word_insertions',
                   'char_substitutions','char_deletions','char_insertions','wer_ci95','cer_ci95',
                   'rtf','latency_p50_s','latency_p95_s','peak_vram_mib','generation_failure_rate',
                   'normalization_id','asr_checkpoint','asr_revision','source_runs','error']
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for result in data['results']:
            writer.writerow({**result, **result['summary'], 'dataset':dataset,'coverage_phase':phase,
                             'asr_checkpoint':data['config']['asr']['checkpoint'],
                             'asr_revision':data['config']['asr']['revision']})
        return Response(output.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition':f'attachment; filename={dataset}-{phase}.csv'})

    @app.get("/files/{relative:path}")
    def file(relative:str):
        p=safe_file(relative)
        if p.suffix not in (".wav",".json",".csv"):
            raise HTTPException(404)
        return FileResponse(p)
    return app
