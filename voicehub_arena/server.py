"""Private, read-only result explorer. Generation is controlled by the CLI."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
        return FileResponse(Path(__file__).parent/"static"/"index.html")

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

    @app.get("/files/{relative:path}")
    def file(relative:str):
        p=safe_file(relative)
        if p.suffix not in (".wav",".json",".csv"):
            raise HTTPException(404)
        return FileResponse(p)
    return app
