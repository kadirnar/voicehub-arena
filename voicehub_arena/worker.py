"""One model per subprocess; ASR runs later so it cannot contaminate TTS timing."""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .storage import write_json, read_json, read_rows


def generate(config_path, model_type):
    import torch
    import numpy as np
    import soundfile as sf
    from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig
    from .metrics import signal_metrics
    cfg = read_json(config_path)
    run = Path(config_path).parent
    output = run / model_type
    output.mkdir(exist_ok=True)
    spec = next(s for s in cfg["catalog"] if s["model_type"] == model_type)
    override = cfg["overrides"].get(model_type, {})
    checkpoint = override.get("checkpoint", spec["checkpoint"])
    status = dict(model_type=model_type, checkpoint=checkpoint, status="loading", rows=[],
                  expected_samples=len(cfg["dataset"])*cfg["repeats"])
    write_json(output/"result.json", status)
    try:
        from .catalog import declared_languages
        languages=declared_languages(model_type)
        status["declared_languages"]=languages
        if languages and not any(lang.lower().startswith("en") for lang in languages):
            status.update(status="unsupported_language",error="The pinned VoiceHub model card does not advertise English: "+", ".join(languages))
            write_json(output/"result.json",status)
            return
        if not checkpoint:
            raise ValueError("No default checkpoint configured; add a reviewed checkpoint override")
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(cfg["seed"])
        config = dict(override.get("config", {}))
        # Record and request the exact primary checkpoint revision. Local-file providers
        # are recorded as such rather than passing an unsupported revision argument.
        if "/" in checkpoint and not Path(checkpoint).exists():
            from huggingface_hub import HfApi
            revision = HfApi().model_info(checkpoint, revision=config.get("revision")).sha
            config["revision"] = revision
            status["revision"] = revision
        write_json(output/"result.json", status)
        sync = lambda: torch.cuda.synchronize() if cfg["device"].startswith("cuda") else None
        started = time.perf_counter()
        model = AutoModelForTextToSpeech.from_pretrained(checkpoint, model_type=model_type,
                                                       device=cfg["device"], **config)
        generation = override.get("generation", {})
        defaults = model.generation_config.to_dict()
        defaults.update(seed=cfg["seed"], **generation)
        prepared = model.prepare_inputs_for_generation(override.get("text_prefix", "")+cfg["dataset"][0]["text"], **defaults)
        model._validate_model_kwargs(prepared)
        model._validate_common_generation_inputs(prepared)
        model._validate_generation_inputs(prepared)
        model.load()
        # VoiceHub providers may initialize lazily. Warm-up is recorded separately.
        sync()
        status["load_s"] = time.perf_counter()-started
        status.update(status="warming_up", effective_config=config, generation=generation)
        write_json(output/"result.json",status)
        first = cfg["dataset"][0]["text"]
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(cfg["warmups"]):
                model.generate(override.get("text_prefix", "")+first,
                               generation_config=TTSGenerationConfig(seed=cfg["seed"]), **generation)
        sync()
        status["warmup_s"] = time.perf_counter()-started
        status["status"] = "generating"
        write_json(output/"result.json", status)
        for item in cfg["dataset"]:
            for repeat in range(cfg["repeats"]):
                row = dict(id=item["id"], category=item["category"], text=item["text"],
                           reference=item.get("reference",item["text"]), repeat=repeat,
                           seed=cfg["seed"]+repeat, status="failed")
                try:
                    if cfg["device"].startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                    sync()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        result = model.generate(override.get("text_prefix", "")+item["text"],
                              generation_config=TTSGenerationConfig(seed=row["seed"]), **generation)
                    sync()
                    row["latency_s"] = time.perf_counter()-started
                    row["peak_vram_mib"] = torch.cuda.max_memory_allocated()/2**20 if cfg["device"].startswith("cuda") else 0
                    audio = result.audio
                    if hasattr(audio,"detach"):
                        audio = audio.detach().float().cpu().numpy()
                    audio = np.asarray(audio).squeeze()
                    if audio.ndim == 2 and audio.shape[0] <= 2:
                        audio = audio.T
                    row.update(signal_metrics(audio, result.sample_rate))
                    row["rtf"] = row["latency_s"]/row["duration_s"]
                    row["audio"] = f"{model_type}/{item['id']}-{repeat}.wav"
                    # Float WAV preserves unclipped signal diagnostics and model output.
                    sf.write(run/row["audio"], audio, result.sample_rate, subtype="FLOAT")
                    row["audio_sha256"] = hashlib.sha256((run/row["audio"]).read_bytes()).hexdigest()
                    row["status"] = "generated"
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                status["rows"].append(row)
                write_json(output/"result.json",status)
        status["status"] = "generated" if any(r["status"]=="generated" for r in status["rows"]) else "failed"
    except Exception as error:
        status.update(status="blocked", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    write_json(output/"result.json", status)


def score(config_path):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    from .metrics import errors, summarize
    cfg = read_json(config_path)
    run = Path(config_path).parent
    # CPU scoring releases all GPU memory for TTS and works without a system cuDNN.
    asr = cfg["asr"]
    model_path = snapshot_download(asr["checkpoint"], revision=asr.get("revision"))
    model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=cfg["cpu_threads"])
    for spec in cfg["catalog"]:
        path = run/spec["model_type"]/"result.json"
        if not path.exists():
            continue
        result = read_json(path)
        for row in result.get("rows", []):
            if row["status"] not in ("generated", "scoring_failed"):
                continue
            try:
                segments, info = model.transcribe(str(run/row["audio"]), language="en", task="transcribe",
                     beam_size=5, temperature=0, condition_on_previous_text=False, vad_filter=False)
                row["transcript"] = " ".join(s.text.strip() for s in segments).strip()
                row["metrics"] = errors([row["reference"]],[row["transcript"]])
                row["status"] = "ok"
                row.pop("scoring_error",None)
            except Exception as error:
                row.update(status="scoring_failed",scoring_error=f"{type(error).__name__}: {error}")
            write_json(path,result)
        result["summary"] = summarize(result.get("rows",[]))
        if result.get("rows"):
            complete = len(result["rows"]) == result.get("expected_samples", len(cfg["dataset"])*cfg["repeats"])
            result["status"] = "completed" if complete and all(r["status"]=="ok" for r in result["rows"]) else "partial"
        write_json(path,result)


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate(sys.argv[2],sys.argv[3])
    elif sys.argv[1] == "score":
        score(sys.argv[2])
