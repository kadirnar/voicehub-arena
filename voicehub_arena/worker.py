"""One model per subprocess; ASR runs later so it cannot contaminate TTS timing."""
import importlib.metadata
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time
import traceback

from .storage import write_json, read_json, read_rows


def generate(config_path, model_type):
    from .auth import configure_hub_auth
    configure_hub_auth()
    from .transport import enable_verified_cache_reuse, resolved_artifacts
    enable_verified_cache_reuse()
    import torch
    import numpy as np
    import soundfile as sf
    from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig
    from .metrics import signal_metrics
    from .inputs import prepare_request
    cfg = read_json(config_path)
    run = Path(config_path).parent
    output = run / model_type
    output.mkdir(exist_ok=True)
    spec = next(s for s in cfg["catalog"] if s["model_type"] == model_type)
    override = cfg["overrides"].get(model_type, {})
    checkpoint = override.get("checkpoint", spec["checkpoint"])
    status = dict(model_type=model_type, checkpoint=checkpoint, status="loading", rows=[],
                  expected_samples=len(cfg["dataset"])*cfg["repeats"])
    if "artifact_provenance" in override:
        status["artifact_provenance"] = override["artifact_provenance"]
    if "runtime_adapter" in override:
        status["runtime_adapter"] = override["runtime_adapter"]
    status["timing_scope"] = override.get("timing_scope", "text preparation and synthesis; excludes model load and warm-up")
    status["runtime_packages"] = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    status["download_policy"] = "sha256-verified native cache; Hub/Xet downloads verify resolved commit, size and content digest; verified blobs share disk by hardlink with atomic-copy fallback; mutable refs are re-resolved before commit-pinned download"
    status["runtime_source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(__file__).parent.glob('*.py')}
    import voicehub
    try:
        native_diff = subprocess.check_output(['git','-C',str(Path(voicehub.__file__).parent),'diff','--binary','HEAD'])
        status['voicehub_diff_sha256'] = hashlib.sha256(native_diff).hexdigest() if native_diff else None
    except (OSError, subprocess.CalledProcessError):
        status['voicehub_diff_sha256'] = None
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
        for key, expected in override.get('artifact_provenance', {}).get('config_files_sha256', {}).items():
            artifact = Path(override.get('config', {})[key])
            digest = hashlib.sha256()
            with artifact.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*2**20), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Configured artifact differs from reviewed provenance: '+key)
        if Path(checkpoint).is_dir():
            for filename, expected in override.get('artifact_provenance', {}).get('files_sha256', {}).items():
                artifact = Path(checkpoint)/filename
                if artifact.resolve().parent != Path(checkpoint).resolve():
                    raise ValueError('Checkpoint manifest must contain flat filenames')
                digest = hashlib.sha256()
                with artifact.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(8*2**20), b''):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError('Checkpoint artifact differs from reviewed provenance: '+filename)
        if Path(checkpoint).is_file() and override.get('artifact_provenance',{}).get('sha256'):
            digest = hashlib.sha256()
            with Path(checkpoint).open('rb') as handle:
                for chunk in iter(lambda: handle.read(8*2**20),b''):
                    digest.update(chunk)
            if digest.hexdigest() != override['artifact_provenance']['sha256']:
                raise ValueError('Local checkpoint digest differs from its reviewed provenance')
        prepared_directory = override.get("prepared_inputs")
        for key, filename in override.get('generation',{}).items():
            if key.endswith('_path') and isinstance(filename,str) and filename in cfg.get('reference_sha256',{}):
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != cfg['reference_sha256'][filename]:
                    raise ValueError('Reference artifact changed after run configuration: '+filename)
        if prepared_directory:
            for filename, expected in cfg.get("prepared_inputs_sha256", {}).items():
                if Path(filename).parent == Path(prepared_directory):
                    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                        raise ValueError("Prepared input changed after the run was configured: "+filename)
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
        if model_type == "vibevoice" and override.get("runtime_adapter") == "arena-native-staged-v1":
            from .vibevoice_adapter import ArenaVibeVoice
            model = ArenaVibeVoice.from_pretrained(checkpoint, device=cfg["device"], **config)
        else:
            model = AutoModelForTextToSpeech.from_pretrained(checkpoint, model_type=model_type,
                                                           device=cfg["device"], **config)
        generation = override.get("generation", {})
        first_text, first_options = prepare_request(model_type, override.get("text_prefix", "")+cfg["dataset"][0]["text"], generation,
                                                   prepared_inputs=override.get("prepared_inputs"))
        defaults = model.generation_config.to_dict()
        defaults.update(seed=cfg["seed"], **first_options)
        prepared = model.prepare_inputs_for_generation(first_text, **defaults)
        model._validate_model_kwargs(prepared)
        model._validate_common_generation_inputs(prepared)
        model._validate_generation_inputs(prepared)
        model.load()
        status['resolved_artifacts'] = resolved_artifacts()
        # VoiceHub providers may initialize lazily. Warm-up is recorded separately.
        sync()
        status["load_s"] = time.perf_counter()-started
        status.update(status="warming_up", effective_config=config, generation=generation)
        write_json(output/"result.json",status)
        first = cfg["dataset"][0]["text"]
        def synthesize(text, seed):
            prepared_text, options = prepare_request(model_type, override.get("text_prefix", "")+text, generation,
                                                    prepared_inputs=override.get("prepared_inputs"), model=model)
            return model.generate(prepared_text, generation_config=TTSGenerationConfig(seed=seed), **options)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(cfg["warmups"]):
                synthesize(first, cfg["seed"])
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
                        result = synthesize(item["text"], row["seed"])
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
                    if model_type == 'cosyvoice':
                        count = result.metadata.get('speech_token_count')
                        row['speech_token_count'] = count
                        limit = generation.get('max_new_tokens') or model.config.generation_config['max_new_tokens']
                        row['generation_token_limit'] = limit
                        if isinstance(count, int) and count >= limit:
                            row['quality_flags'] = ['generation_limit_reached']
                    if model_type in {'dia', 'zonos2'}:
                        row['speech_token_count'] = result.metadata.get('speech_token_count')
                        row['generation_token_limit'] = result.metadata.get('generation_token_limit')
                        if model_type == 'dia':
                            row['kv_cache'] = result.metadata.get('kv_cache')
                        else:
                            row['eos_frame'] = result.metadata.get('eos_frame')
                        if result.metadata.get('generation_limit_reached'):
                            row['quality_flags'] = ['generation_limit_reached']
                    if model_type == 'kokoro':
                        row['frontend'] = {key:result.metadata.get(key) for key in
                            ('phonemes','frontend_ids','source_equivalent_g2p')}
                    # Whitelist numeric diagnostics from the reviewed staged adapter.
                    if override.get("runtime_adapter") == "arena-native-staged-v1":
                        for key in ("ttfa_s", "speech_tokens", "text_tokens_consumed", "text_tokens_total"):
                            row[key] = result.metadata.get(key)
                    row["status"] = "generated"
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                status["rows"].append(row)
                write_json(output/"result.json",status)
        status["status"] = "generated" if any(r["status"]=="generated" for r in status["rows"]) else "failed"
    except Exception as error:
        status.update(status="blocked", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    status['resolved_artifacts'] = resolved_artifacts()
    write_json(output/"result.json", status)


def score(config_path, model_type=None):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    from .metrics import errors, summarize
    cfg = read_json(config_path)
    run = Path(config_path).parent
    specifications = [s for s in cfg['catalog'] if model_type is None or s['model_type']==model_type]
    pending = False
    for spec in specifications:
        path = run/spec['model_type']/'result.json'
        if path.exists() and any(row['status'] in ('generated','scoring_failed') for row in read_json(path).get('rows',[])):
            pending = True
            break
    if not pending:
        return
    # CPU scoring releases all GPU memory for TTS and works without a system cuDNN.
    asr = cfg["asr"]
    model_path = snapshot_download(asr["checkpoint"], revision=asr.get("revision"))
    model = WhisperModel(model_path, device="cpu", compute_type="int8", cpu_threads=cfg["cpu_threads"])
    for spec in specifications:
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
        score(sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
