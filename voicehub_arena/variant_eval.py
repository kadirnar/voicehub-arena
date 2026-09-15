"""Revision-pinned reference-conditioned Llasa/Dia experiments, isolated from old runs.

Run generation and ASR in separate processes. Pilots never enter the full leaderboard.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback

# The Vast base image exports a different HF_HOME; this project has a protected credential/cache store.
os.environ['HF_HOME'] = str(Path.cwd()/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from .storage import write_json


NO_AUDIO_POLICY = 'empty-output-deletions-v1'
SCORED_STATUSES = ('ok', 'generation_failed_scored')


class NoSpeechOutput(RuntimeError):
    """A completed LM generation emitted no speech tokens; not an infrastructure failure."""
    def __init__(self, diagnostics):
        super().__init__('LLaSA emitted no speech tokens')
        self.diagnostics = diagnostics


def validate_scored_row(row):
    if row['status'] == 'generation_failed_scored':
        if row.get('transcript') != '' or row.get('audio') or row.get('failure_policy') != NO_AUDIO_POLICY:
            raise ValueError('Invalid no-audio failure score')
        if row.get('failure_kind') != 'no_speech_tokens':
            raise ValueError('Only verified empty LM output uses the deletion policy')
    elif row['status'] != 'ok':
        raise ValueError('Unscored row')


def summarize_variant(rows):
    from .metrics import summarize
    for row in rows: validate_scored_row(row)
    result = summarize(rows, scored_statuses=SCORED_STATUSES)
    failures = sum(r['status'] == 'generation_failed_scored' for r in rows)
    result.update(generation_failures=failures, asr_scored=result['generated'], failure_policy=NO_AUDIO_POLICY)
    if failures:
        # Also expose the successful-output-only rate, never as the primary corpus score.
        result['successful_audio_only'] = summarize([r for r in rows if r['status']=='ok'])
    return result


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_campaign(path):
    cfg = json.loads(Path(path).read_text())
    if digest(cfg['dataset']) != cfg['dataset_sha256']:
        raise ValueError('Frozen dataset changed')
    if digest(cfg['reference']['path']) != cfg['reference']['sha256']:
        raise ValueError('Frozen reference changed')
    rows = [json.loads(s) for s in Path(cfg['dataset']).read_text().splitlines()]
    if len(rows) != cfg['expected_samples'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Dataset coverage/unique IDs changed')
    return cfg, rows


def llasa_chat(text, reference_text, prefix):
    # A word boundary is required when the last reference word has no punctuation.
    joined = (reference_text.rstrip() + ' ' + text.lstrip()) if reference_text else text
    return [dict(role='user', content='Convert the text to speech:<|TEXT_UNDERSTANDING_START|>' + joined + '<|TEXT_UNDERSTANDING_END|>'),
            dict(role='assistant', content='<|SPEECH_GENERATION_START|>' + ''.join(f'<|s_{i}|>' for i in prefix))]


def snapshot(repo, revision, *, stage_id=None):
    from huggingface_hub import snapshot_download
    patterns = ['*.json', '*.safetensors', '*.model', '*.txt', '*.bin']
    # Reuse complete installed snapshots; partial snapshots must be completed.
    if stage_id is None:
        return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
    cache = Path('.cache/variant-staging') / stage_id
    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, cache_dir=str(cache)))



def verify_snapshot_files(path, repo, revision):
    from huggingface_hub import HfApi
    files=[p for p in sorted(path.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors','.model')]
    info={r.path:r for r in HfApi().get_paths_info(repo,paths=[p.name for p in files],revision=revision)}
    hashes={}
    for p in files:
        expected=info[p.name]
        if p.stat().st_size!=expected.size: raise ValueError('Checkpoint size mismatch: '+p.name)
        sha=digest(p)
        if expected.lfs:
            if sha!=expected.lfs.sha256: raise ValueError('Checkpoint SHA-256 mismatch: '+p.name)
        else:
            raw=p.read_bytes()
            git_hash=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if git_hash!=expected.blob_id: raise ValueError('Checkpoint Git blob mismatch: '+p.name)
        hashes[p.name]=sha
    return hashes


def prepare_llasa(spec, cfg, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from voicehub import AutoModelForTextToSpeech
    # The separately audited native codec is frozen. Use an independent HF LM/tokenizer.
    wrapper = AutoModelForTextToSpeech.from_pretrained('HKUSTAudio/Llasa-1B-Multilingual', model_type='llasa', device='cuda', revision='7f094cb62b0a9779b334c60d039a61c5a6e04456')
    wrapper.load()
    codec = wrapper.codec
    prefix, prefix_samples = (wrapper._encode_reference(cfg['reference']['path']) if spec['conditioning'] != 'none' else ([], 0))
    codec_meta = {'codec_config_sha256': digest(wrapper.codec_artifacts.config), 'codec_weights_sha256': digest(wrapper.codec_artifacts.checkpoint), 'prefix_tokens': len(prefix), 'crop_samples': prefix_samples}
    if prefix:
        # Reconstruction is an encoder/decoder control, independent of the language model.
        with torch.inference_mode():
            reconstructed = codec.decode_code(torch.tensor(prefix, device='cuda').view(1, 1, -1))[0, 0].float().cpu().numpy()
        import soundfile as sf
        sf.write(out/'reference-reconstruction.wav', reconstructed, 16000, subtype='FLOAT')
    wrapper.model = None
    gc.collect(); torch.cuda.empty_cache()
    # The original multilingual checkpoint already exists in the shared cache.
    stage = None if spec['repo'] == wrapper.default_model_name_or_path else spec['id']
    path = snapshot(spec['repo'], spec['revision'], stage_id=stage)
    model_hashes=verify_snapshot_files(path,spec['repo'],spec['revision'])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).eval().cuda()
    end = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
    metadata = dict(backend='Transformers Llama + independently audited native XCodec2', dtype='bfloat16 LM / float32 codec', max_length=2048, temperature=.8, top_p=1., top_k=50, checkpoint_files_sha256=model_hashes, **codec_meta)
    def synth(text):
        ref = cfg['reference']['text'] if prefix else ''
        ids = tokenizer.apply_chat_template(llasa_chat(text, ref, prefix), tokenize=True, return_tensors='pt', continue_final_message=True).cuda()
        if ids.shape[1] >= 2048:
            raise ValueError('Reference plus target exceeds released 2048-token context')
        generated = model.generate(ids, attention_mask=torch.ones_like(ids), max_length=2048, eos_token_id=end, pad_token_id=end, do_sample=True, temperature=.8, top_p=1., top_k=50, use_cache=True)[0, ids.shape[1]:]
        strings = tokenizer.convert_ids_to_tokens(generated.tolist())
        if not any(token.startswith('<|s_') and token.endswith('|>') for token in strings):
            raise NoSpeechOutput(dict(generated_token_ids=generated.tolist(), eos_token_id=end, prompt_tokens=ids.shape[1], ended_with_eos=bool(len(generated) and generated[-1]==end)))
        codes = wrapper._extract_speech_ids(strings)
        audio = codec.decode_code(torch.tensor(prefix+codes, device='cuda').view(1, 1, -1))[0, 0, prefix_samples:]
        return audio, 16000, dict(speech_token_count=len(codes), prompt_tokens=ids.shape[1], generation_limit_reached=len(generated)+ids.shape[1]>=2048, ended_with_eos=bool(generated[-1]==end))
    return synth, metadata


def prepare_dia(spec, cfg, out):
    import torch
    import soundfile as sf
    import librosa
    from transformers import DiaProcessor, DiaFeatureExtractor, AutoTokenizer, DiaForConditionalGeneration, DacModel
    from voicehub.architectures.dac.checkpoint import DESCRIPT_DAC_44KHZ_REVISION
    model = DiaForConditionalGeneration.from_pretrained(spec['repo'], revision=spec['revision'], torch_dtype=torch.float32, attn_implementation='eager').eval().cuda()
    processor = DiaProcessor(
        feature_extractor=DiaFeatureExtractor.from_pretrained(spec['repo'], revision=spec['revision']),
        tokenizer=AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision']),
        audio_tokenizer=DacModel.from_pretrained('descript/dac_44khz', revision=DESCRIPT_DAC_44KHZ_REVISION))
    processor.audio_tokenizer.to('cuda').eval()
    audio, sr = sf.read(cfg['reference']['path'], dtype='float32')
    if audio.ndim == 2: audio = audio.mean(1)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
    def synth(text):
        conditioned = spec['conditioning'] != 'none'
        combined = '[S1] ' + (cfg['reference']['text'].rstrip()+' ' if conditioned else '') + text
        inputs = processor(text=[combined], audio=audio if conditioned else None, padding=True, return_tensors='pt').to('cuda')
        plen = processor.get_audio_prompt_len(inputs['decoder_attention_mask']) if conditioned else None
        # Context includes the prompt. Keep the released maximum total context.
        budget = 3072-inputs['decoder_input_ids'].shape[1]
        if budget < 128: raise ValueError('Dia reference exhausts context')
        tokens = model.generate(**inputs, max_new_tokens=budget, do_sample=True, temperature=1.8, top_k=50, top_p=.9, guidance_scale=3., use_cache=True)
        wave = processor.batch_decode(tokens, audio_prompt_len=plen)[0]
        return wave, 44100, {'generation_limit_reached':tokens.shape[1]>=3072, 'audio_prompt_frames':None if plen is None else int(plen)}
    return synth, dict(backend='Independent Transformers Dia processor + LM + DAC', dtype='float32', temperature=1.8, top_k=50, top_p=.9, guidance_scale=3., max_total_tokens=3072)


def prepare_dia2(spec, cfg, out):
    import torch
    from dia2 import Dia2, GenerationConfig, SamplingConfig, PrefixConfig
    from dia2.runtime.voice_clone import build_prefix_plan, WhisperWord
    import dia2.engine as engine
    path = snapshot(spec['repo'], spec['revision'], stage_id=spec['id'])
    model_hashes=verify_snapshot_files(path,spec['repo'],spec['revision'])
    deps = cfg['dependencies']; mimi = snapshot(deps['mimi_repo'], deps['mimi_revision'])
    model = Dia2.from_local(path/'config.json', path/'model.safetensors', tokenizer_id=path, mimi_id=str(mimi), device='cuda', dtype='bfloat16')
    runtime = model._ensure_runtime()
    from safetensors import safe_open
    with safe_open(path/'model.safetensors', framework='pt') as weights:
        weight_keys=set(weights.keys())
        missing=[name for name,_ in runtime.model.named_parameters() if name not in weight_keys]
        if missing: raise ValueError('Dia2 released weights omit parameters: '+str(missing[:5]))
        mismatched=[name for name,p in runtime.model.named_parameters() if tuple(weights.get_slice(name).get_shape())!=tuple(p.shape)]
        if mismatched: raise ValueError('Dia2 weight shape mismatch: '+str(mismatched[:5]))
    words_data = json.loads(Path('runs/'+cfg['campaign']+'/reference-words.json').read_text())
    if words_data['reference_sha256'] != cfg['reference']['sha256']: raise ValueError('Prefix alignment belongs to different audio')
    words = [WhisperWord(**w) for w in words_data['words']]
    prefix_cfg = PrefixConfig(speaker_1=cfg['reference']['path'], include_audio=False)
    plan = build_prefix_plan(runtime, prefix_cfg, transcribe_fn=lambda path,device: words)
    # Cache exactly the official prefix planner's output; no per-sentence ASR/model loading.
    original = engine.build_prefix_plan
    engine.build_prefix_plan = lambda rt,prefix: plan if prefix else None
    generation = GenerationConfig(cfg_scale=2., audio=SamplingConfig(temperature=.8, top_k=50), prefix=prefix_cfg, use_cuda_graph=True)
    def synth(text):
        result = model.generate('[S1] '+text, config=generation, verbose=False)
        return result.waveform, result.sample_rate, {'audio_token_frames':result.audio_tokens.shape[-1], 'timestamps':result.timestamps}
    return synth, dict(backend='Official nari-labs/dia2', source_revision=deps['dia2_revision'], compatibility_patch='dia2-torch28-cudnn.patch', runtime_context_sha256=digest(Path(engine.__file__).parent/'runtime/context.py'), dtype='bfloat16', cfg_scale=2., audio_temperature=.8, audio_top_k=50, text_temperature=.6, text_top_k=50, use_cuda_graph=True, prefix_alignment='pinned Whisper-large-v3 word timestamps, cached once', prefix_frames=plan.aligned_frames, parameter_coverage_verified=True, weight_tensor_count=len(weight_keys), checkpoint_files_sha256=model_hashes)


def generate(manifest, model_id, phase, no_reference=False):
    import numpy as np
    import torch
    import soundfile as sf
    from .metrics import signal_metrics
    cfg, dataset = load_campaign(manifest)
    spec = dict(next(m for m in cfg['models'] if m['id']==model_id))
    if no_reference: spec['conditioning'] = 'none'
    selected = dataset if phase == 'full' else [dataset[i] for i in cfg['pilot_indices']]
    run = Path('runs')/cfg['campaign']/phase
    variant = model_id + ('-unconditioned' if no_reference else '')
    out = run/variant; out.mkdir(parents=True,exist_ok=True)
    path = out/'result.json'
    status = json.loads(path.read_text()) if path.exists() else dict(model_type=variant, checkpoint=spec['repo'], revision=spec['revision'], phase=phase, conditioning=spec['conditioning'], expected_samples=len(selected), rows=[], config_sha256=digest(manifest))
    if status['config_sha256'] != digest(manifest): raise ValueError('Frozen run configuration changed')
    previous = {r['id']:r for r in status['rows'] if r['status'] in ('generated','ok','generation_failed','generation_failed_scored')}
    for row in previous.values():
        if row['status'].startswith('generation_failed'):
            if row.get('failure_policy') != NO_AUDIO_POLICY: raise ValueError('Saved failure policy changed')
        elif digest(run/row['audio']) != row['audio_sha256']: raise ValueError('Saved audio changed')
    status['rows'] = list(previous.values())
    if len(previous)==len(selected): return
    if status.get('error'):
        status.setdefault('previous_attempt_errors',[]).append(status.pop('error'))
    status.update(status='loading', started_at=status.get('started_at',time.time()))
    write_json(path,status)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    try:
        synth, effective = {'hf_llasa':prepare_llasa,'hf_dia':prepare_dia,'official_dia2':prepare_dia2}[spec['backend']](spec,cfg,out)
        status.update(effective=effective, runner_sha256=digest(__file__), packages={k:importlib.metadata.version(k) for k in ('torch','transformers','huggingface-hub','soundfile')}, status='generating')
        write_json(path,status)
        # One separately timed warmup. Reset the seed for every measured prompt.
        try:
            with torch.inference_mode(): synth(selected[0]['text'])
        except NoSpeechOutput:
            pass  # Warmup is not a measured prompt or a successful sample.
        for item in selected:
            if item['id'] in previous: continue
            if shutil.disk_usage(run).free < 2*2**30: raise RuntimeError('2 GiB output reserve reached')
            torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            try:
                with torch.inference_mode(): wave,sr,diagnostics=synth(item['text'])
            except NoSpeechOutput as exc:
                torch.cuda.synchronize()
                status['rows'].append({**item,'repeat':0,'seed':cfg['seed'],'status':'generation_failed','latency_s':time.perf_counter()-start,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,'conditioning':spec['conditioning'],'normalization_id':cfg['normalization_id'],'failure_policy':NO_AUDIO_POLICY,'failure_kind':'no_speech_tokens','error':str(exc),**exc.diagnostics})
                write_json(path,status)
                print(json.dumps({'model':variant,'phase':phase,'attempted':len(status['rows']),'no_audio':item['id']}),flush=True)
                continue
            torch.cuda.synchronize();latency=time.perf_counter()-start
            wave=wave.detach().float().cpu().numpy().reshape(-1) if hasattr(wave,'detach') else np.asarray(wave).reshape(-1)
            row={**item,'repeat':0,'seed':cfg['seed'],'status':'generated','latency_s':latency,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,'conditioning':spec['conditioning'],'normalization_id':cfg['normalization_id'],**signal_metrics(wave,sr),**diagnostics}
            row['audio']=variant+'/'+item['id']+'.wav';row['rtf']=latency/row['duration_s']
            sf.write(run/row['audio'],wave,sr,subtype='FLOAT');row['audio_sha256']=digest(run/row['audio'])
            status['rows'].append(row);write_json(path,status)
            print(json.dumps({'model':variant,'phase':phase,'generated':len(status['rows']),'expected':len(selected),'latency_s':round(latency,2)}),flush=True)
        status['status']='generated'
    except Exception as exc:
        status.update(status='failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc()
        raise
    finally:
        status['updated_at']=time.time();write_json(path,status)


def score(manifest,model_id,phase,no_reference=False):
    from faster_whisper import WhisperModel
    from .metrics import errors
    cfg,_=load_campaign(manifest);variant=model_id+('-unconditioned' if no_reference else '')
    run=Path('runs')/cfg['campaign']/phase;path=run/variant/'result.json';result=json.loads(path.read_text())
    asr=cfg['asr'];model=WhisperModel(str(snapshot(asr['checkpoint'],asr['revision'])),device='cuda',compute_type='float16',cpu_threads=4)
    result['asr']=asr
    for row in result['rows']:
        if row['status'] in SCORED_STATUSES:
            validate_scored_row(row); continue
        if row['status']=='generation_failed':
            row.update(transcript='',status='generation_failed_scored',asr_latency_s=0.,asr_skipped_reason='No audio exists; fixed empty-hypothesis deletion penalty')
            validate_scored_row(row)
            row['metrics']=errors([row['reference']],[''],normalization='whisper_english')
            write_json(path,result);continue
        if digest(run/row['audio'])!=row['audio_sha256']: raise ValueError('Audio hash mismatch before ASR')
        start=time.perf_counter();segments,_=model.transcribe(str(run/row['audio']),language='en',task='transcribe',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False)
        row['transcript']=' '.join(s.text.strip() for s in segments).strip();row['metrics']=errors([row['reference']],[row['transcript']],normalization='whisper_english');row['status']='ok';row['asr_latency_s']=time.perf_counter()-start
        write_json(path,result)
    result['summary']=summarize_variant(result['rows']);result['status']='completed' if len(result['rows'])==result['expected_samples'] and all(r['status'] in SCORED_STATUSES for r in result['rows']) else 'partial';result['updated_at']=time.time()
    if result['status']=='completed' and result.get('error'):
        result.setdefault('previous_attempt_errors',[]).append(result.pop('error'))
    write_json(path,result)
    print(json.dumps({'model':variant,'phase':phase,'status':result['status'],'summary':result['summary']}),flush=True)


def align_reference(manifest):
    from faster_whisper import WhisperModel
    cfg,_=load_campaign(manifest);asr=cfg['asr'];model=WhisperModel(str(snapshot(asr['checkpoint'],asr['revision'])),device='cuda',compute_type='float16',cpu_threads=4)
    segments,_=model.transcribe(cfg['reference']['path'],language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
    words=[{'text':w.word.strip(),'start':w.start,'end':w.end} for s in segments for w in s.words if w.word.strip()]
    if not words: raise ValueError('No prefix word alignment')
    write_json(Path('runs')/cfg['campaign']/'reference-words.json',dict(reference_sha256=cfg['reference']['sha256'],asr=asr,words=words))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','score','align']);p.add_argument('--manifest',default='configs/variant-campaign.json');p.add_argument('--model');p.add_argument('--phase',choices=['pilot','full'],default='pilot');p.add_argument('--no-reference',action='store_true');a=p.parse_args()
    from .auth import configure_hub_auth
    from .transport import enable_verified_cache_reuse
    configure_hub_auth();enable_verified_cache_reuse()
    if a.action=='align': align_reference(a.manifest)
    else: (generate if a.action=='generate' else score)(a.manifest,a.model,a.phase,a.no_reference)
