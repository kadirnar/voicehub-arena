"""Thin calls into publisher libraries. Settings not specified here retain upstream defaults.

Adapters return an iterator of (mono waveform, sample rate). Streaming timings
measure real yielded audio; reference preprocessing is included in total latency.
"""
from pathlib import Path


def snapshot(spec):
    # Serialize staging admission so two concurrent model loads cannot both
    # reserve the same free disk space before downloading large checkpoints.
    import fcntl
    Path('.cache').mkdir(exist_ok=True)
    with Path('.cache/native-download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _snapshot(spec)


def _snapshot(spec):
    import fnmatch
    import hashlib
    import os
    import shutil
    from huggingface_hub import HfApi,snapshot_download
    info=HfApi().model_info(spec['repo'],revision=spec['revision'],files_metadata=True)
    allow=spec.get('allow_patterns');ignore=spec.get('ignore_patterns',[])
    wanted=[f for f in info.siblings if (not allow or any(fnmatch.fnmatch(f.rfilename,p) for p in allow))
            and not any(fnmatch.fnmatch(f.rfilename,p) for p in ignore)]
    home=Path(os.environ['HF_HOME'])
    shared=home/'hub'/('models--'+spec['repo'].replace('/','--'))/'snapshots'/spec['revision']
    if wanted and all((shared/f.rfilename).is_file() and (shared/f.rfilename).stat().st_size==f.size for f in wanted):
        return str(shared)
    key=hashlib.sha256((spec['repo']+'@'+spec['revision']).encode()).hexdigest()[:20]
    cache=Path('.cache/native-staging')/key
    cache.mkdir(parents=True,exist_ok=True)
    cached=cache/('models--'+spec['repo'].replace('/','--'))/'snapshots'/spec['revision']
    missing=sum(f.size or 0 for f in wanted if not (cached/f.rfilename).is_file())
    if shutil.disk_usage(cache).free<missing+2*2**30:
        raise RuntimeError(f'Insufficient staging space for {spec["repo"]}: {missing/2**30:.2f} GiB plus 2 GiB reserve required')
    return snapshot_download(spec['repo'],revision=spec['revision'],cache_dir=cache,
                             allow_patterns=allow,ignore_patterns=ignore)


def load(spec):
    import torch
    backend = spec['backend']
    method = spec['method']
    settings = spec.get('settings', {})
    path = snapshot(spec)
    if backend == 'kokoro':
        from kokoro import KModel, KPipeline
        model = KModel(config=str(Path(path)/'config.json'),
                       model=str(Path(path)/'kokoro-v1_0.pth')).eval().cuda()
        pipe = KPipeline(lang_code='a', model=model, device='cuda')
        voice = torch.load(Path(path)/'voices'/f"{settings['voice']}.pt", weights_only=True)
        def generate(row):
            for result in pipe(row['text'], voice=voice):
                yield result.audio, 24000
    elif backend == 'transformers_vits':
        from transformers import VitsModel, AutoTokenizer
        model = VitsModel.from_pretrained(path).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path)
        def generate(row):
            inputs = tokenizer(row['text'], return_tensors='pt').to('cuda')
            yield model(**inputs).waveform[0], model.config.sampling_rate
    elif backend == 'transformers_speecht5':
        import numpy as np
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
        model = SpeechT5ForTextToSpeech.from_pretrained(path).eval().cuda()
        processor = SpeechT5Processor.from_pretrained(path)
        vocoder = SpeechT5HifiGan.from_pretrained(snapshot(spec['vocoder'])).eval().cuda()
        speaker = torch.from_numpy(np.load(settings['embedding'])).reshape(1, -1).float().cuda()
        def generate(row):
            inputs = processor(text=row['text'], return_tensors='pt').to('cuda')
            yield model.generate_speech(inputs['input_ids'], speaker, vocoder=vocoder), 16000
    elif backend == 'qwen3tts':
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(path, device_map='cuda:0',
                                             dtype=torch.bfloat16, attn_implementation='sdpa')
        def generate(row):
            kwargs = dict(text=row['text'], language='English')
            if method == 'custom_voice':
                wavs, sr = model.generate_custom_voice(**kwargs, speaker=settings['speaker'])
            elif method == 'voice_design':
                wavs, sr = model.generate_voice_design(**kwargs, instruct=settings['description'])
            elif method in ('voice_clone','x_vector_only'):
                wavs, sr = model.generate_voice_clone(**kwargs, ref_audio=row['reference_audio'],
                    ref_text=row['reference_text'] if method == 'voice_clone' else None,
                    x_vector_only_mode=method == 'x_vector_only')
            else:
                raise ValueError('Unsupported Qwen native method: '+method)
            yield wavs[0], sr
    elif backend == 'omnivoice':
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained(path, device_map='cuda', dtype=torch.float16)
        def generate(row):
            kwargs = dict(text=row['text'])
            if method == 'voice_clone':
                kwargs.update(ref_audio=row['reference_audio'], ref_text=row['reference_text'])
            elif method == 'voice_design':
                kwargs['instruct'] = settings['description']
            elif method != 'auto_voice':
                raise ValueError('Unsupported OmniVoice method')
            yield model.generate(**kwargs)[0], 24000
    elif backend == 'voxcpm':
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained(path, load_denoiser=False)
        def generate(row):
            kwargs = dict(text=row['text'], seed=42)
            if method == 'voice_design':
                kwargs['text'] = '('+settings['description']+')'+row['text']
            if method in ('voice_clone','ultimate_clone'):
                kwargs['reference_wav_path'] = row['reference_audio']
            if method in ('continuation','ultimate_clone'):
                kwargs.update(prompt_wav_path=row['reference_audio'], prompt_text=row['reference_text'])
            if spec.get('streaming'):
                for chunk in model.generate_streaming(**kwargs):
                    yield chunk, model.tts_model.sample_rate
            else:
                yield model.generate(**kwargs), model.tts_model.sample_rate
    elif backend == 'parlertts':
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(path).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path)
        description = tokenizer(settings['description'], return_tensors='pt').to('cuda')
        def generate(row):
            text = tokenizer(row['text'], return_tensors='pt').to('cuda')
            audio = model.generate(input_ids=description.input_ids,
                attention_mask=description.attention_mask, prompt_input_ids=text.input_ids,
                prompt_attention_mask=text.attention_mask)
            yield audio[0], model.config.sampling_rate
    else:
        from .native_extra import load_extra
        return load_extra(spec,path)
    return generate
