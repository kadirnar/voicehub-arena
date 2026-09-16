"""Independent author-provided ASR, MOS and speaker-verification models."""
import importlib.util
import json
import os
from pathlib import Path
import sys


def audio16(path):
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.transforms import Resample
    audio,sr=sf.read(path,dtype='float32',always_2d=True)
    audio=audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Quality metrics require nonempty finite audio')
    return Resample(sr,16000)(torch.from_numpy(audio)).numpy() if sr!=16000 else audio


def make_scorer(action,cfg):
    from .native_protocol import digest
    if action=='asr':
        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel
        from .metrics import errors
        spec=cfg['asr'];path=snapshot_download(spec['checkpoint'],revision=spec['revision'])
        model=WhisperModel(path,device='cuda',compute_type='float16',cpu_threads=4)
        def score(path,row):
            segments,_=model.transcribe(str(path),language='en',task='transcribe',beam_size=5,
                temperature=0,condition_on_previous_text=False,vad_filter=False)
            transcript=' '.join(s.text.strip() for s in segments).strip()
            return {'transcript':transcript,**errors([row['reference']],[transcript],normalization='whisper_english')}
        return score,{**spec,'language':'en','normalization':'whisper-normalizer==0.1.12 EnglishTextNormalizer on both reference and hypothesis'}
    manifest=json.loads(Path('artifacts/native-metrics/manifest.json').read_text())
    spec=manifest[action];root=Path(spec['path'])
    for name,sha in spec['files_sha256'].items():
        if digest(root/name)!=sha:raise ValueError('Metric source/checkpoint digest changed: '+name)
    if action=='dnsmos':
        module_spec=importlib.util.spec_from_file_location('author_dnsmos',root/'dnsmos_local.py')
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        # A100 hosts can expose hundreds of CPU threads. Limit ORT's pool to avoid
        # oversubscription, without changing any weights, features or calibration.
        original_session=module.ort.InferenceSession
        def bounded_session(*args,**kwargs):
            options=module.ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            kwargs.setdefault('sess_options',options)
            return original_session(*args,**kwargs)
        try:
            module.ort.InferenceSession=bounded_session
            predictor=module.ComputeScore(str(root/'sig_bak_ovr.onnx'),str(root/'model_v8.onnx'))
        finally:module.ort.InferenceSession=original_session
        spec={**spec,'onnxruntime_threads':{'intra_op':4,'inter_op':1}}
        def score(path,row):
            import tempfile
            import librosa
            import soundfile as sf
            # Preserve the author's original librosa/kaiser_best resampler while
            # avoiding its obsolete positional API on modern librosa. No trimming.
            audio,sr=sf.read(path,dtype='float32')
            if sr!=16000:audio=librosa.resample(audio,orig_sr=sr,target_sr=16000,res_type='kaiser_best')
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
                sf.write(tmp.name,audio,16000,subtype='FLOAT')
                data=predictor(tmp.name,16000,False)
            return {'sig':float(data['SIG']),'bak':float(data['BAK']),
                    'ovrl':float(data['OVRL']),'p808':float(data['P808_MOS'])}
    elif action=='utmos22':
        import torch
        sys.path.insert(0,str(root))
        from score import Score
        original=Path.cwd()
        try:
            # Author model resolves wav2vec_small.pt relative to its working directory.
            os.chdir(root);model=Score(ckpt_path=str(root/'epoch=3-step=7459.ckpt'),input_sample_rate=16000,device='cuda')
        finally:os.chdir(original)
        def score(path,row):
            audio=torch.from_numpy(audio16(path)).cuda()
            return {'mos':float(model.score(audio)[0])}
    elif action=='wavlm_sim':
        import torch
        import torch.nn.functional as F
        import types
        # s3prl.upstream.__init__ eagerly imports every unrelated SSL backend,
        # including old torchaudio APIs. Load its unchanged WavLM module through
        # a package namespace so only the required author's code is executed.
        package=importlib.util.find_spec('s3prl')
        upstream_path=Path(next(iter(package.submodule_search_locations)))/'upstream'
        if 's3prl.upstream' not in sys.modules:
            namespace=types.ModuleType('s3prl.upstream')
            namespace.__path__=[str(upstream_path)]
            namespace.__package__='s3prl.upstream'
            namespace.__spec__=importlib.util.spec_from_loader('s3prl.upstream',loader=None,is_package=True)
            sys.modules['s3prl.upstream']=namespace
        from s3prl.upstream.wavlm.expert import UpstreamExpert
        sys.path.insert(0,spec['source_path'])
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL
        original=torch.hub.load
        def pinned_hub(repo,model,*args,**kwargs):
            if repo!='s3prl/s3prl' or model!='wavlm_large':
                raise ValueError('Unexpected dynamic metric code request')
            return UpstreamExpert(ckpt=str(root/'speaker_similarity/wavlm_large/wavlm_large.pt'))
        try:
            torch.hub.load=pinned_hub
            model=ECAPA_TDNN_SMALL(feat_dim=1024,feat_type='wavlm_large')
        finally:torch.hub.load=original
        state=torch.load(root/'speaker_similarity/wavlm_large_finetune.pth',map_location='cpu')
        missing,unexpected=model.load_state_dict(state['model'],strict=False)
        # Some official exports omit the already-loaded frozen SSL frontend. Missing
        # speaker-head weights would produce a plausible but invalid random SIM.
        # Released training checkpoints also contain a class-loss projection;
        # author verification.py ignores it, and it is not part of embedding inference.
        allowed_training_only={'loss_calculator.projection.weight'}
        if any(not k.startswith('feature_extract.') for k in missing) or set(unexpected)-allowed_training_only:
            raise ValueError('Speaker verification checkpoint mismatch: '+str((missing,unexpected)))
        spec={**spec,'missing_pretrained_frontend_keys':list(missing),'unexpected_keys':list(unexpected)}
        del state
        device=cfg.get('quality_control_device','cuda')
        model.eval().to(device);cache={}
        def embedding(path):
            with torch.no_grad():return model(torch.from_numpy(audio16(path)).unsqueeze(0).to(device))
        def score(path,row):
            ref=row['reference_audio'];sha=row['reference_audio_sha256']
            if digest(ref)!=sha:raise ValueError('SIM reference digest changed')
            if sha not in cache:cache[sha]=embedding(ref)
            return {'similarity':float(F.cosine_similarity(embedding(path),cache[sha]).item())}
    else:
        raise ValueError('Unknown metric: '+action)
    return score,spec
