"""Additional publisher API adapters; imported only in each model's environment."""
from pathlib import Path
import sys


def load_extra(spec,path):
    import torch
    from .native_adapters import snapshot
    backend=spec['backend'];method=spec['method'];settings=spec.get('settings',{})
    source=Path('.deps/native')/backend
    if backend=='chatterbox':
        from chatterbox.tts import ChatterboxTTS
        model=ChatterboxTTS.from_local(path,device='cuda')
        def generate(row):
            yield model.generate(row['text'],audio_prompt_path=row['reference_audio'] if spec['uses_reference'] else None).squeeze(0),model.sr
    elif backend=='f5tts':
        from f5_tts.api import F5TTS
        model=F5TTS(model='F5TTS_v1_Base',
                    ckpt_file=str(Path(path)/'F5TTS_v1_Base/model_1250000.safetensors'),
                    vocab_file=str(Path(path)/'F5TTS_v1_Base/vocab.txt'),
                    vocoder_local_path=snapshot(spec['vocoder']),device='cuda')
        def generate(row):
            audio,sr,_=model.infer(ref_file=row['reference_audio'],ref_text=row['reference_text'],
                gen_text=row['text'],seed=42,show_info=lambda _:None)
            yield audio,sr
    elif backend=='supertonic':
        sys.path.insert(0,str(source/'py'))
        from helper import load_text_to_speech,load_voice_style
        # The publisher's helper explicitly rejects GPU mode at this revision.
        model=load_text_to_speech(str(Path(path)/'onnx'),use_gpu=False)
        style=load_voice_style([str(Path(path)/'voice_styles/M1.json')])
        def generate(row):
            audio,duration=model(row['text'],lang='en',style=style,total_step=8)
            yield audio[0,:int(duration[0]*model.sample_rate)],model.sample_rate
    elif backend=='dia':
        from dia.model import Dia
        model=Dia.from_pretrained(path,device=torch.device('cuda'))
        def generate(row):
            text='[S1] '+(row['reference_text'].rstrip()+' ' if spec['uses_reference'] else '')+row['text']
            # Publisher generate() removes the prefill tokens before decoding.
            audio=model.generate(text,audio_prompt=row['reference_audio'] if spec['uses_reference'] else None)
            if audio is not None:yield audio,44100
    elif backend=='dia2':
        from dia2 import Dia2,GenerationConfig
        model=Dia2.from_local(config_path=Path(path)/'config.json',weights_path=Path(path)/'model.safetensors',
            tokenizer_id=path,mimi_id=snapshot(spec['codec']),device='cuda',dtype='auto')
        def generate(row):
            if spec['streaming']:raise ValueError('This publisher revision has no streaming output API')
            kwargs={'prefix_speaker_1':row['reference_audio'],'include_prefix':False} if spec['uses_reference'] else {}
            # Native cloning aligns the reference with its own Whisper-large-v3.
            result=model.generate('[S1] '+row['text'],config=GenerationConfig(),**kwargs)
            yield result.waveform,result.sample_rate
    elif backend=='llasa':
        import numpy as np
        import soundfile as sf
        from torchaudio.transforms import Resample
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from xcodec2.modeling_xcodec2 import XCodec2Model
        tokenizer=AutoTokenizer.from_pretrained(path)
        model=AutoModelForCausalLM.from_pretrained(path).eval().cuda()
        codec=XCodec2Model.from_pretrained(snapshot(spec['codec'])).eval().cuda()
        end=tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        def generate(row):
            prefix=[]
            if spec['uses_reference']:
                audio,sr=sf.read(row['reference_audio'],dtype='float32',always_2d=True)
                audio=torch.from_numpy(audio.mean(1)).unsqueeze(0)
                if sr!=16000:audio=Resample(sr,16000)(audio)
                prefix=codec.encode_code(input_waveform=audio.cuda())[0,0].tolist()
            text=(row['reference_text'].rstrip()+' ' if prefix else '')+row['text']
            chat=[dict(role='user',content='Convert the text to speech:<|TEXT_UNDERSTANDING_START|>'+text+'<|TEXT_UNDERSTANDING_END|>'),
                  dict(role='assistant',content='<|SPEECH_GENERATION_START|>'+''.join(f'<|s_{x}|>' for x in prefix))]
            ids=tokenizer.apply_chat_template(chat,tokenize=True,return_tensors='pt',continue_final_message=True).cuda()
            if ids.shape[1]>=2048:raise ValueError('Published reference plus target exceeds native Llasa context')
            tokens=model.generate(ids,max_length=2048,eos_token_id=end,do_sample=True,top_p=1.,temperature=.8)[0,ids.shape[1]:]
            codes=[int(t[4:-2]) for t in tokenizer.convert_ids_to_tokens(tokens.tolist()) if t.startswith('<|s_') and t.endswith('|>')]
            if not codes:return
            wave=codec.decode_code(torch.tensor(prefix+codes,device='cuda').reshape(1,1,-1))[0,0]
            # Codec is 50 Hz; remove the encoded prefix duration, never reference speech in scored WAV.
            yield wave[len(prefix)*320:],16000
    elif backend=='xtts':
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        config=XttsConfig();config.load_json(str(Path(path)/'config.json'))
        model=Xtts.init_from_config(config)
        model.load_checkpoint(config,checkpoint_dir=path,eval=True);model.cuda()
        def generate(row):
            latent,speaker=model.get_conditioning_latents(audio_path=[row['reference_audio']])
            if spec['streaming']:
                for chunk in model.inference_stream(row['text'],'en',latent,speaker):yield chunk,24000
            else:
                yield model.inference(row['text'],'en',latent,speaker)['wav'],24000
    elif backend=='zonos':
        import torchaudio
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        model=Zonos.from_local(str(Path(path)/'config.json'),str(Path(path)/'model.safetensors'),device='cuda')
        def generate(row):
            speaker=None
            if spec['uses_reference']:
                wav,sr=torchaudio.load(row['reference_audio'])
                speaker=model.make_speaker_embedding(wav,sr)
            cond=model.prepare_conditioning(make_cond_dict(text=row['text'],speaker=speaker,language='en-us'))
            codes=model.generate(cond)
            yield model.autoencoder.decode(codes)[0].squeeze(0),model.autoencoder.sampling_rate
    elif backend=='neutts':
        from neutts import NeuTTS,NeuTTS2E
        cls=NeuTTS2E if method=='preset_emotion' else NeuTTS
        model=cls(backbone_repo=path,backbone_device='cuda',codec_repo=snapshot(spec['codec']),codec_device='cuda',seed=42)
        def generate(row):
            if method=='preset_emotion':
                kwargs={'speaker':'emily','emotion':'neutral'}
            else:
                kwargs={'ref_codes':model.encode_reference(row['reference_audio']),'ref_text':row['reference_text']}
            if spec['streaming']:
                for chunk in model.infer_stream(row['text'],**kwargs):yield chunk,model.sample_rate
            else:yield model.infer(row['text'],**kwargs),model.sample_rate
    else:
        raise NotImplementedError('Native adapter pending: '+backend)
    return generate
