"""Compute the real CAMPPlus embedding from the official example reference.

Uses the feature recipe in CosyVoice's upstream cli/frontend.py, executed
explicitly on CPU. No random/zero embedding is used to bypass the model contract.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import HfApi, hf_hub_download

root=Path(__file__).resolve().parents[1]
reference=root/'datasets/reference/emily.wav'
repo='FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
revision=HfApi().model_info(repo).sha
path=hf_hub_download(repo,'campplus.onnx',revision=revision)
audio,sr=sf.read(reference,always_2d=True,dtype='float32')
torch.set_num_threads(2)
speech=torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
speech=torchaudio.functional.resample(speech,sr,16000)
features=torchaudio.compliance.kaldi.fbank(speech,num_mel_bins=80,dither=0,sample_frequency=16000)
features=features-features.mean(dim=0,keepdim=True)
options=ort.SessionOptions()
options.intra_op_num_threads=2
session=ort.InferenceSession(path,sess_options=options,providers=['CPUExecutionProvider'])
embedding=session.run(None,{session.get_inputs()[0].name:features.unsqueeze(0).numpy()})[0].flatten()
assert embedding.shape==(192,) and np.isfinite(embedding).all()
output=root/'datasets/reference/cosyvoice_embedding.json'
output.write_text(json.dumps(embedding.tolist(),indent=2))
provenance={'encoder':repo,'revision':revision,'filename':'campplus.onnx',
            'encoder_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'feature_recipe':'16 kHz mono; Kaldi fbank 80 bins, dither 0; subtract temporal mean',
            'upstream':'https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/frontend.py'}
output.with_name('cosyvoice_embedding_provenance.json').write_text(json.dumps(provenance,indent=2))
config=root/'configs/models.json'
models=json.loads(config.read_text())
models.setdefault('cosyvoice', {})['generation']={'speaker_embedding':embedding.tolist(),'instruction':'Speak clearly.','flow_steps':10}
config.write_text(json.dumps(models,indent=2))
print('Prepared a 192-dimensional CAMPPlus embedding from the official Emily reference.',flush=True)
