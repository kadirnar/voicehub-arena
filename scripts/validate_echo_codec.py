"""Compare the real Echo codec with its digest-pinned reference on CPU.

The reference is used only by this explicit validation, never by Arena inference.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request

import torch
from voicehub.hub import resolve_pretrained_file
from voicehub.models.echo.autoencoder import build_ae
from voicehub.models.echo.sampling import _assign_validated_state, _load_safetensors, _canonicalize_codec_weight_norm
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse, resolved_artifacts

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = '2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e'
SOURCE_SHA256 = '248c161d8e140c557d0d6e7bd4eb8325bdda99b0c44a734aabb1825353f842c7'
CODEC_REVISION = '18b770e5b62b1a58ebd7423401a937cc063d8729'
reference_path = ROOT/'artifacts/reference/echo_autoencoder.py'
reference_path.parent.mkdir(parents=True, exist_ok=True)
if not reference_path.exists():
    url = f'https://raw.githubusercontent.com/jordandare/echo-tts/{SOURCE_REVISION}/autoencoder.py'
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Echo reference download differs from the reviewed source')
    reference_path.write_bytes(data)
if hashlib.sha256(reference_path.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise ValueError('Echo reference source has changed')
spec = importlib.util.spec_from_file_location('echo_audited_reference', reference_path)
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)

configure_hub_auth()
enable_verified_cache_reuse()
torch.set_num_threads(2)
checkpoint = resolve_pretrained_file('jordand/fish-s1-dac-min', 'pytorch_model.safetensors',
                                    revision=CODEC_REVISION)
with torch.device('meta'):
    native = build_ae()
    official = reference.build_ae()
state = _load_safetensors(checkpoint, device='cpu', dtype=torch.float32)
_assign_validated_state(native, _canonicalize_codec_weight_norm(native, state))
official.load_state_dict(state, strict=True, assign=True)
native.eval()
official.eval()
masks = [tensor for name, tensor in native.named_buffers() if name.endswith('causal_mask')]
if not masks or any(tensor.dtype != torch.bool for tensor in masks):
    raise ValueError('Echo codec causal masks lost their boolean type')
with torch.inference_mode():
    torch.manual_seed(53)
    latents = torch.randn(1, 1024, 4)
    expected_audio = official.decode_zq(latents.clone())
    actual_audio = native.decode_zq(latents.clone())
    torch.testing.assert_close(actual_audio, expected_audio, rtol=1e-4, atol=1e-5)
    wave = (0.1*torch.sin(torch.arange(8192).float()*2*torch.pi*220/44100))[None, None]
    expected_latents = official.encode_zq(wave.clone())
    actual_latents = native.encode_zq(wave.clone())
    torch.testing.assert_close(actual_latents, expected_latents, rtol=1e-4, atol=1e-5)
record = {'source_revision': SOURCE_REVISION, 'source_sha256': SOURCE_SHA256,
          'codec_revision': CODEC_REVISION, 'tensor_count': len(state),
          'decoder_max_absolute_error': float((actual_audio-expected_audio).abs().max()),
          'encoder_max_absolute_error': float((actual_latents-expected_latents).abs().max()),
          'mask_dtype': 'bool', 'device': 'cpu', 'compute_dtype': 'float32',
          'scope': 'Real codec weights, two short encode/decode probes; TTS quality measured separately',
          'artifacts': resolved_artifacts()}
output = ROOT/'runs/validations/echo-codec-cpu.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2))
print('Echo actual-codec reference comparison passed:', json.dumps(record), flush=True)
