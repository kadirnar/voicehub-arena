"""Compare native HiFT operations against the pinned vendored publisher methods."""
import ast
import importlib.util
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
import voicehub
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceHiFTConfig
from voicehub.architectures.cosyvoice_native.vocoder import CosyVoiceHiFTGenerator


def generator():
    target = os.environ.get('ARENA_COSY_VOCODER_CANDIDATE')
    cls = CosyVoiceHiFTGenerator
    if target:
        spec = importlib.util.spec_from_file_location('cosy_vocoder_candidate', target)
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.CosyVoiceHiFTGenerator
    torch.manual_seed(12)
    return cls(CosyVoiceHiFTConfig(base_channels=16, f0_hidden_size=16)).eval()


def publisher_method(class_name, method):
    path = Path(voicehub.__file__).parent/'models/cosyvoice/source/cosyvoice/hifigan/generator.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    scope = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method]


def reference(native):
    cfg = native.config
    obj = SimpleNamespace(**{k: getattr(native, k) for k in
                             ('conv_pre','ups','source_downs','source_resblocks','resblocks','conv_post','stft_window')})
    obj.istft_params = {'n_fft': cfg.istft_n_fft, 'hop_len': cfg.istft_hop_length}
    obj.audio_limit = cfg.audio_limit
    obj.lrelu_slope = .1
    obj.num_upsamples = len(native.ups)
    obj.num_kernels = len(cfg.resblock_kernel_sizes)
    obj.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
    obj._stft = MethodType(publisher_method('HiFTGenerator','_stft'), obj)
    obj._istft = MethodType(publisher_method('HiFTGenerator','_istft'), obj)
    obj.decode = MethodType(publisher_method('CausalHiFTGenerator','decode'), obj)
    return obj


def test_istft_matches_publisher_with_large_log_magnitudes():
    native = generator(); official = reference(native)
    torch.manual_seed(15)
    values = torch.randn(2, native.config.istft_n_fft+2, 37)*3
    values[:, 1:4] += 8  # Published clamp applies after exp(), including this regime.
    bins = native.config.istft_n_fft//2+1
    with torch.no_grad():
        expected = official._istft(values[:, :bins].exp(), values[:, bins:].sin())
        actual = native._istft(values)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('frames', [8, 19])
def test_complete_decode_matches_publisher_on_same_features_and_source(frames):
    native = generator(); official = reference(native)
    torch.manual_seed(18)
    mel = torch.randn(1, 80, frames)
    original_stft = native._stft; captured = {}
    def capture(source):
        captured['source'] = source.detach().clone()
        return original_stft(source)
    native._stft = capture
    with torch.no_grad():
        actual, _ = native(mel)
        expected = official.decode(mel, captured['source'].unsqueeze(1), finalize=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
