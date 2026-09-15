"""The generation duration limit must use Mimi's decoded frame rate."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
voicehub=pytest.importorskip('voicehub')


def generator_method():
    # Exercise the shipped loop without loading a 1.5B checkpoint. The dummy
    # decoder below has the observed Mimi stride: 1920 samples at 24 kHz.
    path=Path(voicehub.__file__).parent/'models/conversationtts/source/conversationtts/inference/generator.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Generator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate_v1')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    ast.fix_missing_locations(module)
    scope={'torch':torch}
    exec(compile(module,str(path),'exec'),scope)
    return scope['generate_v1']


@pytest.mark.parametrize('maximum_ms,expected_frames',[(80,1),(160,2),(1000,12)])
def test_nonterminating_model_cannot_exceed_requested_duration(maximum_ms,expected_frames):
    class NeverEnds:
        def reset_caches(self): pass
        def generate_frame(self,*args): return torch.ones(1,32,dtype=torch.long)
    codec=SimpleNamespace(model=SimpleNamespace(target_frame_rate=12.5),
        detokenize=lambda codes:torch.zeros(1,codes.shape[-1]*1920))
    generator=SimpleNamespace(_model=NeverEnds(),_audio_tokenizer=codec,device=torch.device('cpu'),
        _tokenize_text_segment=lambda text:(torch.ones(2,33,dtype=torch.long),torch.ones(2,33,dtype=torch.bool)))
    audio=generator_method()(generator,'Test duration limit.',max_audio_length_ms=maximum_ms)
    assert audio.numel()/24000*1000 <= maximum_ms
    assert audio.numel()==expected_frames*1920
