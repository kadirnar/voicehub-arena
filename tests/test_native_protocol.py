import json
from pathlib import Path
import subprocess
import sys

import pytest

from voicehub_arena.native_protocol import digest, verify_rows
from voicehub_arena.native_eval import summarize_native


def test_native_worker_refuses_voicehub_even_if_installed():
    proc=subprocess.run([sys.executable,'-c',
        'from voicehub_arena.native_protocol import forbid_voicehub; forbid_voicehub(); import voicehub'],
        text=True,capture_output=True)
    assert proc.returncode!=0
    assert 'Native campaign forbids VoiceHub inference' in proc.stderr


def test_native_dataset_has_exact_original_targets_and_hashed_paired_references():
    cfg=json.loads(Path('configs/native-methods.json').read_text())
    path=Path(cfg['dataset'])
    assert digest(path)==cfg['dataset_sha256']
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    original=[json.loads(s) for s in Path('datasets/public/seedtts_en/full.jsonl').read_text().splitlines()]
    assert [(r['id'],r['text'],r['reference']) for r in rows]==[(r['id'],r['text'],r['reference']) for r in original]
    assert len(rows)==1088
    assert all(len(r['reference_audio_sha256'])==64 and r['reference_text']==r['prompt_text'] for r in rows)
    assert len({r['reference_audio'] for r in rows})==666


def test_no_audio_penalty_does_not_disappear_from_corpus():
    result={'expected_samples':1,'rows':[{'id':'silent','reference':'Hello world',
        'text':'Hello world','generation_status':'no_audio','scores':{
            'asr':{'status':'ok','transcript':''},
            'dnsmos':{'status':'not_applicable'},'utmos22':{'status':'not_applicable'},
            'wavlm_sim':{'status':'not_applicable'}}}]}
    summarize_native(result,{'uses_reference':True})
    assert result['status']=='completed'
    assert result['summary']['wer']==1
    assert result['summary']['cer']==1
    assert result['summary']['generated']==0
    assert result['summary']['wavlm_sim_similarity']['value'] is None


def test_missing_real_audio_quality_score_never_counts_as_complete():
    row=dict(id='real',reference='Hello world',generation_status='ok',latency_s=1.,duration_s=2.,
             peak_vram_mib=100.,silence_ratio=0.,clipping_ratio=0.,rms_dbfs=-20.,scores={
                 'asr':{'status':'ok','transcript':'Hello world'},
                 'dnsmos':{'status':'not_applicable'},'utmos22':{'status':'not_applicable'}})
    result={'expected_samples':1,'rows':[row]}
    summarize_native(result,{'uses_reference':False})
    assert result['status']=='partial'


def test_resume_rejects_changed_audio(tmp_path):
    audio=tmp_path/'audio';audio.mkdir();path=audio/'sample.wav';path.write_bytes(b'original')
    row=dict(id='sample',text='Hello',reference='Hello',generation_status='ok',
             audio='audio/sample.wav',audio_sha256=digest(path))
    verify_rows([row],[row],tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='audio changed'):verify_rows([row],[row],tmp_path)


def test_only_selected_llasa_checkpoints():
    cfg=json.loads(Path('configs/native-methods.json').read_text())
    assert {s['repo'] for s in cfg['experiments'] if s['family'].startswith('llasa')}=={
        'HKUSTAudio/Llasa-1B','HKUSTAudio/Llasa-3B','HKUSTAudio/Llasa-8B'}
