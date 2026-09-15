import hashlib
import json
import pytest
from voicehub_arena.variant_eval import llasa_chat, load_campaign


def test_reference_target_boundary_and_speech_prefix():
    chat=llasa_chat('Target sentence.', 'Last reference word', [0,65535])
    assert 'Last reference word Target sentence.' in chat[0]['content']
    assert chat[1]['content']=='<|SPEECH_GENERATION_START|><|s_0|><|s_65535|>'
    assert 'START|>Target sentence.' in llasa_chat('Target sentence.','',[])[0]['content']


def test_changed_frozen_inputs_are_rejected(tmp_path):
    data=tmp_path/'data.jsonl';ref=tmp_path/'ref.wav';manifest=tmp_path/'config.json'
    data.write_text('{"id":"one"}\n');ref.write_bytes(b'reference')
    cfg={'dataset':str(data),'dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'reference':{'path':str(ref),'sha256':hashlib.sha256(ref.read_bytes()).hexdigest()},'expected_samples':1}
    manifest.write_text(json.dumps(cfg));assert len(load_campaign(manifest)[1])==1
    ref.write_bytes(b'changed')
    with pytest.raises(ValueError,match='reference changed'):load_campaign(manifest)


def test_downloaded_checkpoint_must_match_the_pinned_hub_digest(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    from voicehub_arena.variant_eval import verify_snapshot_files
    weight=tmp_path/'model.safetensors';weight.write_bytes(b'verified model weights')
    sha=hashlib.sha256(weight.read_bytes()).hexdigest()
    expected=SimpleNamespace(path=weight.name,size=weight.stat().st_size,lfs=SimpleNamespace(sha256=sha))
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda:SimpleNamespace(get_paths_info=lambda *a,**kw:[expected]))
    assert verify_snapshot_files(tmp_path,'publisher/model','a'*40)=={weight.name:sha}
    weight.write_bytes(b'corruptd model weights')
    with pytest.raises(ValueError,match='mismatch'):verify_snapshot_files(tmp_path,'publisher/model','a'*40)


def test_no_audio_failure_stays_in_corpus_denominator_without_fake_audio():
    from voicehub_arena.variant_eval import summarize_variant, NO_AUDIO_POLICY
    good=dict(id='good',status='ok',category='read',reference='one two',transcript='one two',normalization_id='orthographic',latency_s=1.,duration_s=2.,peak_vram_mib=10.,clipping_ratio=0.,silence_ratio=0.,rms_dbfs=-20.)
    failure=dict(id='empty',status='generation_failed_scored',category='read',reference='three four',transcript='',normalization_id='orthographic',failure_policy=NO_AUDIO_POLICY,failure_kind='no_speech_tokens')
    result=summarize_variant([good,failure])
    assert result['wer']==.5 and result['reference_words']==4
    assert result['attempted']==result['scored']==2
    assert result['generated']==result['asr_scored']==1
    assert result['generation_failures']==1 and result['generation_failure_rate']==.5
    assert result['rtf']==.5 and result['audio_seconds']==2.
    assert result['successful_audio_only']['wer']==0


def test_failure_policy_rejects_arbitrary_exceptions_or_fabricated_recordings():
    from voicehub_arena.variant_eval import validate_scored_row,NO_AUDIO_POLICY
    valid=dict(status='generation_failed_scored',transcript='',failure_policy=NO_AUDIO_POLICY,failure_kind='no_speech_tokens')
    validate_scored_row(valid)
    for override in [{'audio':'fake.wav'},{'transcript':'invented speech'},{'failure_policy':'retry-best'},{'failure_kind':'CUDA OOM'},{'status':'failed'}]:
        with pytest.raises(ValueError):validate_scored_row({**valid,**override})


def test_active_scope_keeps_only_requested_llasa_bases_and_blocks_removed_models():
    from voicehub_arena.variant_scope import load_selection,require_active
    root=__import__('pathlib').Path(__file__).parents[1]
    manifest=root/'configs/variant-campaign.json';cfg=json.loads(manifest.read_text())
    selection,models=load_selection(cfg,manifest)
    assert [m['repo'] for m in models if m['family']=='Llasa']==['HKUSTAudio/Llasa-1B','HKUSTAudio/Llasa-3B','HKUSTAudio/Llasa-8B']
    assert len(models)==6 and selection['excluded_base_model_ids']==['llasa']
    for m in cfg['models']:
        if m['id'] in selection['active_model_ids']:require_active(m['id'],cfg,manifest)
        else:
            with pytest.raises(ValueError,match='removed from active scope'):require_active(m['id'],cfg,manifest)
