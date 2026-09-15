"""Validate published pilot identities against the frozen full dataset and edit counts."""
import json
from pathlib import Path
import pytest
from voicehub_arena.metrics import errors

ROOT=Path(__file__).parents[1]

def test_all_official_variants_are_pinned_and_pilots_are_fixed_before_scoring():
    c=json.loads((ROOT/'configs/variant-campaign.json').read_text())
    assert len(c['models'])==11 and len({r['id'] for r in c['models']})==11
    assert sum(r['family']=='Llasa' for r in c['models'])==8
    assert sum(r['family']=='Dia2' for r in c['models'])==2
    assert all(len(r['revision'])==40 for r in c['models'])
    assert c['asr']['checkpoint']=='Systran/faster-whisper-large-v3'
    assert c['asr']['vad_filter'] is False


def test_published_pilot_targets_and_metrics_match_the_same_eight_source_rows():
    c=json.loads((ROOT/'configs/variant-campaign.json').read_text())
    corpus=[json.loads(s) for s in (ROOT/c['dataset']).read_text().splitlines()]
    selected=[corpus[i] for i in c['pilot_indices']]
    paths=list((ROOT/'hf-space/data/variants/pilot').glob('*.json'))
    assert paths
    for path in paths:
        d=json.loads(path.read_text());assert d['phase']=='pilot' and len(d['rows'])==8
        assert [(r['id'],r['reference']) for r in d['rows']]==[(r['id'],r['reference']) for r in selected]
        actual=errors([r['reference'] for r in d['rows']],[r['transcript'] for r in d['rows']],normalization='whisper_english')
        for k,v in actual.items():assert d['summary'][k]==pytest.approx(v,abs=1e-12)
        assert all(len(r['audio_dataset_revision'])==40 for r in d['rows'])
        assert all('/pilot/' in r['audio_archive'] for r in d['rows'] if r['status']=='ok')
        for row in d['rows']:
            if row['status']=='generation_failed_scored':
                assert row['transcript']=='' and 'audio' not in row and 'audio_archive' not in row
                assert row['metrics']['wer']==1.0


def test_recording_count_preserves_legacy_snapshot_and_excludes_no_audio_failures():
    import runpy
    count=runpy.run_path(str(ROOT/'scripts/publish_variant_progress.py'))['count_recordings']
    base=json.loads((ROOT/'hf-space/data/leaderboard.json').read_text())['table']
    assert count(base)==35904
    assert count(base+[{'scored':1088,'generated':1087}])==35904+1087
