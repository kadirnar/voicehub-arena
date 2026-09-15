"""Guard complete coverage and score consistency in the published static snapshot."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPACE=ROOT/'hf-space'
CORRECTION='corrections/cosyvoice-v3-hift-20260915'


def test_all_published_models_cover_the_same_full_source_texts():
    source_path=ROOT/'datasets/public/seedtts_en/full.jsonl'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='8a9386efb1768a90ffba66f8931e3887b53e9c9d1bf246eb46a8cce32bc7b1c1'
    source={r['id']:r['reference'] for r in map(json.loads,source_path.read_text().splitlines())}
    data=json.loads((SPACE/'data/leaderboard.json').read_text())
    assert len(data['table'])==33 and data['scored_audio']==35904
    for model in data['table']:
        rows=json.loads((SPACE/'data/models'/f"{model['model']}.json").read_text())['rows']
        assert len(rows)==len({r['id'] for r in rows})==1088
        assert {r['id']:r['reference'] for r in rows}==source
        assert all(r['status']=='ok' and r['normalization_id']=='whisper_english' for r in rows)
        prefix=CORRECTION+'/audio/' if model['model']=='cosyvoice' else f"audio/{model['model']}/"
        assert all(r['audio_path'].startswith(prefix) and r['audio_path'].endswith('.wav') for r in rows)
        for metric,prefix in [('wer','word'),('cer','char')]:
            errors=sum(sum(r['metrics'][prefix+'_'+k] for k in ('substitutions','deletions','insertions')) for r in rows)
            count=sum(r['metrics']['reference_'+prefix+'s'] for r in rows)
            assert math.isclose(errors/count,model[metric],rel_tol=1e-12)
        assert math.isclose(sum(r['latency_s'] for r in rows)/sum(r['duration_s'] for r in rows),model['rtf'],rel_tol=1e-12)


def test_csv_and_complete_metrics_report_match_leaderboard():
    data=json.loads((SPACE/'data/leaderboard.json').read_text())
    metrics=json.loads((SPACE/'reports/full-metrics.json').read_text())
    summaries={r['model']:r['summary'] for r in metrics['records']}
    with (SPACE/'data/leaderboard.csv').open() as f:
        records={r['model']:r for r in csv.DictReader(f)}
    assert len(records)==33
    for model in data['table']:
        for key in ('wer','cer','rtf','latency_p50_s','latency_p95_s','peak_vram_mib'):
            assert float(records[model['model']][key])==model[key]==summaries[model['model']][key]
    assert {r['model'] for r in data['table'] if r['quality_review']}=={'dia','llasa'}


def test_every_audio_range_is_unique_and_inside_its_published_archive():
    shards=json.loads((SPACE/'data/audio-shards.json').read_text())
    assert len(shards)==34  # All 33 historical archives plus the versioned correction.
    sizes={r['path']:r['size'] for r in shards}
    seen=set()
    for path in (SPACE/'data/models').glob('*.json'):
        data=json.loads(path.read_text())
        previous_end=0
        for row in data['rows']:
            archive=row['audio_archive'];start=row['audio_offset'];end=start+row['audio_bytes']
            expected=CORRECTION+'/cosyvoice.tar' if data['model']=='cosyvoice' else f"audio_shards/{data['model']}.tar"
            assert archive==expected
            assert start%512==0 and start>=previous_end and end<=sizes[archive]
            assert row['audio_bytes']>44 and len(row['audio_sha256'])==64
            assert (archive,start) not in seen
            seen.add((archive,start));previous_end=end
    assert len(seen)==35904


def test_cosyvoice_full_correction_and_invalidated_snapshot_are_explicit():
    data=json.loads((SPACE/'data/leaderboard.json').read_text())
    row=next(r for r in data['table'] if r['model']=='cosyvoice')
    assert row['upstream_checkpoint']=='FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
    assert row['upstream_revision']=='29e01c4e8d000f4bcd70751be16fa94bf3d85a18'
    assert row['score_status']=='corrected_full_run'
    assert row['wer']==208/11943  # Full corpus edits, never the selected eight-text pilot.
    assert row['cer']==419/67212
    original=json.loads((SPACE/'reports/cosyvoice-correction-2026-09-15/original-leaderboard.json').read_text())
    old=next(r for r in original['table'] if r['model']=='cosyvoice')
    assert old['score_status']=='invalidated_by_implementation_bug'
    assert old['wer']==0.13815624215021352
    assert {r['model']:r for r in original['table'] if r['model']!='cosyvoice'}=={r['model']:r for r in data['table'] if r['model']!='cosyvoice'}
    proof=json.loads((SPACE/'reports/cosyvoice-correction-2026-09-15/verification.json').read_text())
    assert proof['verified_wav_sha256']==proof['verified_archive_ranges']==proof['scored']==1088
    assert row['wer_ci95']==proof['wer_ci95'] and row['cer_ci95']==proof['cer_ci95']
    assert row['correction']['corrected_dataset_revision']==data['dataset_revision']
    assert 'cosyvoice-correction.html' in (SPACE/'index.html').read_text()
