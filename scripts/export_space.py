"""Export a verified full campaign for a static HF Space and an audio dataset.

Never imports GPU libraries or changes original runs. Audio is hard-linked when
possible, and verified against the synthesis SHA256 before inclusion.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

SECRET = re.compile(r'(?:hf_|gh[pousr]_)[A-Za-z0-9]{15,}')


def write_json(path, data):
    raw = json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n'
    if SECRET.search(raw):
        raise ValueError('Credential-like value found; export stopped')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)


def export(report_dir, runs, space_dir, dataset_dir, dataset_id):
    full = json.loads((report_dir/'full-results.json').read_text())
    metrics = json.loads((report_dir/'full-metrics.json').read_text())
    audit = json.loads((report_dir/'backup-verification.json').read_text())
    assert metrics['models'] == len(full['results']) == 33
    assert metrics['scored_audio'] == audit['sha256_verified_audio'] == 35904
    assert audit['completed_jobs'] == 198 and audit['campaign_status_at_sync'] == 'completed'
    summaries = {r['model']: r for r in metrics['records']}
    source = {r['id']: r['reference'] for r in full['config']['dataset']}
    assert len(source) == 1088
    table, manifest = [], []
    scalar_metrics = ('wer','cer','mer','wil','wip','exact_match_rate','utterance_mean_wer',
                      'utterance_mean_cer','rtf','latency_p50_s','latency_p95_s','peak_vram_mib',
                      'audio_seconds','silence_ratio','clipping_ratio','rms_dbfs')
    for result in full['results']:
        model = result['model_type']; summary = result['summary']; rows = result['rows']
        assert re.fullmatch(r'[a-z0-9]+', model)
        assert result['status'] == 'completed' and result['ranking_eligible']
        assert len(rows) == summary['scored'] == summary['unique_prompts_scored'] == 1088
        assert len({r['id'] for r in rows}) == 1088
        assert {r['id']:r['reference'] for r in rows} == source
        assert summary['normalization_id'] == 'whisper_english'
        record = {'model':model,'name':result['name'],'checkpoint':result['checkpoint'],
                  'revision':summaries[model]['revision'],'scored':1088,
                  'quality_review':summaries[model]['quality_review'],
                  'empty_asr_transcripts':summaries[model]['empty_asr_transcripts'],
                  'wer_ci95':summary['wer_ci95'],'cer_ci95':summary['cer_ci95'],
                  **{key:summary.get(key) for key in scalar_metrics}}
        table.append(record)
        dataset_audio = dataset_dir/'audio'/model
        dataset_audio.mkdir(parents=True, exist_ok=True)
        metadata = []
        public_rows = []
        for row in rows:
            assert row['status'] == 'ok' and row['normalization_id'] == 'whisper_english'
            assert row['reference'] == row['synthesis_text']
            source_audio = (runs/row['source_run']/row['audio']).resolve()
            assert source_audio.is_relative_to(runs.resolve()) and source_audio.is_file()
            digest = hashlib.sha256(source_audio.read_bytes()).hexdigest()
            assert digest == row['audio_sha256']
            name = source_audio.name
            target = dataset_audio/name
            if target.exists():
                assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
            else:
                try:
                    os.link(source_audio, target)
                except OSError:
                    shutil.copy2(source_audio, target)
            audio_path = f'audio/{model}/{name}'
            manifest.append({'path':audio_path,'sha256':digest,'size':source_audio.stat().st_size})
            public_rows.append({**row,'audio_path':audio_path})
            metadata.append({'file_name':name,'model':model,'id':row['id'],
                             'reference':row['reference'],'transcript':row['transcript'],
                             'checkpoint':result['checkpoint'],'seed':row['seed'],
                             'normalization':'whisper_english','audio_sha256':digest,
                             **{k:row.get(k) for k in ('duration_s','sample_rate','latency_s','rtf',
                                'peak_vram_mib','silence_ratio','clipping_ratio')},
                             **{k:row['metrics'].get(k) for k in ('wer','cer','mer','wil','wip')}})
        (dataset_audio/'metadata.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metadata))
        write_json(space_dir/'data/models'/f'{model}.json',{'model':model,'rows':public_rows})
        write_json(dataset_dir/'records'/f'{model}.json',{'model':model,'summary':summary,'rows':public_rows})
        print(f'Verified and exported {model}: {len(rows)} audio files',flush=True)
    assert len(manifest) == 35904 and len({r['path'] for r in manifest}) == 35904
    table.sort(key=lambda r:r['wer'])
    release = {k:v for k,v in metrics.items() if k != 'records'}
    release.update(dataset_id=dataset_id,dataset_revision='main',table=table,
                   audio_bytes=sum(r['size'] for r in manifest),
                   source_url=full['config']['dataset_manifest']['provenance']['source_url'])
    write_json(space_dir/'data/leaderboard.json', release)
    write_json(dataset_dir/'leaderboard.json', release)
    write_json(dataset_dir/'audio-manifest.json', manifest)
    with (space_dir/'data/leaderboard.csv').open('w',newline='') as f:
        fields=['model','name','checkpoint','revision','scored']+list(scalar_metrics)+['quality_review','empty_asr_transcripts']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    (space_dir/'reports').mkdir(parents=True,exist_ok=True)
    (dataset_dir/'reports').mkdir(parents=True,exist_ok=True)
    for name in ('full-metrics.json','full-results.json','backup-verification.json','completion-verification.json',
                 'wer-cer-full.png','wer-cer-full.svg','speed-memory-full.png','speed-memory-full.svg'):
        source_path=report_dir/name
        raw=source_path.read_bytes()
        if source_path.suffix == '.json' and SECRET.search(raw.decode()):
            raise ValueError('Credential-like value found; export stopped')
        shutil.copy2(source_path,dataset_dir/'reports'/name)
        if name != 'full-results.json':
            shutil.copy2(source_path,space_dir/'reports'/name)
    shutil.copy2(space_dir/'data/leaderboard.csv',dataset_dir/'leaderboard.csv')
    print(json.dumps({'models':33,'rows':35904,'audio_bytes':release['audio_bytes'],
                      'dataset_dir':str(dataset_dir),'space_dir':str(space_dir)}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--space-dir',type=Path,default=Path('hf-space'))
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--dataset-id',default='kadirnar/voicehub-arena-seed-tts-eval')
    a=p.parse_args();export(a.report_dir,a.runs,a.space_dir,a.dataset_dir,a.dataset_id)
