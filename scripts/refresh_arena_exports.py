"""Refresh every current plot from the verified comparison, preserving historical exports."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone


def refresh(site=Path('hf-space')):
    comparison = site / 'data/current-comparison.json'
    data = json.loads(comparison.read_text())
    progress = json.loads((site / 'data/variant-progress.json').read_text())
    selection = json.loads((site / 'data/variant-selection.json').read_text())
    base = json.loads((site / 'data/leaderboard.json').read_text())
    baseline = {r['model']: r for r in base['table'] if r['model'] not in selection['excluded_base_model_ids']}
    complete = {m['id']: m for m in progress['models'] if m['id'] in selection['active_model_ids'] and m['full']['published'] and m['full']['status'] == 'completed' and m['full']['scored'] == m['full']['expected'] == 1088}
    actual = {r['model']: r for r in data['table']}
    assert len(actual) == len(data['table'])
    assert set(actual) == set(baseline) | set(complete)
    for mid, row in actual.items():
        expected = baseline[mid] if mid in baseline else complete[mid]['full']['metrics']
        assert row['scored'] == 1088
        for key in ('wer', 'cer', 'rtf', 'peak_vram_mib', 'wer_ci95', 'cer_ci95'):
            assert row[key] == expected[key], (mid, key)
    out = site / 'reports/current'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'leaderboard.json').write_bytes(comparison.read_bytes())
    records = [{'model': r['model'], 'name': r['name'], 'quality_review': bool(r.get('quality_review')), 'summary': r} for r in data['table']]
    (out / 'full-metrics.json').write_text(json.dumps({'records': records}, indent=2) + '\n')
    subprocess.run([sys.executable, 'scripts/render_bar_charts.py', '--source', str(comparison), '--output', str(out / 'bars')], check=True)
    subprocess.run([sys.executable, 'scripts/render_full_plots.py', '--source', str(out / 'full-metrics.json'), '--output', str(out)], check=True)
    manifest = {
        'source_sha256': hashlib.sha256(comparison.read_bytes()).hexdigest(),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'progress_updated_at': progress['updated_at'],
        'configurations': len(actual),
        'model_ids': list(actual),
        'completed_experiments': list(complete),
        'expected_experiments': len(selection['active_model_ids']),
        'evaluated_targets': sum(r['scored'] for r in actual.values()),
        'real_recordings': sum(r.get('generated', r['scored']) for r in actual.values()),
        'no_audio_failures': sum(r.get('generation_failures', 0) for r in actual.values()),
        'samples_per_configuration': 1088,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    refresh()
