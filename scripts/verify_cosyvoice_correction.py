"""Verify and package the complete corrected CosyVoice split; preserve originals."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from whisper_normalizer.english import EnglishTextNormalizer

VERSION = 'cosyvoice-v3-hift-20260915'
PREFIX = 'corrections/' + VERSION


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    assert not re.search(r'(?:hf_|gh[pousr]_)[A-Za-z0-9]{15,}', raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw)


def verify(run, original_runs, source, output):
    result_path = run/'cosyvoice/result.json'
    result = json.loads(result_path.read_text())
    cfg = json.loads((run/'config.json').read_text())
    old_run = original_runs/'pub-v2-cosyvoice-seedtts_en-000'
    old_cfg = json.loads((old_run/'config.json').read_text())
    old_result = json.loads((old_run/'cosyvoice/result.json').read_text())
    for key in ['seed', 'repeats', 'warmups', 'asr', 'normalization_id', 'input_text_transform']:
        assert cfg[key] == old_cfg[key], key
    assert cfg['overrides']['cosyvoice'] == old_cfg['overrides']['cosyvoice']
    for key in ['generation', 'artifact_provenance', 'effective_config', 'asr', 'frontend_protocol']:
        assert result[key] == old_result[key], key
    expected = {r['id']: r['reference'] for r in map(json.loads, source.read_text().splitlines())}
    rows = result['rows']
    assert result['status'] == 'completed'
    assert len(rows) == len({r['id'] for r in rows}) == len(expected) == 1088
    assert {r['id']: r['reference'] for r in rows} == expected
    assert len(list((run/'cosyvoice').glob('*.wav'))) == 1088
    norm = EnglishTextNormalizer()
    refs, hyps, word_counts, char_counts, manifest = [], [], [], [], []
    saturated = samples = 0
    for row in rows:
        assert row['status'] == 'ok' and row['normalization_id'] == 'whisper_english'
        assert row['synthesis_text'] == row['reference'] and row['seed'] == 42 and row['repeat'] == 0
        assert row.get('generation_token_limit') is not None
        assert 'generation_limit_reached' not in row.get('quality_flags', [])
        path = (run/row['audio']).resolve()
        assert path.is_relative_to(run.resolve()) and sha(path) == row['audio_sha256']
        audio, sr = sf.read(path, dtype='float32')
        assert sr == row['sample_rate'] == 24000 and audio.ndim == 1 and np.isfinite(audio).all()
        assert abs(len(audio)/sr-row['duration_s']) < 1e-12
        saturated += int(np.sum(np.abs(audio) >= .99-1e-7)); samples += audio.size
        ref, hyp = norm(row['reference']), norm(row['transcript'])
        w, c = jiwer.process_words(ref, hyp), jiwer.process_characters(ref, hyp)
        assert abs(w.wer-row['metrics']['wer']) < 1e-12
        assert abs(c.cer-row['metrics']['cer']) < 1e-12
        refs.append(ref); hyps.append(hyp)
        word_counts.append([w.substitutions+w.deletions+w.insertions, w.hits+w.substitutions+w.deletions])
        char_counts.append([c.substitutions+c.deletions+c.insertions, c.hits+c.substitutions+c.deletions])
    words, chars = jiwer.process_words(refs, hyps), jiwer.process_characters(refs, hyps)
    summary = result['summary']
    assert abs(words.wer-summary['wer']) < 1e-12 and abs(chars.cer-summary['cer']) < 1e-12
    assert summary['scored'] == summary['generated'] == summary['unique_prompts_scored'] == 1088
    # Independently bootstrap corpus edit/count ratios with the recorded protocol.
    rng = np.random.default_rng(42)
    indices = rng.integers(0, 1088, size=(1000, 1088))
    ci = {}
    for metric, counts in [('wer', word_counts), ('cer', char_counts)]:
        counts = np.asarray(counts)
        totals = counts[indices].sum(axis=1)
        ci[metric+'_ci95'] = np.percentile(totals[:, 0]/totals[:, 1], [2.5, 97.5]).tolist()
        assert np.allclose(ci[metric+'_ci95'], summary[metric+'_ci95'], rtol=0, atol=1e-12)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/'cosyvoice.tar'
    assert not archive.exists(), 'Use a fresh output directory; do not replace a published archive'
    with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT) as tar:
        for row in rows:
            path = run/row['audio']
            info = tarfile.TarInfo(path.name); info.size = path.stat().st_size; info.mode = 0o644
            offset = tar.offset + len(info.tobuf(format=tarfile.PAX_FORMAT))
            with path.open('rb') as stream:
                tar.addfile(info, stream)
            row.update(source_run=run.name, audio_path=PREFIX+'/audio/'+path.name,
                       audio_archive=PREFIX+'/cosyvoice.tar', audio_offset=offset, audio_bytes=info.size,
                       implementation_version=VERSION)
            manifest.append(dict(path=row['audio_path'], archive=row['audio_archive'], entry=path.name,
                                 offset=offset, size=info.size, sha256=row['audio_sha256']))
            raw = json.dumps(row, ensure_ascii=False).encode()
            meta = tarfile.TarInfo(path.stem+'.json'); meta.size = len(raw); meta.mode = 0o644
            tar.addfile(meta, io.BytesIO(raw))
    with archive.open('rb') as stream:
        for row in rows:
            stream.seek(row['audio_offset'])
            assert hashlib.sha256(stream.read(row['audio_bytes'])).hexdigest() == row['audio_sha256']
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers()) == 2176
    proof = dict(verified_at=datetime.now(timezone.utc).isoformat(), version=VERSION,
                 generated=1088, scored=1088, unique_source_texts=1088, verified_wav_sha256=1088,
                 verified_archive_ranges=1088, per_sample_metrics_match=True,
                 generation_limit_samples=0, configuration_matches_original=True,
                 dataset_sha256=sha(source), original_result_sha256=sha(old_run/'cosyvoice/result.json'),
                 corrected_result_sha256=sha(result_path), corrected_config_sha256=sha(run/'config.json'),
                 vocoder_source_sha256='212f82751d7bedfd32349fb8aa3393489b6e2e70881677d291aefd2b55f3ed21',
                 wer=words.wer, cer=chars.cer, **ci,
                 word_substitutions=words.substitutions, word_deletions=words.deletions,
                 word_insertions=words.insertions, reference_words=11943,
                 saturation_threshold=.99-1e-7, saturated_waveform_samples=saturated,
                 waveform_samples=samples, saturation_fraction=saturated/samples,
                 archive=dict(path=PREFIX+'/cosyvoice.tar', sha256=sha(archive), size=archive.stat().st_size, samples=1088))
    write(output/'verification.json', proof)
    write(output/'model-records.json', {'model':'cosyvoice', 'summary':summary, 'rows':rows})
    write(output/'audio-manifest.json', manifest)
    # Keep complete reproducibility data while removing machine-local absolute roots.
    for name, value in [('result.json', result), ('config.json', cfg)]:
        raw = json.dumps(value, ensure_ascii=False).replace('/workspace/voicehub-arena/', '')
        write(output/name, json.loads(raw))
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ['run', 'original-runs', 'source', 'output']:
        parser.add_argument('--'+arg, type=Path, required=True)
    a = parser.parse_args()
    verify(a.run, a.original_runs, a.source, a.output)
