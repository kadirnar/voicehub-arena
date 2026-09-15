"""Validate diagnostic records, audio integrity, and paired decoded waveforms."""
import argparse
import hashlib
import json
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from whisper_normalizer.english import EnglishTextNormalizer


def verify(work, archived_runs):
    records = json.loads((work / 'diagnostic-records.json').read_text())
    norm = EnglishTextNormalizer()
    identities = [(r['model'], r['sample'], r['variant']) for r in records]
    assert len(set(identities)) == len(records), 'Duplicate diagnostic identity'
    audio = {}
    for row in records:
        if row['variant'] == 'archived':
            name = row['audio'].split('runs/', 1)[1]
            path = archived_runs / name
        else:
            path = work / Path(row['audio']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['audio_sha256'], path
        wave, sr = sf.read(path, dtype='float32')
        assert wave.size and np.isfinite(wave).all(), path
        ref, hyp = norm(row['reference']), norm(row['rescore']['transcript'])
        assert abs(jiwer.process_words(ref, hyp).wer - row['rescore']['metrics']['wer']) < 1e-12
        assert abs(jiwer.process_characters(ref, hyp).cer - row['rescore']['metrics']['cer']) < 1e-12
        audio[(row['model'], row['sample'], row['variant'])] = (wave, sr)
    pairs = []
    for model in sorted({r['model'] for r in records}):
        comparisons = [('archived', 'baseline')]
        if model == 'bark':
            comparisons.append(('baseline', 'transformers'))
        if model == 'openvoice':
            comparisons.append(('baseline', 'captured_conversion'))
        for a, b in comparisons:
            for i in range(6):
                x, sr = audio[model, i, a]
                y, other_sr = audio[model, i, b]
                assert sr == other_sr
                pair = dict(model=model, sample=i, left=a, right=b,
                            same_shape=x.shape == y.shape)
                if x.shape == y.shape:
                    pair.update(pcm_equal=bool(np.array_equal(x, y)),
                                max_abs_diff=float(np.max(np.abs(x-y))),
                                correlation=float(np.corrcoef(x, y)[0, 1]))
                pairs.append(pair)
    vad = []
    for model in sorted({r['model'] for r in records}):
        group = [r for r in records if r['model'] == model and r['variant'] == 'archived']
        entry = dict(model=model, samples=len(group))
        for key in ['rescore', 'vad_diagnostic']:
            words = jiwer.process_words([norm(r['reference']) for r in group],
                                        [norm(r[key]['transcript']) for r in group])
            entry[key + '_wer'] = words.wer
        vad.append(entry)
    result = dict(records=len(records), scored=len(records),
                  all_audio_hashes_and_metrics_match=True,
                  waveform_pairs=pairs, archived_vad_sensitivity=vad)
    (work / 'diagnostic-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'waveform_pairs'}, indent=2))
    for model in ['bark', 'openvoice']:
        group = [p for p in pairs if p['model'] == model]
        print(model, 'PCM comparisons:', sum(p.get('pcm_equal', False) for p in group), '/', len(group))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--archived-runs', type=Path, required=True)
    args = p.parse_args()
    verify(args.work, args.archived_runs)
