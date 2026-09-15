"""Recompute all published scores and characterize errors without changing results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import jiwer
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer


def audit(source, output):
    leaderboard = json.loads((source / 'leaderboard.json').read_text())
    normalizer = EnglishTextNormalizer()
    expected = None
    reports = []
    selections = {}
    for summary in sorted(leaderboard['table'], key=lambda r: -r['wer']):
        model = summary['model']
        rows = json.loads((source / 'models' / f'{model}.json').read_text())['rows']
        identities = {r['id']: r['reference'] for r in rows}
        if expected is None:
            expected = identities
        assert len(rows) == len(identities) == 1088 and identities == expected, model
        assert all(r['status'] == 'ok' and r['synthesis_text'] == r['reference'] for r in rows), model
        refs = [normalizer(r['reference']) for r in rows]
        hyps = [normalizer(r['transcript']) for r in rows]
        words = jiwer.process_words(refs, hyps)
        chars = jiwer.process_characters(refs, hyps)
        assert abs(words.wer-summary['wer']) < 1e-12 and abs(chars.cer-summary['cer']) < 1e-12, model
        counts = []
        substitutions = Counter()
        deletions = Counter()
        insertions = Counter()
        for row, ref, hyp in zip(rows, refs, hyps):
            w = jiwer.process_words(ref, hyp)
            c = jiwer.process_characters(ref, hyp)
            assert abs(w.wer-row['metrics']['wer']) < 1e-12 and abs(c.cer-row['metrics']['cer']) < 1e-12, (model, row['id'])
            counts.append((w.substitutions+w.deletions+w.insertions, w.hits+w.substitutions+w.deletions))
            for chunk in w.alignments[0]:
                left = ' '.join(w.references[0][chunk.ref_start_idx:chunk.ref_end_idx])
                right = ' '.join(w.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx])
                if chunk.type == 'substitute': substitutions[(left, right)] += 1
                elif chunk.type == 'delete': deletions[left] += 1
                elif chunk.type == 'insert': insertions[right] += 1
        counts = np.asarray(counts)
        ordered = np.argsort(-counts[:, 0], kind='stable')
        report = dict(model=model, checkpoint=summary['checkpoint'], samples=len(rows),
            unique_ids_and_reference_texts_match=True, all_per_sample_metrics_match=True,
            wer=words.wer, cer=chars.cer, corpus_metrics_match=True,
            word_substitutions=words.substitutions, word_deletions=words.deletions,
            word_insertions=words.insertions, reference_words=int(counts[:,1].sum()),
            empty_transcripts=sum(not x for x in hyps), wer_over_100_percent=sum(r['metrics']['wer']>1 for r in rows),
            exact_transcripts=sum(a==b for a,b in zip(refs,hyps)),
            duplicate_audio_hashes=len(rows)-len({r['audio_sha256'] for r in rows}),
            duration_percentiles_s=np.percentile([r['duration_s'] for r in rows],[0,25,50,75,95,100]).tolist(),
            high_clipping_recordings=sum(r['clipping_ratio']>.01 for r in rows),
            quiet_recordings_below_minus_50_dbfs=sum(r['rms_dbfs'] < -50 for r in rows),
            top20_fraction_of_word_errors=float(counts[ordered[:20],0].sum()/max(1,counts[:,0].sum())),
            length_buckets=[],
            common_substitutions=[{'reference':a,'transcript':b,'count':n} for (a,b),n in substitutions.most_common(12)],
            common_deletions=deletions.most_common(8), common_insertions=insertions.most_common(8),
            worst_examples=[{k:rows[int(i)][k] for k in ('id','reference','transcript','duration_s','metrics')} for i in ordered[:8]])
        for lo, hi in [(0,40),(40,80),(80,120),(120,10000)]:
            indices=[i for i,r in enumerate(rows) if lo<=len(r['reference'])<hi]
            if indices:
                ix=np.asarray(indices)
                report['length_buckets'].append(dict(min_chars=lo,max_chars_exclusive=hi,samples=len(ix),
                    wer=float(counts[ix,0].sum()/max(1,counts[ix,1].sum())),
                    empty_transcripts=sum(not hyps[i] for i in indices)))
        if model in ('vui','conversationtts','bark','vits','openvoice','voxcpm'):
            selected=list(dict.fromkeys([int(i) for i in ordered[:3]] + [0, 1] + [max(range(len(rows)),key=lambda i:len(rows[i]['reference']))]))
            selections[model]=[rows[i] for i in selected]
        reports.append(report)
        print(f"{model:18} WER={words.wer*100:6.2f}% S/D/I={words.substitutions}/{words.deletions}/{words.insertions} empty={report['empty_transcripts']} top20={report['top20_fraction_of_word_errors']:.1%}",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    result={'leaderboard_sha256':hashlib.sha256((source/'leaderboard.json').read_bytes()).hexdigest(),
            'scope':'All 33 models, all 1088 original records each; scores are unchanged.',
            'normalization':'whisper_english', 'models':reports}
    (output/'all-model-record-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'diagnostic-selection.json').write_text(json.dumps(selections,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('hf-space/data'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.source,args.output)
