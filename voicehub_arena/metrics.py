"""English ASR error rates and explicitly defined signal diagnostics."""
import re
import unicodedata
import numpy as np
import jiwer
from functools import lru_cache


@lru_cache(maxsize=1)
def english_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer
    return EnglishTextNormalizer()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = "".join(c if not unicodedata.category(c).startswith("P") else ("" if c == "'" else " ") for c in text)
    return re.sub(r"\s+", " ", text).strip()


def errors(references, hypotheses, normalized=True, normalization='orthographic'):
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need equal, non-empty reference/hypothesis lists")
    if normalization not in ('orthographic', 'whisper_english'):
        raise ValueError('Unknown normalization: ' + normalization)
    clean = english_normalizer() if normalization == 'whisper_english' else normalize
    refs = [clean(t) for t in references] if normalized else references
    hyps = [clean(t) for t in hypotheses] if normalized else hypotheses
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    return dict(wer=w.wer, cer=c.cer, mer=w.mer, wil=w.wil, wip=w.wip,
                word_hits=w.hits, word_substitutions=w.substitutions,
                word_deletions=w.deletions, word_insertions=w.insertions,
                reference_words=w.hits+w.substitutions+w.deletions,
                char_hits=c.hits, char_substitutions=c.substitutions,
                char_deletions=c.deletions, char_insertions=c.insertions,
                reference_chars=c.hits+c.substitutions+c.deletions,
                exact_match_rate=sum(a == b for a,b in zip(refs,hyps))/len(refs))


def signal_metrics(audio, sample_rate):
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not x.size or sample_rate <= 0 or not np.isfinite(x).all():
        raise ValueError("Audio must be non-empty, finite mono or samples-by-channels")
    frame_size = max(1, int(sample_rate * .02))
    frames = [x[i:i+frame_size] for i in range(0, len(x), frame_size)]
    silent = np.array([np.sqrt(np.mean(f*f)) < .01 for f in frames])
    active = np.flatnonzero(~silent)
    db = lambda v: float(20*np.log10(max(v, 1e-12)))
    return dict(duration_s=len(x)/sample_rate, sample_rate=sample_rate,
                peak_dbfs=db(np.max(np.abs(x))), rms_dbfs=db(np.sqrt(np.mean(x*x))),
                clipping_ratio=float(np.mean(np.abs(x)>=.999)),
                silence_ratio=float(sum(len(f) for f,s in zip(frames,silent) if s)/len(x)),
                leading_silence_s=float(active[0]*frame_size/sample_rate) if active.size else len(x)/sample_rate,
                trailing_silence_s=max(0., (len(x)-(active[-1]+1)*frame_size)/sample_rate) if active.size else len(x)/sample_rate,
                dc_offset=float(np.mean(x)))


def summarize(rows):
    generated = [r for r in rows if r.get("status") in ("generated", "ok", "scoring_failed")]
    scored = [r for r in rows if r.get("status") == "ok"]
    result = {"attempted": len(rows), "generated":len(generated), "scored":len(scored),
              "generation_failure_rate": 1-len(generated)/len(rows) if rows else None}
    measured_limits = [r for r in generated if 'generation_token_limit' in r]
    if measured_limits:
        limited = sum('generation_limit_reached' in r.get('quality_flags', []) for r in measured_limits)
        result.update(generation_limit_rate=limited/len(measured_limits),
                      generation_limit_samples=limited, generation_limit_checks=len(measured_limits))
    if generated:
        latencies = [r["latency_s"] for r in generated]
        result.update(latency_p50_s=float(np.percentile(latencies,50)),
                      latency_p95_s=float(np.percentile(latencies,95)),
                      rtf=sum(latencies)/sum(r["duration_s"] for r in generated),
                      peak_vram_mib=max(r["peak_vram_mib"] for r in generated),
                      audio_seconds=sum(r["duration_s"] for r in generated))
        for metric in ("clipping_ratio", "silence_ratio", "rms_dbfs"):
            result[metric] = float(np.mean([r[metric] for r in generated]))
        ttfa = [r["ttfa_s"] for r in generated if isinstance(r.get("ttfa_s"),(int,float))]
        if ttfa:
            result.update(ttfa_p50_s=float(np.percentile(ttfa,50)),
                          ttfa_p95_s=float(np.percentile(ttfa,95)),ttfa_samples=len(ttfa))
    if scored:
        modes = {r.get('normalization_id', 'orthographic') for r in scored}
        if len(modes) != 1:
            raise ValueError('Cannot combine different scoring normalizations')
        mode = modes.pop()
        refs, hyps = [r['reference'] for r in scored], [r['transcript'] for r in scored]
        result['normalization_id'] = mode
        result.update(errors(refs, hyps, normalization=mode))
        result["raw"] = errors([r["reference"] for r in scored], [r["transcript"] for r in scored], False)
        if mode == 'whisper_english':
            result['orthographic'] = errors(refs, hyps)
        # Resample prompts as clusters, keeping repeats together. Independent-seed
        # repeats of the same text must not inflate the effective sample count.
        clusters = {}
        for row in scored:
            m = errors([row["reference"]], [row["transcript"]], normalization=mode)
            totals = clusters.setdefault(row["id"], [0,0,0,0])
            totals[0] += m["word_substitutions"]+m["word_deletions"]+m["word_insertions"]
            totals[1] += m["reference_words"]
            totals[2] += m['char_substitutions'] + m['char_deletions'] + m['char_insertions']
            totals[3] += m['reference_chars']
        if len(clusters) >= 3:
            counts = np.asarray(list(clusters.values()))
            rng = np.random.default_rng(42)
            samples = counts[rng.integers(0,len(counts),size=(1000,len(counts)))].sum(axis=1)
            ratios = samples[:,0]/np.maximum(samples[:,1],1)
            result["wer_ci95"] = [float(v) for v in np.percentile(ratios,[2.5,97.5])]
            char_ratios = samples[:,2]/np.maximum(samples[:,3],1)
            result['cer_ci95'] = [float(v) for v in np.percentile(char_ratios,[2.5,97.5])]
        result["unique_prompts_scored"] = len(clusters)
        result["by_category"] = {}
        for category in sorted({r["category"] for r in scored}):
            group = [r for r in scored if r["category"] == category]
            result["by_category"][category] = errors([r["reference"] for r in group], [r["transcript"] for r in group], normalization=mode)
        result['utterance_mean_wer'] = float(np.mean([errors([r['reference']], [r['transcript']], normalization=mode)['wer'] for r in scored]))
        result['utterance_mean_cer'] = float(np.mean([errors([r['reference']], [r['transcript']], normalization=mode)['cer'] for r in scored]))
    return result
