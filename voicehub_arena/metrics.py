"""English ASR error rates and explicitly defined signal diagnostics."""
import re
import unicodedata
import numpy as np
import jiwer


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = "".join(c if not unicodedata.category(c).startswith("P") else ("" if c == "'" else " ") for c in text)
    return re.sub(r"\s+", " ", text).strip()


def errors(references, hypotheses, normalized=True):
    if len(references) != len(hypotheses) or not references:
        raise ValueError("Need equal, non-empty reference/hypothesis lists")
    refs = [normalize(t) for t in references] if normalized else references
    hyps = [normalize(t) for t in hypotheses] if normalized else hypotheses
    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    return dict(wer=w.wer, cer=c.cer, mer=w.mer, wil=w.wil, wip=w.wip,
                word_hits=w.hits, word_substitutions=w.substitutions,
                word_deletions=w.deletions, word_insertions=w.insertions,
                reference_words=w.hits+w.substitutions+w.deletions,
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
    if generated:
        latencies = [r["latency_s"] for r in generated]
        result.update(latency_p50_s=float(np.percentile(latencies,50)),
                      latency_p95_s=float(np.percentile(latencies,95)),
                      rtf=sum(latencies)/sum(r["duration_s"] for r in generated),
                      peak_vram_mib=max(r["peak_vram_mib"] for r in generated),
                      audio_seconds=sum(r["duration_s"] for r in generated))
        for metric in ("clipping_ratio", "silence_ratio", "rms_dbfs"):
            result[metric] = float(np.mean([r[metric] for r in generated]))
    if scored:
        result.update(errors([r["reference"] for r in scored], [r["transcript"] for r in scored]))
        result["raw"] = errors([r["reference"] for r in scored], [r["transcript"] for r in scored], False)
        # Resample prompts as clusters, keeping repeats together. Independent-seed
        # repeats of the same text must not inflate the effective sample count.
        clusters = {}
        for row in scored:
            m = errors([row["reference"]], [row["transcript"]])
            totals = clusters.setdefault(row["id"], [0,0])
            totals[0] += m["word_substitutions"]+m["word_deletions"]+m["word_insertions"]
            totals[1] += m["reference_words"]
        if len(clusters) >= 3:
            counts = np.asarray(list(clusters.values()))
            rng = np.random.default_rng(42)
            samples = counts[rng.integers(0,len(counts),size=(1000,len(counts)))].sum(axis=1)
            ratios = samples[:,0]/np.maximum(samples[:,1],1)
            result["wer_ci95"] = [float(v) for v in np.percentile(ratios,[2.5,97.5])]
        result["unique_prompts_scored"] = len(clusters)
        result["by_category"] = {}
        for category in sorted({r["category"] for r in scored}):
            group = [r for r in scored if r["category"] == category]
            result["by_category"][category] = errors([r["reference"] for r in group], [r["transcript"] for r in group])
    return result
