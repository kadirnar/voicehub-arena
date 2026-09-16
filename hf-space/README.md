---
title: VoiceHub Arena
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
short_description: TTS benchmarks with official Llasa 1B, 3B and 8B
datasets:
  - kadirnar/voicehub-arena-seed-tts-eval
tags:
  - text-to-speech
  - leaderboard
  - benchmark
---

# VoiceHub Arena

The completed **33-model × 1,088-text English Seed-TTS-Eval campaign**:
35,904 generated and scored recordings, on one NVIDIA A100-SXM4 40 GB.

- Sortable full leaderboard, WER/CER confidence intervals, speed and signal metrics.
- Search and listen to **all 1,088 samples of every model**, including A/B comparison
  on the same source text. The sample list is paginated, not truncated.
- Downloadable JSON/CSV records and scientific plots.
- [Permanent audio and results dataset](https://huggingface.co/datasets/kadirnar/voicehub-arena-seed-tts-eval).
- [Source repository](https://github.com/kadirnar/voicehub-arena) (repository access required).

This is a static results viewer. It requires no running GPU or inference server.
The audio dataset revision is pinned in `data/leaderboard.json` after publication.

## Interpretation

ASR: pinned faster-whisper-large-v3, English, CUDA FP16, beam 5, temperature 0;
no VAD or previous-text conditioning. Normalization: `whisper_english`.
WER/CER are corpus-level ratios, not averages of shard scores.
95% intervals use 1,000 prompt-cluster bootstrap resamples (seed 42).

This run uses fixed provider voices/references. It is **not** the official
zero-shot speaker-identity/SIM protocol, a MOS test, or a claim of model superiority
across all voice configurations. The historical unconditioned Dia and Llasa runs
had high error rates. Separately verified fixed-reference results are listed below;
the removed Multilingual Llasa baseline is excluded from active comparisons.

Source texts: [ByteDance Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval),
publisher revision `752f4297f090c46bb1a55a1f7439e5944ddefe8d`, English `en/meta.lst`.
Checkpoint provenance, sample hashes and full verification records accompany the data.

## Quality audit — 15 September 2026

CosyVoice is **Fun-CosyVoice3-0.5B-2512, base llm.pt**. The archived 13.82% WER
is affected by a confirmed HiFT implementation defect and is excluded from ranking.
The corrected full 1,088-text evaluation is verified: **WER 1.7416%, CER 0.6234%**.
The current table, samples and plots use this full run. The selected eight-text
pilot was not used as the replacement score. [Verification and preserved original data](https://kadirnar-voicehub-arena.static.hf.space/cosyvoice-correction.html). Llasa and Dia
remain under quality review after independent LM/codec checks.

[Read the investigation and paired audio](https://kadirnar-voicehub-arena.static.hf.space/quality-audit.html).

## Charts and table

The leaderboard defaults to compact horizontal bar charts with values and
WER/CER confidence intervals; vertical bars remain available. Select a metric
and compare 6, 12, all selected configurations, or your own models.
The complete table remains directly below the chart. Click a bar to
open its recordings. The Plots tab provides downloadable PNG/SVG bar charts for
WER, CER, RTF and GPU memory; all axes start at zero. Review and invalidation
markers apply to chart results as well as the table.

## Additional high-WER audit

[Six-model investigation](https://kadirnar-voicehub-arena.static.hf.space/high-wer-audit.html) covers Vui, ConversationTTS, Bark Small, VITS/MMS, OpenVoice V2 and VoxCPM2. All 35,904 original records were recomputed; 6,528 WAVs passed integrity checks. The audit adds 132 scored diagnostic records, numerical comparisons, error-component plots and sample audio. A confirmed ConversationTTS duration-budget bug is repaired in bootstrap and covered by regression tests. Selected diagnostic rates remain separate from full benchmark scores. See `docs/HIGH_WER_AUDIT_2026-09-15.md` for findings and reproduction steps.

## Active Llasa selection

Only official HKUSTAudio/Llasa-1B, Llasa-3B and Llasa-8B remain in active evaluation. Multilingual, Preserve-TextChat and multi-speaker variants have been removed from the queue, live tables, charts and sample selectors. Dia and Dia2-1B/2B remain active (six experiments total). The original 33-model snapshot is retained for provenance; its Multilingual Llasa row is excluded from current comparisons. The current selected baseline contains 32 models.

## Verified full reference results — 16 September 2026

All four rows use **1,088 English targets**, fixed reference audio and pinned Whisper-large-v3. Dia2 full runs are still in progress in this snapshot.

| Model | WER | CER | Real recordings | No audio |
|---|---:|---:|---:|---:|
| Llasa-1B | 1.1555% | 0.4151% | 1,088 | 0 |
| Llasa-3B | 0.9378% | 0.2574% | 1,088 | 0 |
| Llasa-8B | 5.1830% | 4.2924% | 1,031 | 57 |
| Dia 1.6B · reference | 4.9653% | 4.3162% | 1,088 | 0 |

Llasa-8B's 57 failures emitted only EOS before synthesis. They remain in corpus WER/CER as empty-output deletion penalties, with no fabricated WAVs. Targets, real audio hashes and corpus edit counts were independently verified. [Full table, chart and recordings](https://kadirnar-voicehub-arena.static.hf.space/variants.html).
