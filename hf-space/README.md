---
title: VoiceHub Arena
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
short_description: 33 TTS models with full English audio and metrics
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
across all voice configurations. Dia and Llasa have high transcript error rates
whose root causes are unresolved; their results remain visible.

Source texts: [ByteDance Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval),
publisher revision `752f4297f090c46bb1a55a1f7439e5944ddefe8d`, English `en/meta.lst`.
Checkpoint provenance, sample hashes and full verification records accompany the data.

## Quality audit — 15 September 2026

CosyVoice is **Fun-CosyVoice3-0.5B-2512, base llm.pt**. The archived 13.82% WER
is affected by a confirmed HiFT implementation defect and is excluded from ranking.
The vocoder is repaired; a fresh full 1,088-text run is in progress. The selected
eight-text 2.30% WER diagnostic is not a replacement full score. Llasa and Dia
remain under quality review after independent LM/codec checks.

[Read the investigation and paired audio](https://kadirnar-voicehub-arena.static.hf.space/quality-audit.html).

## Charts and table

The leaderboard now includes colored vertical bar charts with values and
WER/CER confidence intervals. Select a metric and compare 6, 12, all 33, or your
own models. The complete table remains directly below the chart. Click a bar to
open its recordings. The Plots tab provides downloadable PNG/SVG bar charts for
WER, CER, RTF and GPU memory; all axes start at zero. Review and invalidation
markers apply to chart results as well as the table.

## Additional high-WER audit

[Six-model investigation](https://kadirnar-voicehub-arena.static.hf.space/high-wer-audit.html) covers Vui, ConversationTTS, Bark Small, VITS/MMS, OpenVoice V2 and VoxCPM2. All 35,904 original records were recomputed; 6,528 WAVs passed integrity checks. The audit adds 132 scored diagnostic records, numerical comparisons, error-component plots and sample audio. A confirmed ConversationTTS duration-budget bug is repaired in bootstrap and covered by regression tests. Selected diagnostic rates remain separate from full benchmark scores. See `docs/HIGH_WER_AUDIT_2026-09-15.md` for findings and reproduction steps.
