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
