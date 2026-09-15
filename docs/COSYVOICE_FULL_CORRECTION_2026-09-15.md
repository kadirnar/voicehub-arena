# CosyVoice 3 — corrected full English evaluation

The complete 1,088-text English Seed-TTS-Eval resynthesis and Whisper-large-v3 scoring are finished. The corrected full-run WER is **1.7416%** and CER is **0.6234%**. The original, invalidated scores were 13.8156% and 7.7099%. These are full-split results, not the earlier eight-text pilot.

| Metric | Original implementation | Corrected implementation |
|---|---:|---:|
| Texts generated and scored | 1,088 | 1,088 |
| Corpus WER | 13.8156% | 1.7416% |
| WER 95% bootstrap interval | 12.6853–14.9670% | 1.4639–2.0445% |
| Corpus CER | 7.7099% | 0.6234% |
| CER 95% bootstrap interval | 7.0946–8.3776% | 0.5188–0.7429% |
| Word substitutions / deletions / insertions | 1,286 / 114 / 250 | 167 / 24 / 17 |
| Normalized exact transcript match | 41.91% | 85.48% |
| Generated audio duration | 4,108.52 s | 4,108.52 s |
| Synthesis RTF | 0.7114 | 0.7026 |

The repair changes the HiFT magnitude clamp order and final LeakyReLU slope to match the publisher. The checkpoint is [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/tree/29e01c4e8d000f4bcd70751be16fa94bf3d85a18), December 2025 base `llm.pt`, not the RL checkpoint. No checkpoint substitution was made.

The original and corrected runs use the same 1,088 targets, speaker embedding, instruction, flow steps, seed 42, one repeat and evaluator settings. Configuration equality was checked against the original run. Scoring uses `Systran/faster-whisper-large-v3` revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, CUDA FP16, English, beam 5, temperature 0, no VAD and no previous-text conditioning; `whisper_english` normalization.

All 1,088 corrected WAV SHA256 hashes, sample rates, durations, finite samples and per-record WER/CER calculations were independently verified. Corpus rates and both 1,000-resample confidence intervals were independently recomputed. No generation-limit truncation was reported. A second check verified all 1,088 byte ranges in the packed audio archive.

The corrected waveform has 5,756 samples at or above absolute amplitude 0.9899999 out of 98,604,480 samples (0.00584%). The earlier investigation found severe saturation in every original recording; the corrected audio no longer exhibits that pervasive limiting. Timing differences here are single-run measurements, not evidence of a statistically significant speedup.

## Data, preservation and reproducibility

- [Corrected audio, records, configuration and verification](https://huggingface.co/datasets/kadirnar/voicehub-arena-seed-tts-eval/tree/main/corrections/cosyvoice-v3-hift-20260915).
- [Current leaderboard, charts and all samples](https://kadirnar-voicehub-arena.static.hf.space/index.html).
- [Historical CosyVoice records](https://kadirnar-voicehub-arena.static.hf.space/reports/cosyvoice-correction-2026-09-15/original-model-records.json) and [original leaderboard](https://kadirnar-voicehub-arena.static.hf.space/reports/cosyvoice-correction-2026-09-15/original-leaderboard.json).
- [Earlier controlled vocoder, Llasa and Dia investigation](https://kadirnar-voicehub-arena.static.hf.space/quality-audit.html).
- [Additional six-model audit](https://kadirnar-voicehub-arena.static.hf.space/high-wer-audit.html).

The original `audio_shards/cosyvoice.tar` remains intact at immutable dataset revision `7d02b49322a9731ea9b1cc6279ef3f7a80949ae8`. Corrected audio is stored in a new versioned archive. The active table and CosyVoice sample rows now use the corrected result; all 32 other model measurements are unchanged. Earlier analytical plots are preserved with the correction report, and current plots are regenerated from the updated data.

Use `scripts/verify_cosyvoice_correction.py` with the backed-up corrected run, original runs, complete source JSONL and a fresh output directory to reproduce validation and packing. Source files are retained in the run metadata; the corrected vocoder SHA256 is `212f82751d7bedfd32349fb8aa3393489b6e2e70881677d291aefd2b55f3ed21`. The three publisher-parity vocoder regression tests passed before the full run.

This remains a fixed-provider-voice intelligibility benchmark. WER/CER do not measure naturalness or speaker similarity. Llasa and Dia remain under quality review. Successful correction of CosyVoice does not resolve those models' separate issues.
