# CosyVoice, Llasa and Dia quality audit — 15 September 2026

**A real CosyVoice vocoder defect was confirmed and repaired. Its archived full-run score is invalidated for ranking.** The original 1,088 recordings and scores remain available as evidence. The corrected full 1,088-text result is now verified: WER 1.7416%, CER 0.6234% ([full correction](https://kadirnar-voicehub-arena.static.hf.space/cosyvoice-correction.html)); the eight-text diagnostic below is not a replacement benchmark.

The audit independently recomputed WER/CER from all 1,088 target/transcript pairs for each of the three models. All three matched the published corpus scores exactly. There is no arithmetic discrepancy in those scores. Poor synthesis and ASR behavior can still make them unsuitable as estimates of a correctly configured model's quality.

| Model / frozen checkpoint | Archived WER | Archived CER | Finding |
|---|---:|---:|---|
| CosyVoice 3, 0.5B, 2512 base | 13.8156% | 7.7099% | Confirmed native HiFT defect; excluded from ranking |
| HKUSTAudio/Llasa-1B-Multilingual | 73.9931% | 51.4343% | Quality unresolved; tested LM/codec paths agree closely with independent implementations |
| nari-labs/Dia-1.6B-0626 | 67.3533% | 59.2826% | Quality unresolved; ASR repetition amplifies some failures |

## Exact CosyVoice version and repair

The checkpoint is [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/tree/29e01c4e8d000f4bcd70751be16fa94bf3d85a18), revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`: **CosyVoice 3, 0.5B, December 2025 base `llm.pt`, not the RL checkpoint**. Local artifacts were reconstructed from hash-verified publisher weights. The table previously displayed only the family name and local artifact path; it now exposes the upstream version and revision.

The native HiFT decoder differed from the vendored publisher implementation in two places:

1. It clamped log magnitude before exponentiation. The publisher clamps magnitude after exponentiation: `exp().clamp_max(100)`.
2. Its final LeakyReLU used slope `0.1`; the publisher's final activation uses `0.01`.

All 1,088 archived WAV SHA256 values were verified. Across these files, **52.04% of waveform samples were at the internal ±0.99 limiter**; every recording had more than 10% of samples there. The existing `clipping_ratio` detects ±0.999 and therefore missed this internal saturation. This diagnostic limitation does not change the WER arithmetic.

The patch is in `patches/voicehub-runtime.patch`. Three numerical regression cases compare the complete decoder and extreme magnitude regime against publisher methods: **3 failed before, 3 passed after**, on the A100 host using CPU execution. The patch also applies cleanly to the pinned VoiceHub checkout.

### Controlled eight-text ablation

The selected examples span archived WER strata, including extreme failures. They are deliberately diagnostic and not a random or representative benchmark. Every variant receives the same LM tokens, mel features and captured RNG state. Recognition uses the original pinned Whisper-large-v3 protocol.

| Vocoder variant | Corpus WER on these 8 texts |
|---|---:|
| Archived implementation | 41.3793% |
| Magnitude clamp fix only | 3.4483% |
| Final activation fix only | 2.2989% |
| Both publisher corrections | 2.2989% |

**2.30% is an eight-text diagnostic result, not the corrected 1,088-text score.** Full resynthesis and scoring run separately in `runs/quality-cosyvoice-v3-fixed-seedtts-en-20260915`, preserving the historical campaign. Same texts, speaker embedding, instruction, flow steps, seed, repeats and ASR configuration.

## Llasa

Checkpoint revision: `7f094cb62b0a9779b334c60d039a61c5a6e04456`.
The native default omits `top_k`; the [publisher's Transformers example](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual) inherits `top_k=50`. An eight-text sampling ablation improved some failures but remained poor. This parameter difference alone did not fix the problem.

Four fixed source indices (0, 128, 512, 900) were generated with the independent Transformers LM in FP32 and publisher sampling settings. Their diagnostic WER was **61.40%**. Native/HF full-prefix FP32 logits agreed within a maximum absolute difference of `7.63e-5` across eight prefix checks; argmax agreed in all checks. These checks are limited evidence, not proof of equivalence for all autoregressive sequences.

The same speech codes were also decoded through the **legacy publisher XCodec2 decoder and original checkpoint**, revision `e412427ed30f0cf9d5e3c95562113deb10a32d03`, independently of the converted native decoder. FSQ latents matched exactly. Waveform RMS differences were `5.05e-7` to `7.31e-6`. Thus the tested failures are not explained by a large native decoder discrepancy. No claim that Llasa's inherent quality is this poor follows: voice conditioning, model/sampling configuration and ASR robustness remain possible factors. The score stays under review.

## Dia

Checkpoint revision: `ef2795fcc29c5abe6ffc91fd33808588b49bbc66`; FP32, `[S1]` prefix, temperature 1.8, top-k 50, top-p 0.9, guidance 3.0. Three named cases were tested: the first source row, the first empty transcript, and the largest archived WER.

Native and Transformers initial logits were exactly equal on all three prepared inputs. The native audio-prefix preparation was shared; text tokenization was checked independently. The first two generated outputs matched; the third diverged later. Both implementations failed on the two selected bad cases. The shared decoder was then checked independently against Transformers DAC using the same released weights; waveform RMS differences were `1.67e-5` to `1.05e-4`. These bounded checks did not establish an implementation defect.

In the complete run, **183/1,088 transcripts were empty**. Some transcripts contain unrelated boilerplate and repeated phrases. In one six-word target, Whisper emitted 185 inserted words, giving WER **3,183.33%**. Diagnostic VAD yielded an empty transcript and WER **100%** on that audio. This supports ASR hallucination as an amplifier of the error rate; it does not recover the missing target speech. The published no-VAD protocol was not silently changed.

Short inputs were particularly difficult: WER was 187.55% for texts under 40 characters (89 texts), 71.46% for 40–79 characters (805), and 30.60% for 80+ characters (194). The [publisher warns about unnatural output from inputs shorter than five seconds](https://github.com/nari-labs/dia) and recommends audio conditioning. This is consistent with a protocol limitation, not proof of the entire root cause. The full-run protocol uses fixed provider voices/references, not official per-prompt zero-shot speaker cloning.

## Evidence and reproduction

All machine-readable ablations, transcript checks and parity measurements are in [`hf-space/reports/quality-audit-2026-09-15/`](../hf-space/reports/quality-audit-2026-09-15/). A paired before/after WAV example accompanies the report. Test source: [`tests/test_cosy_vocoder_parity.py`](../tests/test_cosy_vocoder_parity.py).

ASR remained `Systran/faster-whisper-large-v3`, revision `edaa852ec7e145841d8ffdb056a99866b5f0a478`, CUDA FP16, English, beam 5, temperature 0, no VAD, no previous-text conditioning. Normalizer: `whisper_english`, whisper-normalizer 0.1.12. VAD results are separately labeled diagnostic variants.

Full source SHA256: `8a9386efb1768a90ffba66f8931e3887b53e9c9d1bf246eb46a8cce32bc7b1c1`. Historical run data and tar archive bytes were not overwritten. The current table uses the corrected CosyVoice full score. The invalidated original remains in the historical records; Llasa/Dia remain under review.
