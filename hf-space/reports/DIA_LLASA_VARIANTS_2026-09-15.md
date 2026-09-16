# Dia, Dia2 and Llasa evaluation

## Active scope — revised by user

Only **Llasa-1B, Llasa-3B and Llasa-8B** remain active, alongside Dia 1.6B and Dia2 1B/2B (six experiments total). Multilingual, TextChat and multi-speaker variants were removed from the queue and active UI. The partially generated Multilingual full run was cancelled. `configs/variant-selection.json` controls execution and visibility independently of the immutable provenance manifest. The main comparison excludes the old Multilingual Llasa baseline. The following broader inventory and paired controls describe historical work, not additional queued models.

## Verified full results — 16 September 2026

Each row evaluates all 1,088 English targets with the frozen fixed-reference and Whisper-large-v3 protocol. Local verification independently matched target IDs/text, every real WAV hash and finite waveform, corpus edit counts, publication receipts and pinned audio byte ranges. Machine-readable verification is in `hf-space/reports/variants/*-full-verification.json`.

| Model | Evaluated | Real recordings | No audio | Corpus WER | Corpus CER |
|---|---:|---:|---:|---:|---:|
| Llasa-1B | 1,088 | 1,088 | 0 | 1.1555% | 0.4151% |
| Llasa-3B | 1,088 | 1,088 | 0 | 0.9378% | 0.2574% |
| Llasa-8B | 1,088 | 1,031 | 57 | 5.1830% | 4.2924% |
| Dia 1.6B · reference | 1,088 | 1,088 | 0 | 4.9653% | 4.3162% |
| Dia2-1B | 1,088 | 1,088 | 0 | 2.7548% | 1.4194% |
| Dia2-2B | 1,088 | 1,088 | 0 | 2.1686% | 0.9537% |


All 57 Llasa-8B failures emitted only EOS token 128261, before codec decoding or ASR. Their 505 reference words are retained as deletions under `empty-output-deletions-v1`; no audio is fabricated. The 1,031 successful recordings alone yield 0.9967% WER, which is a secondary diagnostic and does not replace the all-target 5.1830% score. The fixed-seed sampling settings remain unchanged.

All six selected reference experiments are complete. The current comparison contains 38 configurations: 32 selected baselines and 6 verified full reference experiments. Original unconditioned results remain historical evidence, with the removed Multilingual baseline excluded from the active view.

## Historical baseline and inventory

The previous full scores are **unconditioned** experiments: Llasa-1B-Multilingual WER 73.99%, Dia-1.6B-0626 WER 67.35%. Previous audits reproduced the corpus edit counts and compared independent LM/decoder implementations. They did not establish that reference conditioning or synthesis protocol was optimal. Do not describe these scores as an intrinsic limit of either model.

A new fixed-reference campaign is frozen in `configs/variant-campaign.json`. It covers every official HKUSTAudio Llasa checkpoint found on 15 September 2026 (eight models), both official Dia2 checkpoints, and an independently prepared Dia 1.6B reference experiment. Community fine-tunes and duplicate quantizations are outside this finite inventory. Each checkpoint and the official Dia2 source are commit-pinned.

## Paired pilot evidence

Eight evenly spaced source indices were chosen before running the variants: 0, 155, 310, 465, 621, 776, 931, 1087. They contain 91 normalized reference words. Independent Transformers Llasa-1B-Multilingual, identical BF16 LM, seed, sampling and ASR settings:

| Condition | Word edits | WER | CER |
|---|---:|---:|---:|
| No reference | 39 / 91 | 42.86% | 38.43% |
| Fixed Emily reference | 3 / 91 | 3.30% | 1.01% |

A separate, fully independent Transformers Dia processor/LM/DAC comparison on the same eight texts measured WER **51.65% without reference → 7.69% with reference**. This is also a pilot, not a replacement full score.

Reference conditioning materially changes these small controls. These eight-text controls alone do not establish a full-corpus score; the separately completed full experiments are reported above. The 73.99% full result must not be compared numerically to the 3.30% eight-text pilot as if coverage were the same. No original score or artifact has been overwritten.

The Llasa reference/text join explicitly preserves a word boundary. The previous unconditioned experiment had no reference text, so this boundary fix alone does **not** explain its high WER. Llasa uses explicit top-k 50 as in the publisher's Transformers default. Its native codec was independently checked in the earlier audit; the new pilot additionally saves the reference reconstruction.

## Llasa 8B early-EOS investigation

The 8B pilot emitted only token 128261 (`SPEECH_GENERATION_END`) on source index 931, after six successful samples. Pinned checkpoint/tokenizer hashes matched the release. An isolated replay with identical seed 42 and prompt reproduced the same immediate EOS in BF16 and after casting the same loaded weights to FP32 for computation; this failure occurs before codec decoding or Whisper and is not an OOM. Switching precision did not fix it. Token-level evidence is saved in `hf-space/reports/llasa-8b-early-eos-diagnostic.json`. The completed eight-text pilot contains seven real recordings and one explicit failure: corpus WER 15.38%, CER 13.28%.

The evaluator previously stopped the whole model on this empty result. It now records **`generation_failed_scored`**, continues through the remaining targets and applies the explicit `empty-output-deletions-v1` policy: an empty hypothesis contributes all reference words/characters as deletions. No WAV is fabricated, no failed text is dropped, and neither seeds nor outputs are selected using ASR quality. Infrastructure exceptions still stop the model and are not converted to speech-quality scores. Generation success/failure counts remain separate from evaluated-text coverage; speed and waveform metrics use real recordings only. A successful-audio-only score is secondary and never replaces the all-target corpus WER. Earlier successful measurements are unchanged.

## Protocol and execution

Every full experiment targets all **1,088 exact English Seed-TTS-Eval texts**, seed 42, one generation each. The reference is the existing Emily audio and transcript, hashed in the manifest. Reference audio is removed from the scored output. This remains a fixed-voice intelligibility track, not the publisher's per-sample zero-shot speaker-similarity protocol.

ASR remains the pinned Whisper-large-v3 conversion, CUDA FP16, beam 5, temperature 0, no VAD, no previous-text conditioning. Corpus WER and CER sum edit counts; unsuccessful texts are never silently dropped. Pilots never enter full-run rankings.

- Llasa: independent Transformers tokenizer/LM, BF16, explicit top-k 50, top-p 1, temperature 0.8, total context 2,048; frozen audited XCodec2 in FP32.
- Dia 1.6B: independent Transformers feature extractor/tokenizer/processor/LM/DAC; FP32; released generation settings, total context 3,072 including reference. The processor's own audio prompt trimming is used.
- Dia2: official source `8687268f4ed3ed20704638fd353b51491de3b476`, BF16, CFG 2, audio temperature 0.8/top-k 50, text temperature 0.6/top-k 50, CUDA graph. The upstream runtime assumes the newer `torch.backends.cudnn.conv` API despite declaring Torch 2.8 support; `patches/dia2-torch28-cudnn.patch` selects the equivalent Torch 2.8 cuDNN backend when that namespace is absent. The loaded runtime file hash is recorded. Its official prefix planner consumes cached word timestamps from the pinned Whisper-large-v3 evaluator. Prefix planning and model loading are excluded from per-sentence timing.

Install with `scripts/setup_variant_runtime.sh` after the base environment. Run from the repository root:

```bash
source scripts/env.sh
.venv/bin/python -u scripts/run_variant_campaign.py
```

Use supervisor for the remote campaign. The controller takes `runs/.gpu.lock`, runs all eight-text pilots first, then all full experiments. Each generation/scoring phase uses a separate process. A failed pilot blocks only that model's full run. Audio hashes are verified before resume and before ASR. Model download staging is disposable and removed between experiments to fit the 8B checkpoint; results and original benchmark audio are retained.

`publish_variant_progress.py` verifies coverage, every target, every WAV hash, tar offsets and independently recomputed corpus metrics before publishing. Versioned dataset paths are under `experiments/dia-llasa-variants-20260915/{pilot,full}/`. Browser audio is commit-pinned and verified using an actual HTTP byte range. If publication fails, local evaluation artifacts remain available for retry.

The HF Space's `variants.html` shows the frozen inventory, separate pilot/full status, complete-run chart and per-sample recordings. Only verified, complete reference-conditioned runs are also appended to the main table/chart with a Reference badge; original rows are preserved. Combined JSON/CSV snapshots are saved alongside versioned dataset artifacts. Partial and pilot results cannot enter this combined view. The main view has a current-chart SVG download. The main chart now defaults to compact horizontal bars, with a shared zero-based scale across columns, all model labels/values, search and a retained vertical-bar option.

## Sources

- [Official Llasa models](https://huggingface.co/HKUSTAudio/models?search=Llasa)
- [Llasa multilingual usage and reference example](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual)
- [Official Dia2 runtime](https://github.com/nari-labs/dia2)
- [Transformers Dia reference preparation](https://huggingface.co/docs/transformers/model_doc/dia)
