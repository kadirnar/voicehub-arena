# Native methods campaign — paused for migration to another GPU

The A100 campaign and its recurring monitor were stopped at the user's request.
The new scheduler supports measured VRAM admission and parallel workers; it has
not started a replacement evaluation. See [parallel execution and new GPU setup](NATIVE_PARALLEL_TR.md).

The user requested all retained model families in their author libraries, every
supported generation method, and a broader metric suite. They confirmed all
methods with official defaults, rather than a Cartesian search over sampling
parameters or voices. Only English is in scope. Llasa remains limited to the
official 1B, 3B and 8B checkpoints.

This is a new campaign, `native-methods-20260916`. The completed historical
benchmark and its audio remain unchanged. `configs/native-methods.json` is a
review inventory, not a claim that every listed method already works: entries
with `verified_api=false` are blocked from execution. A native runtime import
guard rejects `voicehub` imports. Publisher code, checkpoint revisions and
per-environment dependencies are recorded separately.

## Dataset and controls

- All 1,088 original English Seed-TTS-Eval target IDs and texts, in their existing
  deterministic order. New dataset SHA256:
  `727f1b6821e59e135f88c0667bb71cceaa4537b2908e38dfbe931b4cb09a6d8f`.
- The original publisher archive is SHA256 verified before extraction. Its 666
  unique English conditioning recordings are mapped to targets using the
  published `en/meta.lst`; every reference WAV has a recorded SHA256.
- Voice-cloning/continuation methods receive the paired reference recording and
  reference transcript where the API supports it. Scored output excludes the
  prompt. Preset/auto/designed voices have no target speaker and SIM is N/A.
- Eight predeclared, evenly spaced diagnostic indices are used for smoke tests.
  Pilot scores never substitute for full results. No WER threshold selects
  successful models, seeds or outputs. Verified empty generations contribute
  empty transcripts/all deletions, and have no invented quality score or WAV.
- Seed 42 per target; one measured generation. Model loading and a separate
  warmup are excluded from timing. Reference preprocessing is included. TTFA is
  reported only for native APIs that actually yield audio chunks.

## Metrics

- WER, CER, MER, WIL, WIP and exact match: pinned
  `Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478`,
  CUDA FP16, English, beam 5, temperature 0, no VAD/previous-text conditioning.
  Both strings use `whisper-normalizer==0.1.12` EnglishTextNormalizer before
  corpus edit counts; raw target and transcript are retained.
- DNSMOS SIG, BAK, OVRL and P.808: Microsoft's released ONNX weights and source
  at `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`, non-personalized polynomial
  calibration. Resampling uses the original librosa `kaiser_best` algorithm;
  a temporary 16 kHz file avoids the upstream deprecated positional resample
  call. Original 9.01 s windows/repetition for short recordings are preserved.
  ONNX Runtime uses four intra-op threads and one inter-op thread to avoid CPU
  oversubscription on large hosts; equivalence is checked against saved pilot scores.
- UTMOS22 **strong learner**, not the full challenge ensemble: official
  `sarulab-speech/UTMOS-demo@47212055c2ecfb02d40cec2395233b83295d3d30`,
  `epoch=3-step=7459.ckpt`, official model and `Score` code.
- WavLM-large SIM-o: Microsoft's UniSpeech ECAPA speaker-verification code at
  `6112826ac13a4327f4c9a7afa2a505e35b763514` and pinned s3prl implementation;
  WavLM-large and fine-tuned speaker-head weights from the documented
  `k2-fsa/TTS_eval_models@e876de7154845cd668b599bd4866f1d354c723df` mirror.
  This is not mean-pooled raw SSL feature cosine similarity. The speaker-head
  parameter inventory must match; a missing/random head fails closed.
  The training-only `loss_calculator.projection.weight` is ignored as in the
  author's inference script; all embedding parameters must match.
- RTF, latency p50/p95, streaming TTFA, peak PyTorch allocated VRAM, silence,
  clipping, RMS and generation failures. Quality means include their measured
  coverage and deterministic 1,000-resample confidence intervals.

DNSMOS and UTMOS are model predictions, not human listening-test MOS. Unrelated
reference speakers are never used to fill SIM values for preset/auto/design
methods. There is no aligned natural recording of each generated waveform;
PESQ/STOI are not asserted as valid TTS quality metrics in this protocol.

## Runtime and reproducibility

`scripts/setup_native_initial.sh` creates an isolated native core runtime and a
Python 3.10 / Torch 1.13.1 legacy metric runtime. The historical environment is
preserved. The author's fairseq build is compiled without optional CUDA
training extensions (`CUDA_HOME` unset only for its build), avoiding the host
CUDA 12.8 versus Torch 11.7 build mismatch; inference still runs on the A100.
The old s3prl package's floating git dependency on huggingface_hub is replaced
with a pinned compatible installed dependency. Only its WavLM upstream module
is imported, rather than the hub's unrelated model collection.

Per-model native environments inherit compatible base packages through a `.pth`
file but override incompatible libraries locally. `arena-runtime.json` records
effective packages. TTS, Whisper and quality predictors run as separate processes.
The controller owns the legacy GPU lock, then admits its workers using live VRAM
and reservations (measured peak ×1.35 +1 GiB, plus at least 2 GiB/10% headroom).
Unknown memory profiles run alone first. CPU DNSMOS uses a separate worker pool.
Supervisor retains long-running jobs across
SSH disconnects. Downloads are resumable; completed recordings are hash checked
before reuse. No model is reported complete merely because its process exits.

Generation defaults and runtime versions are different provenance fields. Some
publisher packages pin older Torch versions; the native Dia, Chatterbox and XCodec2 adapters
currently use the host's Torch 2.8 runtime. This compatibility deviation is recorded
and must pass the native pilot before any full result is accepted. New generation
rows include an implementation/package provenance ID so later adapter additions
cannot silently rewrite the provenance of already measured rows.

Supertonic's pinned Python helper explicitly rejects GPU mode, so its official
CPU path is used and its settings identify the device. The pinned Dia2 source has
no public streaming-output API (listed as upcoming); those candidates are marked
unsupported, without inventing streaming chunks or TTFA. Its native reference
alignment uses Whisper-large-v3 internally and excludes the prefix from output.
NeuTTS-2E has fixed speakers (official Emily/neutral defaults); arbitrary reference
cloning belongs to NeuTTS-Air. Both Torch adapters use the author's NeuCodec 0.0.6
BIN loader, with immutable codec and W2V-BERT semantic-encoder revisions. The
author's codec rejects local repo paths, so narrow load hooks redirect its file
reads to the frozen snapshot while preserving the inference code. NeuTTS GGUF
streaming requires a separate runtime audit and remains pending.

Zonos has separate auto-voice, speaker-embedding cloning, audio-prefix, and
speaker-plus-prefix experiments. These use the author transformer backbone,
conditioning defaults and sampling defaults. Native audio-prefix decoding includes
the reference: exactly its encoded frame count times the DAC 512-sample hop is
removed before scoring. DAC and the optional speaker encoder are pinned separately.
Chatterbox uses its native five-file English checkpoint and native built-in or
paired-reference conditioning. Source audit enables a pilot; it does not establish
that inference and all metrics have passed. Runtime import checks and measured
pilot results are recorded separately.

Each experiment freezes its own contract before its first run. Publication
independently checks exact target coverage, WAV hashes, archive offsets and
recomputed metric aggregates. Immutable HF dataset revisions back browser audio;
the browser verifies each requested audio SHA256 before playback. Candidate and
failed methods stay visible; only fully verified published results enter charts.

After a completed full result is published, `offload_native_audio.py` verifies
the remote immutable archive's SHA256 and the unchanged result before reclaiming
its local WAVs and duplicate TAR. Original results/transcripts/receipts remain.
Historical recordings are outside this tool's scope. To restore a native run:

```sh
python scripts/offload_native_audio.py --experiment vits--default --phase full --restore
```

## Commands

From the repository root, after setting `PYTHONPATH` to the root:

```sh
python scripts/prepare_native_seed.py --archive /path/to/seedtts_testset.tar
bash scripts/setup_native_initial.sh
.venv/bin/python scripts/run_native_campaign.py --campaign native-new-gpu-20260916 --dry-run
.venv/bin/python scripts/publish_native_progress.py
```

The old A100 service `voicehub-native-campaign` is stopped. A new, explicitly
started service can use `deploy/voicehub-native-parallel.conf.example`. Add
`--publish` to opt into artifact publication/offload after reviewing the dry run.
Do not resume the historical campaign when changing GPUs or timing modes.

Source references: [DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS),
[UTMOS22](https://github.com/sarulab-speech/UTMOS22),
[speaker verification](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification),
[metric checkpoint mirror](https://huggingface.co/k2-fsa/TTS_eval_models),
[Seed-TTS-Eval](https://github.com/BytedanceSpeech/seed-tts-eval).
