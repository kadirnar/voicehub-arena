# VoiceHub Arena v1 — Seed-TTS-Eval English

The first release synthesizes **only Seed-TTS-Eval English** with all 33 English-capable
VoiceHub providers and transcribes each generated waveform with **Whisper large-v3**.
The active scope is **1,088 texts per model, 35,904 outputs across all 33 models**.
The user reduced the earlier five-split campaign to this scope. Existing Seed
results retain their original source hashes, model settings and scores. Other
datasets and their results remain archived; their queued work is deferred.
The existing eight authored prompts remain an integration diagnostic, with their
original scores preserved as history. They are excluded from this campaign.

## Frozen data catalogue

| Dataset / split | Full texts | Release scope | Source |
| --- | ---: | --- | --- |
| Seed-TTS-Eval, `en/meta.lst` | 1,088 | **Active** | [ByteDance release](https://github.com/BytedanceSpeech/seed-tts-eval) |
| EmergentTTS-Eval, publisher evaluation split named `train` | 1,645 | Deferred | [Boson AI release](https://huggingface.co/datasets/bosonai/EmergentTTS-Eval) |
| LibriTTS, `test-clean` | 4,837 | Deferred | [OpenSLR 60](https://www.openslr.org/60/) |
| LibriSpeech, `test-clean` | 2,620 | Deferred | [OpenSLR 12](https://www.openslr.org/12/) |
| LibriSpeech, `test-other` | 2,939 | Deferred | [OpenSLR 12](https://www.openslr.org/12/) |

The actual Seed archive contains 1,088 English records; the publisher's README
rounds this to 1,000. Every record from `en/meta.lst` is retained. Mandarin and
Mandarin hard cases are outside this English-only task.

The importer downloads from publisher sources, checks OpenSLR archive MD5s and
Emergent Parquet LFS SHA256s, records source archive SHA256s, and hashes every local
JSONL. Emergent is pinned to `a7406fa315a1df2a5ffcc1a782404648bf84fbd3`.
See each `datasets/public/*/manifest.json` and `datasets/public/suite.json`.
No dataset repository code is executed. Published synthetic baseline audio is
not used as ground-truth human speech.

The following describes archived LibriSpeech preparation, outside the active v1 scope.
LibriSpeech publishes all-capital ASR transcripts. Its runs apply
`librispeech_lowercase_v1` to the synthesis input of **every provider** on both
LibriSpeech splits. Original source text, IDs, references and JSONL hashes remain
unchanged; every output records `synthesis_text` and `input_text_transform`.
This avoids treating corpus capitalization as an instruction to spell words.
Other datasets keep identity preparation, including intentional acronyms.
MeloTTS/GPT-SoVITS offline features use the same prepared text as synthesis.

The pilot exposed the issue: InflectTTS spelled whole words in all-capital
LibriSpeech inputs; Kokoro and other providers also showed casing sensitivity.
The original attempts remain archived. All LibriSpeech attempts, including
providers with lower initial error rates, were assigned `-lc1` run names before deferral.
The report rejects a mixture of input-preparation versions. This correction was
made during pilot validation before the 256-text panels or full-split evaluation.

## Coverage and order

Each text is generated once with seed 42. No best-of-N selection is performed.
Within category, evolution-depth and text-length strata, a seed-42 SHA256 order
is frozen and the strata are interleaved. Stages use disjoint shards:

1. Pilot: first 32 Seed texts per model, **1,056 outputs** across 33 models.
2. Panel: extend to 256 Seed texts per model, **8,448 outputs** cumulatively.
3. Full: evaluate every remaining Seed text, 1,088 per model, **35,904 outputs** cumulatively.

Every provider is visited in each stage before moving to the next stage. These
are planned counts, not completed results. Dataset, phase, generated/scored/planned
counts and failures are displayed separately. The first panel deliberately
balances diagnostic strata; its score is not an unbiased estimate of the full
split. Full-split results use all source rows in their original proportions.
Failed synthesis or recognition reduces coverage; it never receives zero WER.
Models with incomplete coverage are ineligible for ranking in that comparison.

## Recognizer and metrics

The recognizer is [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3),
the CTranslate2 conversion of OpenAI Whisper large-v3, pinned to
`edaa852ec7e145841d8ffdb056a99866b5f0a478`. Public runs use CUDA FP16,
English transcription, beam size 5, temperature 0, no VAD, no reference-text
prompt and no conditioning on the previous segment's transcript. Synthesis and
recognition run in separate processes under the same exclusive GPU lock.

Primary WER/CER use `whisper-normalizer==0.1.12` EnglishTextNormalizer for both
reference and hypothesis. This normalizes English spelling, contractions and
many numeric expressions. It does not resolve every ambiguous spoken form.
Raw and simple orthographic error rates are retained for inspection. LibriTTS
uses the publisher's original text for synthesis and normalized text as reference.

Reports contain corpus WER/CER, utterance-mean WER/CER, MER, WIL/WIP, exact match,
word and character substitutions/deletions/insertions, and per-category metrics.
WER/CER 95% intervals bootstrap prompts (1,000 resamples; repeats, when present,
stay in their prompt cluster). These are prompt-level intervals, not speaker- or
book-level intervals. Latency p50/p95, RTF, PyTorch peak allocated VRAM, clipping,
silence and supported generation-limit flags are reported alongside error rates.
Warm-up and model download/load time are excluded from synthesis timing.
Prepared MeloTTS/GPT-SoVITS linguistic inputs are generated from each actual shard,
hashed and labelled with their separate preparation timing.

StyleTTS2's `styletts2_nltk_quotes_v1` frontend removes the paired backticks that
NLTK introduces for opening double quotes. The released StyleTTS2 TextCleaner
also skips those unsupported characters. Prepared token IDs are tested against
that released cleaner; other unknown phonemes still fail strict validation.
This fixes two generation failures in the first Seed pilot. The old attempt is
preserved, and StyleTTS2 public jobs use fresh `-sf1` run names. Frontend versions
are frozen in run configurations and checked before resuming or pooling results.

## Interpretation

This is a **fixed-voice intelligibility track using published test texts**.
Each provider uses its reviewed default voice or fixed reference from the Arena
configuration. Seed prompt metadata is retained, but per-utterance Seed speaker
prompts are not used in this common track. Consequently these results are not a
reproduction of the publisher's zero-shot speaker-similarity protocol. The common
normalizer and CTranslate2 decode settings also differ from the original Seed
scoring script. Do not compare these numbers directly with paper leaderboards.

Emergent includes questions, emotion, paralinguistics, foreign words, syntactic
complexity and difficult pronunciation. WER/CER measure recognizable content;
they do not reproduce its official model-as-judge expressiveness win rate.
Naturalness, prosody and emotional fidelity need additional evaluations. No
human MOS, SIM, PESQ or STOI is fabricated.

LibriTTS and LibriSpeech share source books and speakers, so their results are
reported separately rather than treated as independent evidence. Original
LibriSpeech `test-other` recording noise is not transferred into newly synthesized
audio. Published splits do not establish that every model's training data excluded
their text or speakers. Cross-model training contamination is not verified.

## Run and resume

```bash
uv pip install --python .venv/bin/python -e '.[datasets]'
.venv/bin/python scripts/prepare_public_datasets.py --cache .cache/public-sources
.venv/bin/python scripts/run_public_suite.py
```

On the configured RTX 3090, Supervisor service `voicehub-arena-public-suite`
owns the campaign. Its durable plan is `runs/public-english-v2/suite.json`;
each model/shard has an independent `runs/pub-v2-*` directory. The GPU lock is
shared with diagnostic/repair jobs. The controller checks hashes and resumes
verified generated waveforms after interruption. Existing completed shards are
not generated again.

`configs/public-scope.json` limits the first release to `seedtts_en`. The importer
also defaults to Seed only; other corpora require explicit `--datasets` selection.
After stopping the controller, `scripts/set_public_scope.py` applies a narrower
scope under both controller and GPU locks. It backs up the previous plan, keeps
selected job records unchanged, and moves excluded jobs and manifests to deferred
history without deleting audio. Expanding the scope requires a new explicit plan.
The UI defaults to full Seed coverage and shows per-model and all-model totals.

To pause the campaign without interrupting a model/shard, create
`runs/public-english-v2/pause.request`. The controller exits at the next job
boundary with `paused_at_job_boundary`. Remove that request before resuming.

There is a 32 GiB checkpoint-download reserve and a 4 GiB per-waveform storage
floor. This is a long-running campaign on one GPU. If storage becomes limiting,
the campaign records `waiting_for_storage` for verified archival and resumption;
it does not discard recordings, start another instance or report completion.
The existing scheduled monitor checks progress, errors and available disk space.

The UI provides dataset/coverage selectors and dataset-specific JSON/CSV exports:
`/api/public-suite/{dataset}?phase=pilot|panel|full` and
`/api/public-suite/{dataset}/leaderboard.csv?phase=pilot|panel|full`.
