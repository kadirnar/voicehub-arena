# VoiceHub Arena

An English TTS benchmark application for every model family registered by
[VoiceHub](https://github.com/kadirnar/voicehub). The first version includes a
serial GPU runner, isolated model processes, reproducible manifests, saved WAVs,
ASR evaluation, JSON/CSV exports, and a private result explorer with audio playback.

## On this RTX 3090

The project is at `/workspace/voicehub-arena` on SSH host `vast-3090-voicehub`.
VS Code opens this folder in its own remote window. The local source mirror is
the `outputs/voicehub-arena` folder delivered with this task.

Open the explorer at <http://127.0.0.1:7860> while the SSH tunnel is connected.
Reconnect from the Mac with:

```bash
ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 vast-3090-voicehub
```

## Install and run

```bash
git clone https://github.com/kadirnar/voicehub.git ../voicehub
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ../voicehub -e '.[test]'
.venv/bin/voicehub-arena catalog
.venv/bin/voicehub-arena run --models all --output runs/english-v1
.venv/bin/voicehub-arena serve --runs runs
```

Run from the repository root so the dataset and reference paths resolve.
`run` defaults to eight English prompts, three seeds, one warm-up and a
15-minute timeout per model. A lightweight **coverage probe** can use
`--limit 2 --repeats 1 --timeout 240`; it is not a quality leaderboard.
Use `--models vits,supertonic` to select a subset.

`--resume` uses the saved configuration and skips all terminal model records,
including failures. After fixing a provider, use a new output directory to
retry it. `score RUN_DIRECTORY` retries unscored audio without regenerating it.
The run-level GPU lock prevents overlapping Arena runs in the same runs parent.
External GPU processes are not controlled by this application.

## Metrics and fairness

* **WER, CER, MER, WIL, WIP:** jiwer corpus-level edit metrics from independent
  faster-whisper ASR. Lower is better except WIP. WER can exceed 100% when
  there are many insertions. Results contain word hits, deletions, insertions,
  substitutions, exact sentence match and raw, unnormalized error rates.
* **WER confidence interval:** deterministic 1,000-sample bootstrap over prompt
  clusters, keeping repeated seeds together. Requires at least three unique
  prompts. Small-corpus intervals remain exploratory.
* **Performance:** synchronized generation latency p50/p95, real-time factor
  (sum of generation time / sum of generated audio duration), model load time,
  warm-up time, audio duration and peak PyTorch CUDA allocation.
* **Signal diagnostics:** RMS and peak dBFS, clipping ratio (absolute amplitude
  at least 0.999), DC offset, and silence / leading / trailing silence. Silence
  uses 20 ms frames below −40 dBFS RMS; it is not a VAD speech probability.
* **Coverage:** model and sample status, errors, generated/scored counts,
  generation failure rate and category-level error rates. Failed models receive
  no quality or performance score. Partial rows retain their real measurements.

The same text and seed schedule are used for all providers. Models use their
native default voices/settings except explicit overrides; Dia receives its
required speaker marker, excluded from the reference transcript. F5-TTS and
NeuTTS use the public official Emily reference, with provenance in
`datasets/reference/provenance.json`. This is not a speaker-similarity comparison.

Normalization: Unicode NFKC, lowercase, punctuation removal, apostrophe joining,
and whitespace collapse. CER includes spaces. Numeric spellings and abbreviations
are **not** automatically equated. The numbers item provides an explicit spoken
reference; inspect its transcript when interpreting errors.

ASR is pinned `Systran/faster-whisper-small.en` by default, CPU INT8, beam size 5,
temperature 0, no VAD, no previous-text conditioning and no reference prompt.
ASR runs after TTS processes exit so it does not occupy their GPU memory. WER/CER
combine ASR mistakes with synthesis mistakes; they are not human listening scores.
Use `--asr Systran/faster-whisper-large-v3` in a separate run for a stronger judge.

Primary checkpoint revisions, VoiceHub commit, package versions, dataset hash,
hardware and effective generation options are recorded. Secondary codecs and
tokenizers can still follow provider defaults; this first version does not claim
full transitive artifact pinning. Timeout includes metadata, download, load and
generation. Weight downloads are cached. A free-disk guard stops starting new
models below 8 GiB; a single in-flight download can still consume additional disk.
Load times include downloads on a cache miss; cache states are not identical
across providers, so these load times are not a controlled cold-start ranking.

**Not measured:** human MOS, DNSMOS/UTMOS, speaker-embedding similarity, aligned
PESQ/STOI or streaming TTFA. They need separate validated judges, ratings,
paired reference audio or streaming adapters. The UI never invents these scores.
The eight authored diagnostic prompts are not a standardized evaluation corpus.
An `all` run covers one primary checkpoint per registered TTS provider, not every
checkpoint/voice/size variant in each model family.

## Operations

The web explorer is read-only, binds to `127.0.0.1`, and is reached through SSH.
A feature-detected WebMCP read tool exposes the selected run; no supported WebMCP
validation context was available in Arc, so this optional interface is unverified.
`scripts/install_service.sh` installs its Supervisor service on this Vast image.
`scripts/run_extended.py` waits for the all-model coverage probe, then evaluates
providers that produced audio, timed out in the short probe, or have repaired
Kokoro, Bark and CosyVoice configurations on all eight English prompts with
three seeds and a 20-minute per-model deadline. It does not retry every blocker.
`scripts/prepare_cosyvoice.py` explicitly computes a real 192-dimensional CAMPPlus
embedding on CPU from the official Emily audio, using the upstream feature recipe.
Install `.[prepare]` for this optional reference preparation step. Encoder revision
and both encoder/reference hashes are saved with the embedding.

```bash
supervisorctl status voicehub-arena voicehub-arena-benchmark
tail -f benchmark.log
.venv/bin/voicehub-arena report runs/all-models-en
.venv/bin/python -m pytest -q
```

The existing instance has no mounted persistent volume. Recycle/destroy removes
its files, so keep the delivered local source and results mirror. Stop/start
preserves container files. This application never stops or destroys the instance.

Sources: [VoiceHub](https://github.com/kadirnar/voicehub),
[JiWER metrics](https://jitsi.github.io/jiwer/usage/),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[official reference samples](https://github.com/neuphonic/neutts).
