# VoiceHub Arena

A portable English TTS benchmark for all **33 English-capable VoiceHub model families**.
The default dataset is **Seed-TTS-Eval English**: 1,088 texts per model and **35,904
completed outputs** in the published A100 campaign. Recognition uses pinned **Whisper large-v3**; results include WER,
CER, edit counts, MER/WIL/WIP, confidence intervals, latency, RTF and audio quality flags.

The previous RTX 3090 campaign and its monitoring have been stopped. This repository
starts no benchmark, web server or cloud instance automatically. Previous recordings
remain in the owner's separate local backup and are not included in Git.

## Quick start on a new Linux GPU

Use a **fresh clone for each GPU experiment** so hardware timing and saved configurations
remain separate. Checkpoint and dataset revisions are pinned; no random fallback models
or synthetic placeholder scores are used.
This repository is private: use authenticated Git on the new host, or transfer the
source ZIP and extract it before running the setup commands.

```bash
git clone https://github.com/kadirnar/voicehub-arena.git
cd voicehub-arena
# On a Debian/Ubuntu image, if these tools are not already installed:
sudo apt-get update
sudo apt-get install -y git espeak-ng libsndfile1 ffmpeg
# Install uv with your environment's package manager if it is missing.
bash scripts/bootstrap.sh
source scripts/env.sh
hf auth login
python scripts/prepare_all.py
python scripts/doctor.py
```

`bootstrap.sh` installs Python 3.12 dependencies, checks out VoiceHub at
`d67853dcfdf4385ce504dde66d6f513a446ca294`, applies the bundled runtime fixes and
installs PyTorch 2.8 / torchaudio 2.8. Its default wheel index is CUDA 12.8.
Set `PYTORCH_INDEX_URL` before setup if your host requires another compatible wheel
build. The script does not modify NVIDIA drivers. `doctor.py` verifies the selected
CUDA device, frozen dataset and required converted artifacts without TTS generation.
The original diagnostic runs used a 24 GB RTX 3090; a new GPU's compatibility and
full-split behavior still need validation with its pilot.

The Hugging Face login is interactive and stores credentials outside source files
under the ignored `$HF_HOME` cache. Your account needs access to any gated models,
including `neuphonic/neutts-2e` and `neuphonic/neucodec`. Credentials are never bundled.
`prepare_all.py` rebuilds required audited local checkpoints on CPU and checks them
against frozen hashes. Reference voice/vector files are bundled with provenance.
MeloTTS, GPT-SoVITS and OpenVoice linguistic inputs are prepared from each actual shard.

## Plan, then run

```bash
source scripts/env.sh
# No downloads, GPU work or saved-state changes:
python scripts/run_public_suite.py --plan-only
# Explicitly start all 33 English models on one selected GPU:
CUDA_VISIBLE_DEVICES=0 python scripts/run_public_suite.py
```

The plan contains **198 jobs**: six disjoint shards for each model. All models run
32-text pilots first (1,056 outputs), then 256-text cumulative panels (8,448 outputs),
then the full 1,088-text split. Existing verified samples within the same run are reused.
Models and ASR run serially under a GPU lock. This is a single-GPU runner; it does not
split a model across GPUs or launch a separate cloud machine.

To view results, run this in a second shell:

```bash
source scripts/env.sh
voicehub-arena serve --runs runs --port 7860
```

The server binds to localhost. From your laptop, replace `YOUR_GPU_SSH_HOST` with
your actual SSH host: `ssh -N -L 7860:127.0.0.1:7860 YOUR_GPU_SSH_HOST`, then visit
<http://127.0.0.1:7860/>. The UI provides coverage, transcripts, audio playback,
per-dataset WER/CER tables and JSON/CSV export.

For a managed process on a host with Supervisor installed:

```bash
sudo bash scripts/install_service.sh  # installs services in STOPPED state
sudo supervisorctl start voicehub-arena-benchmark voicehub-arena-web
# Stop all Arena work and the web service:
sudo supervisorctl stop voicehub-arena-benchmark voicehub-arena-web
```

On Vast base images, the installer uses the image's logging/environment wrappers
and sends service logs to `/var/log/portal/voicehub-arena-*.log`. Access remains
private through SSH forwarding; it does not change the portal or public ports.
The managed benchmark accepts `ARENA_CACHE_BUDGET_GIB` and `ARENA_MIN_FREE_GIB`
environment overrides. See [the A100 setup record](A100_SETUP_TR.md) for the
owner's current host and initial validation; it is separate from the old GPU archive.

For a pause at the next shard boundary, create `runs/public-english-v2/pause.request`.
Remove that marker before an explicit resume. An immediate managed stop also terminates
child workers; checkpointed rows remain available for `--resume-samples` verification.
Do not copy an old GPU's `runs/` into a new hardware comparison: keep it as an archive.
The controller rejects a saved plan whose model or dataset scope differs from configuration.

## Models and scope

The pinned English catalog is in `configs/catalog-english.json` and all entries are
included by default. Irodori-TTS declares Japanese and is excluded from this English
evaluation. `configs/public-scope.json` can optionally specify `model_types` for a
smaller *new* experiment. EmergentTTS, LibriTTS and LibriSpeech importers remain
available, but the default campaign remains **Seed-TTS-Eval only**.

The 33 families are Bark, Chatterbox, ConversationTTS, CosyVoice, CSM, Dia, EchoTTS,
F5TTS, FishTTS, GPT-SoVITS, HiggsTTS, InflectTTS, Kokoro, LLaSA, MeloTTS, MOSS-TTS,
NeuTTS, OmniVoice, OpenVoice, OrpheusTTS, OuteTTS, ParlerTTS, Qwen3-TTS, SpeechT5,
StyleTTS2, Supertonic, VibeVoice, VITS, VoxCPM, Vui, XTTS, Zonos and Zonos2.

Primary revisions/generation settings are frozen in `configs/public-models-lock.json`.
Checkpoint files are downloaded from their upstream publishers or reconstructed using
hash-verified conversion scripts; large weights and recordings are excluded from Git.
A dedicated checkpoint cache is limited to 45 GiB by default, with a 32 GiB download
reserve and 4 GiB audio-write floor. These are cache controls, not a total disk-size
estimate; allow additional room for environments, converted weights and recordings.
Use `--cache-budget-gib` and `--min-free-gib` to configure the new host's storage budget.

All 33 models completed an earlier 24-sample integration diagnostic at least once.
The A100 campaign completed on **15 September 2026**: **33/33 models, 198/198
shards, 35,904/35,904 scored recordings**, with all WAV hashes verified. New GPU
experiments still need their own validation. Successful inference does not by
itself establish good speech quality; Dia and Llasa retain unresolved high error rates. See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
for interpretation and limitations. Historical eight-text `small.en` scores are not
mixed into the Seed / large-v3 leaderboard.

## Published results and Hugging Face Space

- [Interactive Space](https://huggingface.co/spaces/kadirnar/voicehub-arena): sortable
  full leaderboard, all metrics, plots, and all 1,088 samples per model with A/B listening.
- [Permanent dataset](https://huggingface.co/datasets/kadirnar/voicehub-arena-seed-tts-eval):
  all 35,904 original WAV files in 33 indexed WebDataset tar shards, targets, ASR transcripts, per-sample metrics and SHA256 manifest.
- [`hf-space/`](hf-space/): the static app, all 35,904 text/metric records split by model,
  complete summary JSON/CSV and plots, versioned in this GitHub repository.

The Space does not require the A100 or a local SSH tunnel. Its audio URLs use the
pinned dataset revision recorded in `hf-space/data/leaderboard.json`. Model samples
are paginated across the **full 1,088-text set**, not limited to the first 256.
The viewer range-loads one WAV from its archive and verifies its SHA256 before playback.
Audio binaries live in the HF dataset; GitHub holds their hashes, paths and all
associated text/metric records. The source repository keeps its existing visibility.

Preview without GPU dependencies:

```bash
python -m http.server 7862 --bind 127.0.0.1 --directory hf-space
```

Regenerate a publication from a verified full campaign with
`scripts/export_space.py --help`, then run `scripts/pack_space_audio.py` to build
the indexed audio shards. Use a separate CPU environment for publishing:

```bash
python -m venv .venv-publication
source .venv-publication/bin/activate
pip install -r requirements/publication.txt
hf auth login
python scripts/publish_dataset.py --dataset-dir /path/to/exported-dataset
# Use the verified dataset commit SHA returned by the upload:
python scripts/publish_space.py --dataset-revision COMMIT_SHA
```

The dataset publisher verifies all 33 remote archive sizes and SHA256 digests.
Use `--verify-only --revision COMMIT_SHA` to repeat that read-only check.
The Space publisher pins playback to this immutable dataset commit. Authentication
may also use a protected `HF_TOKEN`; never put a token in the static app.

## Development checks

```bash
source scripts/env.sh
python -m pytest -q
python scripts/doctor.py --data-only
```

`environment.lock.txt` records the original machine's package inventory;
`requirements/runtime.txt` supplies portable Python dependencies while bootstrap
installs the GPU framework separately. Native model tests require the full environment.

Detailed native fixes and their historical validation are recorded in
[Runtime repair notes](docs/RUNTIME_NOTES.md).

## Quality audit — 15 September 2026

CosyVoice is **Fun-CosyVoice3-0.5B-2512, base llm.pt**. The archived 13.82% WER
is affected by a confirmed HiFT implementation defect and is excluded from ranking.
The vocoder is repaired; a fresh full 1,088-text run is in progress. The selected
eight-text 2.30% WER diagnostic is not a replacement full score. Llasa and Dia
remain under quality review after independent LM/codec checks.

[Read the investigation and paired audio](https://kadirnar-voicehub-arena.static.hf.space/quality-audit.html).

## Comparison bar charts

The Space shows a colored bar chart above the full table, with numerical labels,
zero-baseline axes and the measured WER/CER confidence intervals. Choose among
nine metrics and compare six, twelve, all 33 or a custom model selection. Click a
bar to open that model's audio samples. Invalidated archived scores are hatched;
quality-review flags remain visible. No intervals are invented for other metrics.

The Plots tab includes PNG/SVG bar charts for WER, CER, RTF and GPU allocation,
in both best-six and complete 33-model scopes. Rebuild these assets after every
leaderboard update, using the optional `plots` dependency:

```bash
python scripts/render_bar_charts.py
node --test tests/test_charts.cjs
```

`hf-space/reports/bars/manifest.json` records the input JSON hash, model IDs,
exact values, units and intervals for every export. The interactive chart reads
`data/leaderboard.json` directly, just like the table.
