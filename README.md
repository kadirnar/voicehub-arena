# VoiceHub Arena

A portable English TTS benchmark for all **33 English-capable VoiceHub model families**.
The default dataset is **Seed-TTS-Eval English**: 1,088 texts per model and **35,904
planned outputs**. Recognition uses pinned **Whisper large-v3**; results include WER,
CER, edit counts, MER/WIL/WIP, confidence intervals, latency, RTF and audio quality flags.

The previous RTX 3090 campaign and its monitoring have been stopped. This repository
starts no benchmark, web server or cloud instance automatically. Previous recordings
remain in the owner's separate local backup and are not included in Git.

## Quick start on a new Linux GPU

Use a **fresh clone for each GPU experiment** so hardware timing and saved configurations
remain separate. Checkpoint and dataset revisions are pinned; no random fallback models
or synthetic placeholder scores are used.

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
The complete 35,904-output Seed benchmark has **not** finished, and this repository
has **not** yet been evaluated on your next GPU. Successful inference does not by
itself establish good speech quality. See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md)
for interpretation and limitations. Historical eight-text `small.en` scores are not
mixed into the Seed / large-v3 leaderboard.

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
