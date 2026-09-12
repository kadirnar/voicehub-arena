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
git -C ../voicehub checkout d67853dcfdf4385ce504dde66d6f513a446ca294
git -C ../voicehub apply ../voicehub-arena/patches/voicehub-runtime.patch
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ../voicehub -e '.[test,prepare,linguistic,kokoro]'
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
`--score-each-model`, enabled in the extended job, runs CPU ASR after each model's
TTS worker exits. Scored results appear as models finish without overlapping
ASR with TTS timing. The final scoring phase retries any remaining unscored audio.

For the prepared English providers, install the system `espeak-ng` library and
put NLTK's `cmudict`, `averaged_perceptron_tagger` and
`averaged_perceptron_tagger_eng` resources in a private directory. On this server,
that directory is `/workspace/.nltk_data`. Run these from the repository root
after setting `HF_HOME=/workspace/.hf_home`:

```bash
.venv/bin/python scripts/prepare_cosyvoice.py
.venv/bin/python scripts/prepare_cosyvoice_checkpoint.py
.venv/bin/python scripts/prepare_styletts2.py
.venv/bin/python scripts/prepare_speecht5.py
.venv/bin/python scripts/prepare_linguistic.py melotts
.venv/bin/python scripts/prepare_linguistic.py gptsovits
.venv/bin/python scripts/prepare_vibevoice.py
.venv/bin/python scripts/prepare_conversationtts.py
.venv/bin/python scripts/validate_echo_codec.py
.venv/bin/python scripts/preflight.py
```

Hugging Face credentials stay outside the repository in the protected Hub token
store. The worker bridges this store to VoiceHub's native HTTP environment;
credentials are never included in run manifests. Audited primary release
revisions are explicit in `configs/models.json`. A gated model can still require
account access independently of a valid token. NeuTTS also downloads the
separately gated `neuphonic/neucodec`; access to the TTS checkpoint alone is
insufficient.

The bundled `patches/voicehub-runtime.patch` fixes runtime issues in the pinned
VoiceHub release: SpeechT5's actual CHAR SentencePiece model was rejected by
the unigram-only reader, and OuteTTS's non-persistent rotary buffers stayed on
the meta device after checkpoint assignment. It also fixes SpeechT5’s public
generation boundary to keep native acoustic processing inside synthesis, and
OpenVoice's lazy MeloTTS loading so parameters retain version counters under
inference mode. The patch has a checkpoint reload
forward-parity regression and a CHAR framing regression. The published
SpeechT5 tokenizer also matches SentencePiece 0.2.1 on all eight diagnostic
texts plus five edge cases. Run manifests identify the applied native diff.
Further patches restore CosyVoice's released repetition-aware sampling (RAS),
including full-distribution nucleus probabilities, and keep VibeVoice's
deterministic diffusion schedule on CPU during meta-device graph construction.
CosyVoice's sampler matches the vendored reference in 96 seed/history cases;
VibeVoice's scheduler regression checks a real solver step after meta construction.
Bark's public generation boundary also preserves raw text and generation controls;
its keyword-only processor runs inside synthesis. The native Bark suite passes
13 tests, including public generation through a tiny real Bark/Encodec graph.
ConversationTTS's pinned official archive contains NumPy scalar and duration
metadata. Its preparation script verifies the audited archive size and SHA-256,
uses a restricted weights-only allowlist, and validates all 187 tensor names and
shapes against the native graph before writing Safetensors. The run verifies the
converted checkpoint digest; the native loader's global policy is unchanged.
Dia now pads finished DAC channels while delayed channels drain their final
frames. Its optional native KV cache preserves the full-prefix path for
comparison. Twelve native tests cover real DAC decoding, cached/prefill logits,
padding, seeded token agreement, the public generation boundary, and training/export behavior. The repair uses
float32 and a 3,071-step limit, corresponding to the pinned release's 3,072-token
length including BOS. Generation-limit hits remain flagged in the scored rows.
`scripts/validate_dia_cache.py` compares actual release logits and greedy tokens
on the GPU before the repair set; this is distinct from the acoustic benchmark.
This check passed on the RTX 3090 with a maximum absolute logit difference of
0.000012875 and identical greedy tokens. Its record is saved in
`runs/validations/dia-cache-gpu.json`; no full waveform parity is claimed.
Fish S2-Pro explicitly enables its pinned one-time codec conversion: the native
converter verifies the official 1,871,099,728-byte archive and SHA-256, loads it
with `weights_only=True`, and validates every tensor name and shape before
writing Safetensors in `artifacts/fish-s2-codec`.
The pinned archive includes six deterministic attention masks/rotary tables.
Only those exact names are discarded after the official hash passes; unknown
tensors and unaudited archives remain strict errors. The real conversion passed
with 535 tensors, alongside 15 native tests and six subtests.
Echo's codec now matches the convolutional decoder actually returned by the
[pinned reference](https://github.com/jordandare/echo-tts/blob/2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e/autoencoder.py).
Legacy weight-normalization g/v pairs are translated only for declared modules,
with shape and collision checks; the full inventory remains strict. Floating
weights may change precision while boolean causal masks and integer buffers
retain their types. Six native tests passed. The real 541-tensor codec matched
the reference exactly in two short CPU encode/decode probes; the record is
`runs/validations/echo-codec-cpu.json`. This does not establish full TTS quality.
The reference comparison uses `einops==0.8.1` from the optional validation extra;
Arena inference continues to use the native implementation.

The initial four-minute coverage timeout includes downloads. Its timeouts are
followed by a full-registry run with sixty minutes per provider, eight texts
and three seeds. The longer deadline covers observed synthesis time across
24 samples, not only model download. Abandoned native download temporary files are reclaimed only
when no process holds them open and they have not changed for five minutes.
Resumable official Hub downloads are preserved. The job now manages native
and official Hub model caches together, counting hardlinked payloads once.
Between workers, a 45 GiB total cache budget and 32 GiB free-space reserve evict
older unused repository groups through native cleanup and the Hub revision API.
The next checkpoint, ASR checkpoint, open files and repositories with unfinished
downloads are protected. Source, credentials, prepared artifacts, audio, results
and dataset repositories are outside eviction. The cache tests use real Hub
cache layouts and hardlinks; all 32 Arena tests pass.
Immutable native cache files are SHA-256 verified before reuse without HTTP;
uncached immutable revisions use the resumable Hugging Face/Xet client. The
commit, size and SHA-256 (or Git blob SHA-1 for small Git objects) are verified
before atomically publishing the native cache entry. On the same filesystem,
the native entry hardlinks the immutable Hub blob, avoiding a second allocation;
cross-filesystem caches use a verified atomic copy. Existing immutable snapshots
cannot be overwritten with different bytes. Mutable branch references are
re-resolved through Hub metadata on each access and downloaded by the returned
commit with the same resumable client. A branch changing during a download
cannot silently change the audited bytes. Dead Linux download locks left by a terminated
worker are reclaimed before the next worker; live process locks remain intact.

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

Kokoro receives the official [Misaki](https://github.com/hexgrad/misaki) English
G2P output, with a pinned spaCy English model and eSpeak fallback, inside the
measured request. The first extended attempt used the native grapheme fallback
and produced poor pronunciation; it remains visible as a failed quality baseline.
The corrected provider is evaluated in `repairs-en-01`.

CosyVoice's `repairs-en-02` baseline scored 17.62% WER and includes two 40.96-second
outputs with incomplete transcripts. This is a completed measurement, not accepted
quality validation. The RAS correction in `repairs-en-04` reduced WER to 10.71%,
with zero generation-limit hits across 24 samples. Numbers and narration remain
weaker categories; the benchmark records these errors without hiding them. CosyVoice
rows record the actual speech-token count and flag generation-limit hits while
retaining those outputs in the ASR evaluation.

Prepared-input providers state their timing scope in the explorer. MeloTTS and
GPT-SoVITS use the vendored English G2P recipes and pinned auxiliary checkpoints;
their offline linguistic preparation is timed separately in each input manifest.
OpenVoice measures native MeloTTS base synthesis plus voice conversion. For the
English path, the upstream protocols deliberately zero the unused Chinese BERT
branch; these are the documented language-specific inputs. Inflect, StyleTTS2
and Zonos phonemization runs inside the measured request. SpeechT5 uses a real
published CMU Arctic SLT x-vector, with separate provenance, rather than its
runtime's all-zero default. Compare timing scopes as well as sample coverage.

VibeVoice uses the explicitly identified `arena-native-staged-v1` adapter over
VoiceHub's native realtime graph. Its five-text-token / six-speech-token loop
follows Microsoft's pinned reference implementation, using the official Emma
voice cache converted to tensor-only safetensors. It produced and scored 24
real GPU utterances in `repairs-en-04`: WER 4.66%, CER 3.97%, and median TTFA
104 ms on this RTX 3090. Upstream waveform parity is not claimed.
It records actual time to the first decoded audio chunk (TTFA); other providers
do not receive inferred streaming measurements.

Normalization: Unicode NFKC, lowercase, punctuation removal, apostrophe joining,
and whitespace collapse. CER includes spaces. Numeric spellings and abbreviations
are **not** automatically equated. The numbers item provides an explicit spoken
reference; inspect its transcript when interpreting errors.

ASR is pinned `Systran/faster-whisper-small.en` by default, CPU INT8, beam size 5,
temperature 0, no VAD, no previous-text conditioning and no reference prompt.
ASR runs after TTS processes exit so it does not occupy their GPU memory. WER/CER
combine ASR mistakes with synthesis mistakes; they are not human listening scores.
Use `--asr Systran/faster-whisper-large-v3` in a separate run for a stronger judge.

Requested primary revisions, actual native download commits/content digests,
VoiceHub commit and patch digest, package versions, dataset hash,
hardware and effective generation options are recorded. Secondary codecs and
tokenizers can still follow provider defaults; this first version does not claim
full transitive artifact pinning. Echo’s wrapper currently uses its default branch
for internal downloads; resolved artifact commits disclose what was actually used.
ZONOS2 follows VoiceHub’s pinned independent BF16 conversion, separately identified
in its provenance. Timeout includes metadata, download, load and
generation. Weight downloads are cached. A free-disk guard stops starting new
models below the configured reserve (32 GiB for new runs; historical runs used
8 GiB). This provides room for the large backbone and codec downloads; one
unusually large download can still exceed the reserve.
Load times include downloads on a cache miss; cache states are not identical
across providers, so these load times are not a controlled cold-start ranking.

**Not measured:** human MOS, DNSMOS/UTMOS, speaker-embedding similarity or aligned
PESQ/STOI. They need separate validated judges, ratings,
paired reference audio or streaming adapters. The UI never invents these scores.
The eight authored diagnostic prompts are not a standardized evaluation corpus.
An `all` run covers one primary checkpoint per registered TTS provider, not every
checkpoint/voice/size variant in each model family.
When the pinned checkout includes an audited language list, providers without
English support receive `unsupported_language` (for example Irodori-TTS, Japanese).
Missing language metadata is treated as unknown and does not skip a provider.

## Operations

The web explorer is read-only, binds to `127.0.0.1`, and is reached through SSH.
A feature-detected WebMCP read tool exposes the selected run; no supported WebMCP
validation context was available in Arc, so this optional interface is unverified.
`scripts/install_service.sh` installs its Supervisor service on this Vast image.
`scripts/run_extended.py` waits for the all-model coverage probe, then retries
the full registry with the current repaired configurations on all eight English
prompts with three seeds and a 60-minute per-model deadline. Japanese-only
providers remain explicitly outside the English evaluation scope.
`scripts/prepare_cosyvoice.py` explicitly computes a real 192-dimensional CAMPPlus
embedding on CPU from the official Emily audio, using the upstream feature recipe.
Install `.[prepare]` for this optional reference preparation step. Encoder revision
and both encoder/reference hashes are saved with the embedding.
`scripts/prepare_cosyvoice_checkpoint.py` converts the three officially audited
CosyVoice3 `.pt` files on CPU. Size, SHA-256 and complete tensor inventories are
checked before restricted weights-only conversion to native Safetensors. The
tokenizer comes from the same immutable snapshot. The run records and verifies
the resulting local artifact manifest before model loading. No upstream Python
or YAML code executes during conversion. The source's literal special-token list
is applied to the BlankEN tokenizer, then native token IDs are compared with
Qwen2TokenizerFast on eight prompts, an instruction and all added tokens.

```bash
supervisorctl status voicehub-arena voicehub-arena-benchmark
tail -f benchmark.log
.venv/bin/voicehub-arena report runs/all-models-en
.venv/bin/python -m pytest -q
```

`scripts/interleave_repair.py` waits until a named model finishes, pauses the
full runner, evaluates a selected repair set in a separate run, then resumes the
full run. The GPU lock and subprocess cleanup keep these phases serial; completed
results are preserved. `repairs-en-01` completed Kokoro, Inflect, StyleTTS2 and
SpeechT5 on all 24 samples per model. Further repair sets receive separate run
directories so the original failures and changed configuration remain inspectable.
`repairs-en-03` exposed VibeVoice's scheduler issue after successful weight loading.
`repairs-en-04` completed VibeVoice and CosyVoice after OuteTTS finished all 24
samples (WER 3.97%). Chatterbox completed at 3.45% WER; `repairs-en-05` completed
Bark at 6.56% WER and 4.48% CER. CSM completed at 5.35% WER and 3.94% CER.
`repairs-en-06` completed ConversationTTS at 17.62% WER and 15.56% CER; the
numbers prompt has extra words in its ASR transcripts and still needs listening
review to distinguish synthesis errors from ASR errors. F5-TTS
completed at 3.97% WER and 3.42% CER. HiggsTTS completed at 4.84% WER and 3.60% CER.
`repairs-en-07` passed Dia's GPU comparison, but the first acoustic attempt
rejected `use_cache` at the public API boundary. That allowlist is now fixed and
covered by a real tiny-model public generation test. Echo then exposed the codec
architecture and weight-normalization mismatch described above; those are now
corrected. Echo completed 24 GPU samples in `repairs-en-10`: WER 3.97%, CER
3.30%, RTF 0.592 and zero generation failures.
Fish S2-Pro exposed the six extra runtime tables, which are now handled by the
audited converter. `repairs-en-08` completed Dia and Fish S2-Pro.
Dia completed all 24 repair samples with 11.74% WER, 9.51% CER and no generation
limit hits. Llasa completed with 19.00% WER and 14.17% CER; this remains a quality
concern. Fish completed all 24 repair samples with 3.80% WER and 3.27% CER.
There are now 23 families with a completed 24-sample evaluation.
The initial extended pass has finished; eight later families hit the disk
preflight limit. Unused official CSM and CosyVoice caches were removed after
checking active file handles, reclaiming 9.03 GiB. Their outputs and prepared
artifacts remain intact. `repairs-en-09` also hit the disk preflight limit.
`repairs-en-10` completed Echo. OmniVoice then exposed a non-persistent RoPE
buffer left on meta after loading. Its loader now reconstructs the deterministic
frequencies from the attention config; the roundtrip test loads on meta and
compares logits with the original CPU graph. All 18 OmniVoice tests pass.
Orpheus subsequently ran out of disk, and later families hit preflight limits.
Inactive completed Fish, Higgs, Dia and Echo Hub cache revisions were removed,
reclaiming 32.24 GiB. The paired cache management above prevents those retained
Hub references from escaping the budget. `repairs-en-11` now runs MOSS-TTS,
OmniVoice, Orpheus, Parler, Qwen3-TTS, VoxCPM, XTTS, Zonos and Zonos2 serially. MOSS-TTS v1.5 exposed an incorrect
audited-header fingerprint. Independent parsing of the four SHA-verified shards
confirmed all 463 BF16 tensor names and shapes exactly match the native meta
graph (8,489,841,664 parameters). Its pinned fingerprint was corrected to
`b7ae3960b1982743228dc0a8b7c5abad602a6e7c4ca782787b7d8aaa22ed90f0`;
strict inventory and revision rejection remain in place. The full header audit
is saved in `runs/validations/moss-v15-header-inventory.json` and its compact
immutable config/artifact fixture is included in the native patch. The MOSS
native suite passed 10 tests and 8 subtests, including graph inventory and
altered-header/revision rejection. MOSS synthesis is being validated in
`repairs-en-11`.
NeuTTS-2e weights are accessible, but the separate
NeuCodec repository still returned HTTP 403 in the latest access check.

The existing instance has no mounted persistent volume. Recycle/destroy removes
its files, so keep the delivered local source and results mirror. Stop/start
preserves container files. This application never stops or destroys the instance.

Sources: [VoiceHub](https://github.com/kadirnar/voicehub),
[JiWER metrics](https://jitsi.github.io/jiwer/usage/),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[official reference samples](https://github.com/neuphonic/neutts).
