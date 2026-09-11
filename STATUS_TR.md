# VoiceHub Arena — durum

Anlık kopya: 2026-09-11T20:15:19+00:00. Canlı arayüz: http://127.0.0.1:7860/

RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.
`/workspace/voicehub-arena` sunucudaki proje; bu klasör yerel kaynak ve sonuç kopyasıdır.

**Tüm modellerin çalıştığı henüz doğrulanmadı.** İngilizce dışı Irodori-TTS kapsam dışıdır.
Gösterilen koşu: `all-models-en`; durum `running`, aşama `generation`.
Kapsam: 34 model ailesi, 2 İngilizce metin × 1 seed.

NeuTTS için açılan hesap erişimi yeniden doğrulandı. Kimlik bilgileri proje dışında saklanıyor.
33 İngilizce modelin giriş sözleşmesi kontrolü geçti. Bu, GPU üretim başarısı anlamına gelmez.
Uygulama: 17 test geçti. VoiceHub düzeltmeleri: 48 test ve 6 alt test geçti.
SpeechT5 gerçek tokenizer’ı, 13 örnekte SentencePiece referansıyla eşleşti.

İlk kısa tarama `all-models-en` eski ayarlarla iki metin kullandı. Düzeltilmiş ayarlarla
`english-extended` tüm kaydı, sekiz metin ve üç seed ile tekrar değerlendirir.
Kısa taramadaki zaman aşımı indirme süresini de içerir; kalite puanı değildir.

WER, CER, MER, WIL/WIP, tam eşleşme, sözcük hataları, RTF, p50/p95 süre, GPU bellek
tepesi, RMS/peak dBFS, clipping, sessizlik ve DC offset gerçek çıktılardan hesaplanır.
VibeVoice için yeni Arena adaptörü ayrıca etiketlenir; tam referans dalga biçimi eşliği henüz doğrulanmadı.

| Model | Durum | Ses / planlanan | Açıklama |
|---|---|---:|---|
| vits | generated | 2 / 2 |  |
| supertonic | generated | 2 / 2 |  |
| kokoro | blocked | 0 / 2 | RuntimeError: Input and parameter tensors are not the same dtype, found input tensor with Float and parameter tensor with Half |
| vui | generated | 2 / 2 |  |
| bark | blocked | 0 / 2 | PermissionError: The pinned `suno/bark-small` release contains only a legacy pickle archive. Convert it once with `convert_official_bark_checkpoint(...)`, or explicitly pass `trust_official_pickle=True` for the digest-pinned source. |
| chatterbox | generated | 2 / 2 |  |
| conversationtts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| cosyvoice | blocked | 0 / 2 | ValueError: Native CosyVoice requires a precomputed `speaker_embedding`. The frozen CAMPPlus frontend is not silently executed. |
| csm | blocked | 0 / 2 | PermissionError: Hugging Face denied access to sesame/csm-1b@c92a71e1c419772e25be7dc14d952c2521a740ab/model.safetensors (HTTP 401). Check the repository permissions and token. |
| dia | timeout | 0 / 2 | Worker exit 124; see worker.log |
| echo | timeout | 0 / 2 | Worker exit 124; see worker.log |
| f5tts | generated | 2 / 2 |  |
| fishtts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| gptsovits | blocked | 0 / 2 | ValueError: `text_language` must specify the synthesis-text language. |
| higgstts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| inflecttts | blocked | 0 / 2 | ValueError: owensong/Inflect-Micro-v2 is audited only at immutable revision 132edf8700da8e0d3fa81cdc0fd8844926fc6582; found 96e9360236ffcd734344067f20b4005b13e6358d. |
| irodoritts | unsupported_language | 0 / 2 | The pinned VoiceHub model card does not advertise English: ja |
| llasa | timeout | 0 / 2 | Worker exit 124; see worker.log |
| melotts | blocked | 0 / 2 | ValueError: Native MeloTTS requires checkpoint-compatible precomputed linguistic features. Missing: input_ids, tone_ids, language_ids, bert_features, ja_bert_features. |
| mosstts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| neutts | blocked | 0 / 2 | PermissionError: Hugging Face denied access to neuphonic/neutts-2e@e24ca17d47b8cdd1cdab45792013b5a81547d1a5/config.json (HTTP 403). Check the repository permissions and token. |
| omnivoice | timeout | 0 / 2 | Worker exit 124; see worker.log |
| openvoice | blocked | 0 / 2 | ValueError: OpenVoice requires `speaker_audio_path` or `target_embedding`. |
| orpheustts | blocked | 0 / 2 | ValueError: Orpheus generation requires a non-empty `voice`. |
| outetts | blocked | 0 / 2 | RuntimeError: Tensor on device meta is not on the expected device cuda:0! |
| parlertts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| qwen3tts | timeout | 0 / 2 | Worker exit 124; see worker.log |
| speecht5 | blocked | 0 / 2 | TokenizerAssetError: VoiceHub supports SentencePiece UNIGRAM and BPE model types, not enum value 4. |
| styletts2 | blocked | 0 / 2 | ValueError: No default checkpoint configured; add a reviewed checkpoint override |
| vibevoice | blocked | 0 / 2 | RuntimeError: VoiceHub has native VibeVoice TTS graphs, but high-level cached-prompt synthesis is not enabled: cache serialization, chunk boundaries, and waveform parity have not yet been independently verified. Load the realtime checkpoint and use ` |
| voxcpm | timeout | 0 / 2 | Worker exit 124; see worker.log |
| xtts | blocked | 0 / 2 | FileNotFoundError: XTTS reference audio was not found; pass an existing path or a non-empty sequence of existing `speaker_audio_path`s. |
| zonos | loading | 0 / 2 |  |
| zonos2 | pending | 0 / 2 |  |

İşler Supervisor altında seri GPU kullanımıyla devam eder. Bu görevde 30 dakikalık
kontrol ve düzeltme takibi etkindir; değişmeyen durumlarda bildirim göndermez.
Ölçüm protokolü, hazırlama komutları ve yeniden üretme adımları README.md içindedir.
