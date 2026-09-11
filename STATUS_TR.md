# VoiceHub Arena — durum

Anlık kopya: 2026-09-11T20:54:15+00:00. Canlı arayüz: http://127.0.0.1:7860/

RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.
`/workspace/voicehub-arena` sunucudaki proje; bu klasör yerel kaynak ve sonuç kopyasıdır.

**Tüm modellerin çalıştığı henüz doğrulanmadı.** İngilizce dışı Irodori-TTS kapsam dışıdır.
Gösterilen koşu: `english-extended`; durum `running`, aşama `generation`.
Kapsam: 34 model ailesi, 8 İngilizce metin × 3 seed.

NeuTTS-2e ana ağırlıklarına erişim doğrulandı; NeuCodec bağımlılığı ayrıca yetki istiyor.
Kimlik bilgileri proje dışında saklanıyor.
33 İngilizce modelin giriş sözleşmesi kontrolü geçti. Bu, GPU üretim başarısı anlamına gelmez.
Uygulama: 23 test geçti. VoiceHub düzeltmeleri: 48 test ve 6 alt test geçti.
Ek OpenVoice yükleme düzeltmesi: 12 test geçti.
SpeechT5 gerçek tokenizer’ı, 13 örnekte SentencePiece referansıyla eşleşti.

İlk kısa tarama `all-models-en` eski ayarlarla iki metin kullandı. Düzeltilmiş ayarlarla
`english-extended` tüm kaydı, sekiz metin ve üç seed ile tekrar değerlendirir.
Kısa taramadaki zaman aşımı indirme süresini de içerir; kalite puanı değildir.

WER, CER, MER, WIL/WIP, tam eşleşme, sözcük hataları, RTF, p50/p95 süre, GPU bellek
tepesi, RMS/peak dBFS, clipping, sessizlik ve DC offset gerçek çıktılardan hesaplanır.
VibeVoice için yeni Arena adaptörü ayrıca etiketlenir; tam referans dalga biçimi eşliği henüz doğrulanmadı.

| Model | Durum | Ses / planlanan | Açıklama |
|---|---|---:|---|
| neutts | blocked | 0 / 24 | PermissionError: Hugging Face denied access to neuphonic/neucodec@30c1fdd19e68aee65d542cf043750d4c0165893e/config.json (HTTP 403). Check the repository permissions and token. |
| vits | completed | 24 / 24 |  |
| supertonic | completed | 24 / 24 |  |
| kokoro | completed | 24 / 24 |  |
| vui | completed | 24 / 24 |  |
| inflecttts | blocked | 0 / 24 | ValueError: The official Inflect release uses a PyTorch pickle container. Review its origin and pass `trust_pickle_checkpoint=True` for one restricted, weights-only load, then export Safetensors for steady-state use. |
| styletts2 | blocked | 0 / 24 | ValueError: Unpinned YAML cannot be interpreted without a YAML runtime. Convert the configuration to typed VoiceHub JSON explicitly. |
| melotts | completed | 24 / 24 |  |
| speecht5 | blocked | 0 / 24 | TypeError: SpeechT5Processor.__call__() takes 1 positional argument but 2 were given |
| gptsovits | completed | 24 / 24 |  |
| openvoice | blocked | 0 / 24 | RuntimeError: Inference tensors do not track version counter. |
| cosyvoice | blocked | 0 / 24 | FileNotFoundError: The official CosyVoice3 snapshot publishes audited legacy llm.pt/flow.pt/hift.pt files, not native Safetensors. Run `convert_audited_cosyvoice_legacy_checkpoint` explicitly once, then load the resulting local artifact. |
| vibevoice | loading | 0 / 24 |  |
| outetts | pending | 0 / 24 |  |
| bark | pending | 0 / 24 |  |
| chatterbox | pending | 0 / 24 |  |
| conversationtts | pending | 0 / 24 |  |
| csm | pending | 0 / 24 |  |
| dia | pending | 0 / 24 |  |
| echo | pending | 0 / 24 |  |
| f5tts | pending | 0 / 24 |  |
| fishtts | pending | 0 / 24 |  |
| higgstts | pending | 0 / 24 |  |
| irodoritts | pending | 0 / 24 |  |
| llasa | pending | 0 / 24 |  |
| mosstts | pending | 0 / 24 |  |
| omnivoice | pending | 0 / 24 |  |
| orpheustts | pending | 0 / 24 |  |
| parlertts | pending | 0 / 24 |  |
| qwen3tts | pending | 0 / 24 |  |
| voxcpm | pending | 0 / 24 |  |
| xtts | pending | 0 / 24 |  |
| zonos | pending | 0 / 24 |  |
| zonos2 | pending | 0 / 24 |  |

## Ayrı doğrulama: repairs-en-01

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| kokoro | completed | 24 | 3.63% | 3.27% |
| inflecttts | completed | 24 | 4.15% | 3.36% |
| styletts2 | completed | 24 | 3.80% | 3.30% |
| speecht5 | completed | 24 | 3.11% | 2.91% |

## Tamamlanan 24 örneklik son doğrulamalar

Farklı koşuların en yeni tamamlanan ayarları gösterilir; ayarlar ve süre kapsamı README içindedir.

| Model | Koşu | WER | CER | RTF |
|---|---|---:|---:|---:|
| gptsovits | english-extended | 5.87% | 3.63% | 0.416 |
| inflecttts | repairs-en-01 | 4.15% | 3.36% | 0.015 |
| kokoro | repairs-en-01 | 3.63% | 3.27% | 0.018 |
| melotts | english-extended | 4.66% | 3.60% | 0.019 |
| speecht5 | repairs-en-01 | 3.11% | 2.91% | 0.268 |
| styletts2 | repairs-en-01 | 3.80% | 3.30% | 0.040 |
| supertonic | english-extended | 3.80% | 3.33% | 0.061 |
| vits | english-extended | 7.77% | 4.75% | 0.030 |
| vui | english-extended | 14.51% | 12.93% | 0.640 |

İşler Supervisor altında seri GPU kullanımıyla devam eder. Bu görevde 30 dakikalık
kontrol ve düzeltme takibi etkindir; değişmeyen durumlarda bildirim göndermez.
Ölçüm protokolü, hazırlama komutları ve yeniden üretme adımları README.md içindedir.
