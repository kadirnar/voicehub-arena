# VoiceHub Arena — durum

Bu dosya 2026-09-11T19:37:06+00:00 tarihindeki yerel anlık kopyadır. Canlı durum: http://127.0.0.1:7860/

RTX 3090 (24 GB) bağlantısı `vast-3090-voicehub` olarak kuruldu. VS Code’da `/workspace/voicehub-arena` ayrı pencerede açıldı.

Uygulama testleri: 11 geçti. Arayüz JavaScript sözdizimi ve HTTP erişimi doğrulandı.

**34 modelin tamamı henüz değerlendirilmedi.** `all-models-en` iki İngilizce metin ve tek seed ile ilk kapsam taramasıdır; sunucuda çalışmaya devam ediyor. `voicehub-arena-extended` bu taramayı bekliyor ve uygun/onarılmış modeller için sekiz metin × üç seed değerlendirmesini başlatacak.

WER, CER, MER, WIL/WIP, tam eşleşme, sözcük ekleme/silme/değiştirme sayıları, RTF, p50/p95 gecikme, GPU bellek tepesi, süre, RMS/peak dBFS, clipping, sessizlik ve DC offset kaydedilir. İnsan MOS’u veya doğrulanmamış kalite puanı üretilmez.

CSM gibi kapalı checkpoint’ler yetkili Hugging Face erişimi gerektirir. Kısa taramadaki timeout bir kalite puanı değildir; indirme/yükleme süresi de sınıra dahildir. Irodori-TTS için İngilizce desteklenmiyor durumu kullanılır.

| Model | Anlık durum | Açıklama |
|---|---|---|
| vits | generated |  |
| supertonic | generated |  |
| kokoro | blocked | RuntimeError: Input and parameter tensors are not the same dtype, found input tensor with Float and parameter tensor with Half |
| vui | generated |  |
| bark | blocked | PermissionError: The pinned `suno/bark-small` release contains only a legacy pickle archive. Convert it once with `convert_official_bark_checkpoint(...)`, or explicitly pass `trust_official_pickle=True` for the digest-pinned source. |
| chatterbox | generated |  |
| conversationtts | timeout | Worker exit 124; see worker.log |
| cosyvoice | blocked | ValueError: Native CosyVoice requires a precomputed `speaker_embedding`. The frozen CAMPPlus frontend is not silently executed. |
| csm | blocked | PermissionError: Hugging Face denied access to sesame/csm-1b@c92a71e1c419772e25be7dc14d952c2521a740ab/model.safetensors (HTTP 401). Check the repository permissions and token. |
| dia | timeout | Worker exit 124; see worker.log |
| echo | timeout | Worker exit 124; see worker.log |
| f5tts | generated |  |
| fishtts | loading |  |
| gptsovits | pending |  |
| higgstts | pending |  |
| inflecttts | pending |  |
| irodoritts | pending |  |
| llasa | pending |  |
| melotts | pending |  |
| mosstts | pending |  |
| neutts | pending |  |
| omnivoice | pending |  |
| openvoice | pending |  |
| orpheustts | pending |  |
| outetts | pending |  |
| parlertts | pending |  |
| qwen3tts | pending |  |
| speecht5 | pending |  |
| styletts2 | pending |  |
| vibevoice | pending |  |
| voxcpm | pending |  |
| xtts | pending |  |
| zonos | pending |  |
| zonos2 | pending |  |

Sunucudaki uygulama ve işler Supervisor tarafından yönetilir. Kaynak kod ile bu anlık ses/rapor kopyaları yerelde saklandı; sonraki sonuçlar canlı arayüzde ve sunucudaki runs/ klasöründe oluşur.

Proje kullanımı ve ölçüm sınırlamaları için README.md dosyasını inceleyin.
