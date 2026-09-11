# VoiceHub Arena — durum

Anlık kopya: 2026-09-11T23:37:10+00:00. Canlı arayüz: http://127.0.0.1:7860/

RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.
`/workspace/voicehub-arena` sunucudaki proje; bu klasör yerel kaynak ve sonuç kopyasıdır.

**Tüm modellerin çalıştığı henüz doğrulanmadı.** İngilizce dışı Irodori-TTS kapsam dışıdır.
Gösterilen koşu: `english-extended`; durum `running`, aşama `generation`.
Kapsam: 34 model ailesi, 8 İngilizce metin × 3 seed.

NeuTTS-2e ana ağırlıklarına erişim doğrulandı; NeuCodec bağımlılığı ayrıca yetki istiyor.
Kimlik bilgileri proje dışında saklanıyor.
33 İngilizce modelin giriş sözleşmesi kontrolü geçti. Bu, GPU üretim başarısı anlamına gelmez.
Uygulama: 29 test geçti. İlk VoiceHub düzeltmeleri: 48 test ve 6 alt test geçti.
Ek OpenVoice yükleme düzeltmesi: 12 test geçti.
CosyVoice: 11 test ve 101 alt test; VibeVoice: 16 test geçti.
Bark public generate düzeltmesi: 13 test geçti; repairs-en-05 içinde 24 ses puanlandı.
ConversationTTS: sabit arşivden 187 tensörün adı ve şekli doğrulanarak Safetensors hazırlandı.
repairs-en-06: ConversationTTS WER %17,62, CER %15,56; sayı metninin ASR çıktısı dinleme incelemesi istiyor.
F5-TTS 24 örnekte WER %3,97, CER %3,42 ile tamamlandı.
HiggsTTS 24 örnekte WER %4,84, CER %3,60 ile tamamlandı.
Dia: dolgu düzeltmesi, KV cache ve public generate sınırı için 12 test geçti.
Dia RTX 3090 karşılaştırması geçti: en büyük logit farkı 0,000012875; greedy tokenlar aynı.
repairs-en-07 içindeki ilk Dia ses denemesinde use_cache API kontrolü hata verdi; bu kontrol düzeltildi.
Fish codec: resmî dosyadaki altı sabit hesaplama tablosu ayrıldı; 535 ağırlık tensörü doğrulandı.
Fish için 15 test ve 6 alt test geçti. repairs-en-08, Llasa sonrası Dia ve Fish seslerini sınar.
Echo indirmesi tamamlandı ancak codec mimarisi ağırlıklarla uyuşmuyor; sorun açık.
Echo indirmeleri artık değişebilir dalı commit ile sabitleyerek yeniden sürdürülebilir istemciyi kullanır.
İndirme önbelleği, doğrulanmış sabit dosyaları aynı diskte hardlink ile paylaşır.
SpeechT5 gerçek tokenizer’ı, 13 örnekte SentencePiece referansıyla eşleşti.

İlk kısa tarama `all-models-en` eski ayarlarla iki metin kullandı. Düzeltilmiş ayarlarla
`english-extended` tüm kaydı, sekiz metin ve üç seed ile tekrar değerlendirir.
Kısa taramadaki zaman aşımı indirme süresini de içerir; kalite puanı değildir.

`completed`, üretim ve puanlamanın bitmesini belirtir; kalite garantisi değildir.
CosyVoice repairs-en-02: WER %17,62, iki 40,96 saniyelik eksik transkriptli çıktı.
repairs-en-04: CosyVoice WER %10,71, üretim sınırına ulaşan örnek 0/24.
VibeVoice WER %4,66, CER %3,97, medyan ilk ses süresi 104 ms; 24 örnek puanlandı.
OuteTTS 24 örnekte WER %3,97 ile tamamlandı.
Bark 24 örnekte WER %6,56, CER %4,48 ile tamamlandı.

CSM 24 örnekte WER %5,35, CER %3,94 ile tamamlandı.

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
| vibevoice | blocked | 0 / 24 | EntryNotFoundError: 404 Client Error. (Request ID: Root=1-6aa46c9f-5d6576094095f03b6b6bb01d;4c9825bd-aaf7-431d-80ef-99eb04a352f4)  Entry Not Found for url: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/6bce5f06044837fe6d2c5d7a71a84 |
| outetts | completed | 24 / 24 |  |
| bark | blocked | 0 / 24 | TypeError: BarkProcessor.__call__() takes 1 positional argument but 2 were given |
| chatterbox | completed | 24 / 24 |  |
| conversationtts | blocked | 0 / 24 | UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, [1mdo those steps only if you trust the source of the checkpoint[0m.  	(1) In PyTorch 2.6, we changed the default value of the `weights_only` a |
| csm | completed | 24 / 24 |  |
| dia | blocked | 0 / 24 | ValueError: Dia output contains a special token inside decoded DAC frames. |
| echo | blocked | 0 / 24 | TimeoutError: The read operation timed out |
| f5tts | completed | 24 / 24 |  |
| fishtts | blocked | 0 / 24 | PermissionError: Fish Audio publishes S2-Pro's ModifiedDAC only as `codec.pth`. VoiceHub never loads that pickle during steady-state inference. Either pass `codec_name_or_path` pointing to an audited Safetensors conversion, call `convert_legacy_fish_ |
| higgstts | completed | 24 / 24 |  |
| irodoritts | unsupported_language | 0 / 24 | The pinned VoiceHub model card does not advertise English: ja |
| llasa | generating | 21 / 24 |  |
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

## Ayrı doğrulama: repairs-en-02

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| openvoice | completed | 24 | 4.84% | 3.45% |
| cosyvoice | completed | 24 | 17.62% | 13.35% |

## Ayrı doğrulama: repairs-en-03

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| vibevoice | blocked | 0 | — | — |

## Ayrı doğrulama: repairs-en-04

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| cosyvoice | completed | 24 | 10.71% | 7.33% |
| vibevoice | completed | 24 | 4.66% | 3.97% |

## Ayrı doğrulama: repairs-en-05

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| bark | completed | 24 | 6.56% | 4.48% |

## Ayrı doğrulama: repairs-en-06

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| conversationtts | completed | 24 | 17.62% | 15.56% |

## Ayrı doğrulama: repairs-en-07

| Model | Durum | Puanlanan | WER | CER |
|---|---|---:|---:|---:|
| dia | blocked | 0 | — | — |
| echo | blocked | 0 | — | — |
| fishtts | blocked | 0 | — | — |

## Tamamlanan 24 örneklik son doğrulamalar

Farklı koşuların en yeni tamamlanan ayarları gösterilir; ayarlar ve süre kapsamı README içindedir.

| Model | Koşu | WER | CER | RTF |
|---|---|---:|---:|---:|
| bark | repairs-en-05 | 6.56% | 4.48% | 2.095 |
| chatterbox | english-extended | 3.45% | 3.21% | 1.216 |
| conversationtts | repairs-en-06 | 17.62% | 15.56% | 4.178 |
| cosyvoice | repairs-en-04 | 10.71% | 7.33% | 1.096 |
| csm | english-extended | 5.35% | 3.94% | 3.539 |
| f5tts | english-extended | 3.97% | 3.42% | 0.613 |
| gptsovits | english-extended | 5.87% | 3.63% | 0.416 |
| higgstts | english-extended | 4.84% | 3.60% | 1.685 |
| inflecttts | repairs-en-01 | 4.15% | 3.36% | 0.015 |
| kokoro | repairs-en-01 | 3.63% | 3.27% | 0.018 |
| melotts | english-extended | 4.66% | 3.60% | 0.019 |
| openvoice | repairs-en-02 | 4.84% | 3.45% | 0.039 |
| outetts | english-extended | 3.97% | 3.36% | 5.269 |
| speecht5 | repairs-en-01 | 3.11% | 2.91% | 0.268 |
| styletts2 | repairs-en-01 | 3.80% | 3.30% | 0.040 |
| supertonic | english-extended | 3.80% | 3.33% | 0.061 |
| vibevoice | repairs-en-04 | 4.66% | 3.97% | 1.015 |
| vits | english-extended | 7.77% | 4.75% | 0.030 |
| vui | english-extended | 14.51% | 12.93% | 0.640 |

İşler Supervisor altında seri GPU kullanımıyla devam eder. Bu görevde 30 dakikalık
kontrol ve düzeltme takibi etkindir; değişmeyen durumlarda bildirim göndermez.
Ölçüm protokolü, hazırlama komutları ve yeniden üretme adımları README.md içindedir.
