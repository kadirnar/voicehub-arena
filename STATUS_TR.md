# VoiceHub Arena — durum

Yerel sonuç kopyasının rapor zamanı: 2026-09-12T09:31:39+00:00.
Canlı arayüz: http://127.0.0.1:7860/

RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.
Bu rapor kaydedilmiş sonuçlardan üretilir; canlı iş ilerledikçe arayüz daha güncel olabilir.

## Yayımlanmış veri setleri ve Whisper large-v3

Kampanya durumu: **running**. Son etkin iş: `pub-v2-chatterbox-seedtts_en-000`.
33 İngilizce model; model başına 1,088 tam bölüm metni.
Tamamlanan model/bölüm parçaları: 9 / 198; eksik veya hatalı: 0.
Parça tamamlanması tam veri setinin veya bütün kampanyanın tamamlandığı anlamına gelmez.

| Veri seti | Tam bölüm | Üretilen ses | Puanlanan ses |
|---|---:|---:|---:|
| seedtts_en | 1,088 | 288 | 288 |

Etkin kapsam: **v1 · Seed-TTS-Eval English**.
Her model için önce 32 metin/bölüm pilotu, sonra 256 metin/bölüm paneli, ardından kalan bütün metinler çalışır.
Panel: 256 metin/model; tam kapsam: 1,088 metin/model, toplam 35,904 ses. Üretim seed42 ve tek tekrarlıdır.
İlk sürüm yalnız Seed-TTS-Eval İngilizce kapsamını çalıştırır. Diğer veri setleri ertelenmiştir; önceki dosyalar korunur ve otomatik olarak yeniden başlatılmaz.
ASR: `Systran/faster-whisper-large-v3` @ `edaa852ec7e145841d8ffdb056a99866b5f0a478`; CUDA FP16.
Bu tabloda ses sayıları modellerin toplamıdır; kaynak metin sayısıyla karıştırılmamalıdır.
WER/CER ve güven aralıkları her veri seti ve kapsam için ayrı gösterilir. Eksik kapsam sıralamaya uygun değildir.
Bu bir sabit ses ile anlaşılabilirlik testidir; Seed ses klonlama SIM veya Emergent duygu/doğallık değerlendirmesinin tekrarı değildir.
LibriTTS ve LibriSpeech ortak kaynak içerir; sonuçlar tek bağımsız veri havuzu olarak birleştirilmez.
LibriSpeech girişleri tüm modeller için küçük harfe dönüştürülür (librispeech_lowercase_v1); kaynak metin ve referanslar korunur. Eski büyük harfli denemeler ayrı arşivdir.
StyleTTS2 NLTK tırnak hazırlama düzeltmesi, özgün tokenlarla doğrulanır; eski başarısız deneme korunur ve yeni sf1 koşuları ayrı izlenir.
Yöntem ve kaynaklar: [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

## Geçmiş tanı doğrulamaları

33 İngilizce modelin sekiz metin × üç seed tanı üretimi ve puanlaması en az bir kez tamamlandı.
Her modelin son tamamlanan tanı kaydı aşağıdadır. Bunlar yeni yayımlanmış-veri benchmark sonuçları değildir.

| Model | Tanı koşusu | ASR | WER | CER |
|---|---|---|---:|---:|
| bark | repairs-en-05 | faster-whisper-small.en | 6.56% | 4.48% |
| chatterbox | english-extended | faster-whisper-small.en | 3.45% | 3.21% |
| conversationtts | repairs-en-06 | faster-whisper-small.en | 17.62% | 15.56% |
| cosyvoice | repairs-en-04 | faster-whisper-small.en | 10.71% | 7.33% |
| csm | english-extended | faster-whisper-small.en | 5.35% | 3.94% |
| dia | repairs-en-08 | faster-whisper-small.en | 11.74% | 9.51% |
| echo | repairs-en-10 | faster-whisper-small.en | 3.97% | 3.30% |
| f5tts | english-extended | faster-whisper-small.en | 3.97% | 3.42% |
| fishtts | repairs-en-08 | faster-whisper-small.en | 3.80% | 3.27% |
| gptsovits | english-extended | faster-whisper-small.en | 5.87% | 3.63% |
| higgstts | english-extended | faster-whisper-small.en | 4.84% | 3.60% |
| inflecttts | repairs-en-01 | faster-whisper-small.en | 4.15% | 3.36% |
| kokoro | repairs-en-01 | faster-whisper-small.en | 3.63% | 3.27% |
| llasa | english-extended | faster-whisper-small.en | 19.00% | 14.17% |
| melotts | english-extended | faster-whisper-small.en | 4.66% | 3.60% |
| mosstts | repairs-en-12 | faster-whisper-small.en | 3.80% | 3.30% |
| neutts | repairs-en-13 | faster-whisper-small.en | 7.60% | 5.39% |
| omnivoice | repairs-en-11 | faster-whisper-small.en | 3.97% | 3.39% |
| openvoice | repairs-en-02 | faster-whisper-small.en | 4.84% | 3.45% |
| orpheustts | repairs-en-11 | faster-whisper-small.en | 7.08% | 6.87% |
| outetts | english-extended | faster-whisper-small.en | 3.97% | 3.36% |
| parlertts | repairs-en-12 | faster-whisper-small.en | 4.66% | 3.15% |
| qwen3tts | repairs-en-12 | faster-whisper-small.en | 4.32% | 3.51% |
| speecht5 | repairs-en-01 | faster-whisper-small.en | 3.11% | 2.91% |
| styletts2 | repairs-en-01 | faster-whisper-small.en | 3.80% | 3.30% |
| supertonic | english-extended | faster-whisper-small.en | 3.80% | 3.33% |
| vibevoice | repairs-en-04 | faster-whisper-small.en | 4.66% | 3.97% |
| vits | english-extended | faster-whisper-small.en | 7.77% | 4.75% |
| voxcpm | repairs-en-12 | faster-whisper-small.en | 3.63% | 3.27% |
| vui | english-extended | faster-whisper-small.en | 14.51% | 12.93% |
| xtts | repairs-en-12 | faster-whisper-small.en | 3.80% | 3.33% |
| zonos | repairs-en-12 | faster-whisper-small.en | 4.49% | 3.57% |
| zonos2 | repairs-en-11 | faster-whisper-small.en | 12.95% | 11.90% |

NeuCodec erişimi doğrulanmış ve NeuTTS 24/24 tanı örneğini tamamlamıştır.
repairs-en-13 içindeki Zonos2 tanı tekrarı, henüz ölçüm üretmeden yayımlanmış veri setlerine geçiş nedeniyle superseded olarak kapatıldı.
Önceki tamamlanmış Zonos2 sonucu korunur; yeni kampanya Zonos2 modelini de içerir.
Irodori-TTS Japonca ilan ettiği için İngilizce kapsamı dışındadır.

## İşletim

Supervisor hizmeti: `voicehub-arena-public-suite`; plan: `runs/public-english-v2/suite.json`.
Aynı GPU kilidiyle seri sentez ve ASR; model dosyaları için 32 GiB, ses yazımı için 4 GiB alan tabanı.
Alan biterse sonuçlar silinmeden waiting_for_storage kaydı oluşur. Tam koşu devam eden uzun süreli bir kampanyadır.
30 dakikalık takip, hata düzeltme ve sonuç kopyalama otomasyonu etkin. Değişmeyen durumda bildirim verilmez.
Kimlik bilgileri proje dışında korunur.
