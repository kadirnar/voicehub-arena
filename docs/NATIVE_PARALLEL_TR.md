# Yeni GPU’da VRAM’e göre paralel değerlendirme

16 Eylül 2026: önceki A100 çalışması kullanıcı isteğiyle durduruldu ve periyodik
başlatma/takip otomasyonu silindi. Tamamlanan sonuçlar ve kısmi sesler korundu.
Bu değişiklik yeni bir benchmark başlatmaz. Ortam kurulumu da artık kendiliğinden
Kokoro pilotu çalıştırmaz.

## Çalışma şekli

- Her GPU için canlı `nvidia-smi` boş bellek ve süreç belleği okunur. Varsayılan üst
  sınır GPU başına **4 iş**; yeterli VRAM yoksa daha az iş kabul edilir.
- Belleği henüz ölçülmeyen model/yöntem veya metrik o GPU’da önce tek başına
  çalıştırılır. Pilot ölçümü tam değerlendirmede kullanılır. Tahmin, ölçülen tepe
  süreç/ayırıcı belleğinin **1,35 katı + 1.024 MiB** değeridir. Ayrıca en az
  **2.048 MiB veya GPU kapasitesinin %10’u** boşluk olarak ayrılır.
- Henüz ağırlıklarını yüklememiş çalışan işlerin bellek rezervasyonu da hesaba
  katılır. Ayrılmış bellek, canlı kullanım üzerinden ikinci kez sayılmaz. Başka
  uygulamaların kullandığı VRAM kullanılabilir kapasiteden zaten düşülür.
- Bu bir bellek tahminidir, CUDA için kesin bir kota değildir. Daha uzun bir
  örnekte bellek taşarsa, kalan iş **bir kez tek başına** denenir. Başarılı sesler
  tekrar üretilmez; aynı seed ve resmî üretim parametreleri korunur. Tek başına da
  sığmayan iş hata olarak kalır; sessizce quantization veya başka model kullanılmaz.
- DNSMOS ve Supertonic’in resmî CPU üretimi ayrı CPU kuyruğundadır. Varsayılan
  2 CPU işi, iş başına 4 hesaplama thread’i kullanılır. CPU işleri CUDA açmaz.
- Üretim ve farklı modellerin ASR/MOS/SIM adımları belleğe sığdıkları ölçüde
  örtüşebilir. Bir deneyin aynı sonuç dosyasına iki iş aynı anda yazamaz.
- Tam değerlendirme yalnız o yöntemin pilotu ve uygulanabilir tüm metrikleri
  tamamlandıktan sonra başlar. SIM kimlik kontrolleri yeni kampanya için yeniden
  yapılır. Destek/API incelemesi bitmemiş adaylar başarı diye gösterilmez.
- Kurulumlar, büyük checkpoint indirmeleri ve HF yayımları kendi kuyruk/kilitleriyle
  korunur. GPU telemetrisi bozulursa bellek kontrolsüz yeni işler açılmaz.
- Birden fazla GPU varsa `--gpus auto` görünür GPU’ları kullanır. `--gpus 0,1` veya
  GPU UUID’leri seçilebilir; `CUDA_VISIBLE_DEVICES` kısıtlaması korunur. MIG bu
  sürümün desteklediği bellek muhasebesi değildir; belirsiz cihazlar reddedilir.

## Yeni makinede hazırlık

Linux, NVIDIA GPU, çalışan `nvidia-smi` ve yeterli checkpoint/ses diski gerekir.
Repository README’sindeki `scripts/bootstrap.sh` ile temel ortamı hazırlayın;
kimlik bilgileri repo içine yazılmadan `hf auth login` kullanın. Ardından:

```bash
cd /workspace/voicehub-arena
source scripts/env.sh
.venv/bin/python -m gdown 'https://drive.google.com/uc?id=1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP' -O /workspace/seedtts_testset.tar
.venv/bin/python scripts/prepare_native_seed.py --archive /workspace/seedtts_testset.tar
bash scripts/setup_native_initial.sh
```

Referans çıkarma komutu yayımlanmış arşivin SHA256 değerini doğrular; diğer
benchmark veri setlerini indirmez. Model ortamları çalışma kuyruğunda gerektiğinde
kurulur. İlk kurulum hâlâ legacy MOS/SIM ortamında Torch 1.13.1/CUDA 11.7 kullanır;
yeni GPU mimarisinin bu ortamla uyumluluğu ayrıca doğrulanmalıdır. Paralel kuyruk
tek başına model veya CUDA sürümü uyumluluğunu çözmez.

Önce yalnız planı görün; bu komut model yüklemez veya benchmark başlatmaz:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_native_campaign.py \
  --campaign native-new-gpu-20260916 --gpus auto \
  --max-gpu-jobs 4 --max-cpu-jobs 2 --dry-run
```

Yeni GPU’da çalıştırmaya hazır olduğunuzda `--dry-run` kaldırılır. Doğrulanmış
sonuçları mevcut HF Space/veri kümesine kaydetmek ve yayımlanmış seslerin yerel
kopyalarını kontrollü boşaltmak için `--publish` eklenir:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_native_campaign.py \
  --campaign native-new-gpu-20260916 --gpus auto \
  --max-gpu-jobs 4 --max-cpu-jobs 2 --publish
```

Uzun işler için `deploy/voicehub-native-parallel.conf.example` dosyasındaki yolları
ve kampanya adını yeni makineye uyarlayıp Supervisor’a ekleyin. Şablonun
`autostart=false` ayarı kurulum sırasında değerlendirme başlamasını engeller.
`supervisorctl start voicehub-native-parallel` açıkça başlatır;
`supervisorctl stop voicehub-native-parallel` çalışan alt süreç gruplarını da durdurur.
Ctrl+C/SIGTERM aynı davranışı sağlar. SIGUSR1 yeni iş kabulünü keser, etkin işleri
bitirip durur. Başarısız veya durdurulmuş bir kampanya aynı komutla eksik satırlardan
devam eder. Yeni GPU, farklı paralellik ayarı veya `--mode isolated` için ayrı bir
`--campaign` kullanın; eski donanım süreleri yeni sonuçlarla karıştırılmaz.

## Sonuçların yorumu ve doğrulama

`runs/<kampanya>/campaign-status.json` bütün etkin işleri, GPU atamalarını ve
rezervasyonları içerir. `gpu-profiles.json` ölçülen tepe belleği, `execution-config.json`
donanım/paralellik ayarını, `manifest.json` ise o kampanyanın sabit kapsamını tutar.
Yeni sonuçlar `timing_scope=parallel_throughput` ile etiketlenir. WER/CER/MOS/SIM
parametreleri korunur; eşzamanlı çalışırken ölçülen gecikme ve RTF, GPU’nun tek bir
isteğe ayrıldığı gecikmeyle aynı ölçüm değildir. Ayrı hız karşılaştırması için
farklı kampanyada `--mode isolated` kullanın.

Testler gerçek hafif alt süreçlerle eşzamanlılık ve bağımlılıkları, simüle VRAM
telemetrisiyle rezervasyonları, OOM sonrası tekil yeniden denemeyi, CPU’nun CUDA
kullanmamasını ve süreç gruplarının durdurulmasını doğrular. Bu değişiklik kapsamında
eski A100’de yeni TTS inference veya yeni GPU’da gerçek yük testi başlatılmadı.
