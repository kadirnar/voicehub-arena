# A100 kurulumu — 12 Eylül 2026

VoiceHub Arena, Vast instance **50735875** üzerinde kuruldu ve İngilizce Seed
benchmarkı başlatıldı. Donanım **NVIDIA A100-SXM4 40 GB**, disk 111 GB,
NVIDIA sürücüsü 570.133.20. Python 3.12 ve PyTorch 2.8.0 + CUDA 12.8 kullanılıyor.

## Bağlantı ve arayüz

Mac üzerindeki SSH alias: `vast-a100-voicehub`.

```bash
ssh vast-a100-voicehub
```

Alias, `root@212.13.234.23:13876` bağlantısını ve istenen
`LocalForward 8080 localhost:8080` ayarını içerir. Diğer projeye ait `vast-a100`
aliası değiştirilmedi. Yerel 8080 zaten başka bir tünelde kullanılıyorsa salt SSH
komutları için `ssh -o ClearAllForwardings=yes vast-a100-voicehub` kullanılabilir.

Sunucudaki repo: `/workspace/voicehub-arena`. VS Code bu klasöre ayrı bir
`SSH: vast-a100-voicehub` penceresiyle bağlandı. Bu instance'ın otomatik tmux
başlangıcına gerçek terminal kontrolü eklendi; VS Code'un ortam çözümlemesi geçti.

Arena arayüzü: **http://127.0.0.1:7860/**. Yerel 7860, özel SSH tüneliyle yeni
instance'ın localhost:7860 portuna bağlanır. Tünel kapandığında:

```bash
ssh -F /dev/null -i ~/.ssh/id_ed25519 -p 13876 \
  -o IdentitiesOnly=yes -o ForwardAgent=no -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -N \
  -L 127.0.0.1:7860:127.0.0.1:7860 root@212.13.234.23
```

## Doğrulanan kurulum

- 161 Python paketi için bağımlılık uyumluluğu kontrolü geçti.
- **49 test geçti; atlanan test yok.** XTTS metin temizleyicisinin eksik paketleri
  kurulum listesine eklendi ve test artık bu eksikliği sessizce atlamıyor.
- StyleTTS2, CosyVoice3, ConversationTTS, VoxCPM2 ve XTTS dönüşümleri sabit
  SHA256 değerleriyle doğrulandı. Diğer model ağırlıkları sıra geldiğinde indirilir.
- Seed-TTS-Eval English: 1.088 kaynak metin ve altı ayrık shard doğrulandı.
- Whisper large-v3, CUDA FP16 ile gerçek Emily referans sesini başarıyla çözdü.
  Model yükleme dahil bu kısa kontrol 4,65 saniye sürdü; bu bir TTS benchmark skoru değildir.
- İlk üç pilotta Kokoro, Supertonic ve VITS için **96/96 üretim ve ASR skoru**
  tamamlandı. Kokoro'nun 32 metinlik pilotunda WER %0,93 ve CER %0,11 ölçüldü.

Bu kayıt kurulum anının durumudur. **33 modelin tam benchmarkı henüz bitmedi.**
Canlı kapsam ve hatalar arayüzden ve `runs/public-english-v2/suite.json` üzerinden
izlenmelidir. İlk sürüm yalnızca İngilizce Seed-TTS-Eval kullanır; 33 × 1.088 =
35.904 ses planlanır. Eski RTX 3090 kayıtları bu sonuçlarla birleştirilmedi.

## Hizmetler ve devam etme

```bash
supervisorctl status voicehub-arena-benchmark voicehub-arena-web
tail -f /var/log/portal/voicehub-arena-benchmark.log
# İhtiyaç olduğunda açıkça durdur / yeniden başlat:
supervisorctl stop voicehub-arena-benchmark voicehub-arena-web
supervisorctl start voicehub-arena-benchmark voicehub-arena-web
```

Hizmetler Supervisor ile yönetilir; instance açılışında otomatik başlamaz.
`runs/public-english-v2/pause.request` varsa kuyruk shard sınırında durur.
Devam etmek için bu işaret kaldırılmalıdır; doğrulanmış kayıtlar yeniden kullanılır.

Bu host için `/workspace/.env` içinde `ARENA_CACHE_BUDGET_GIB=28` ve
`ARENA_MIN_FREE_GIB=32` seçildi. Modeller ve ASR GPU'da sırayla çalışır.
Önbellek sınırı bütün modellerin ağırlıklarını aynı anda diskte tutmayı gerektirmez.
HF kimlik bilgisi korumalı, Git dışında kalan cache dosyasında saklanır.

Otomatik takip 30 dakikada bir yeni A100'ın durumunu denetler. Yerel A100 sonuç
yedeği `outputs/voicehub-arena-a100-results/runs/` altındadır; eski GPU arşivi
ayrı kalır. Instance'ta kalıcı volume yoktur; silme veya recycle öncesinde yeni
sonuçlar dışarı alınmalıdır.

## VITS sayısal düzeltmesi — 12 Eylül 2026

VITS panelindeki tek başarısız metinde FP16 süre spline hesabı negatif
diskriminant üretti. Aynı A100, checkpoint ve seed ile hata yeniden üretildi;
yakalanan girdilerle yüksek hassasiyetli hesap doğrulandı. Runtime yaması yalnız
bu başarısız sayısal dalı yüksek hassasiyetle hesaplayıp özgün dtype'a döndürür.
GPU kontrolünde önceki 255 başarılı ses bit düzeyinde aynı kaldı, eksik örnek
3,328 saniyelik sonlu ses üretti. Checkpoint, metin, seed ve ASR ayarları değişmedi.

`--resume --resume-samples` artık `partial` durumundaki üretimlerde de çalışır.
Sağlam satırlar ve dosyalar korunur. Yeni A100 yedeğindeki
`runs/validations/vits-repair-gpu-parity.json` gerçek GPU doğrulamasıdır;
`runs/pub-v2-vits-seedtts_en-001/runtime-repair.json` onarımın kaynak ve sonuç
karmalarını, korunan satırları ve yeni üretilen örneği kaydeder.
