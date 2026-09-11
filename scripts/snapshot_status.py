"""Refresh the Turkish delivery note from saved run manifests."""
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root/'runs/english-extended'
if not (run/'config.json').exists():
    run = root/'runs/all-models-en'
cfg = json.loads((run/'config.json').read_text())
state = json.loads((run/'state.json').read_text())
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
lines = [
    '# VoiceHub Arena — durum', '',
    f'Anlık kopya: {now}. Canlı arayüz: http://127.0.0.1:7860/', '',
    'RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.',
    '`/workspace/voicehub-arena` sunucudaki proje; bu klasör yerel kaynak ve sonuç kopyasıdır.', '',
    '**Tüm modellerin çalıştığı henüz doğrulanmadı.** İngilizce dışı Irodori-TTS kapsam dışıdır.',
    f'Gösterilen koşu: `{run.name}`; durum `{state.get("status")}`, aşama `{state.get("phase")}`.',
    f'Kapsam: {len(cfg["catalog"])} model ailesi, {len(cfg["dataset"])} İngilizce metin × {cfg["repeats"]} seed.', '',
    'NeuTTS-2e ana ağırlıklarına erişim doğrulandı; NeuCodec bağımlılığı ayrıca yetki istiyor.',
    'Kimlik bilgileri proje dışında saklanıyor.',
    '33 İngilizce modelin giriş sözleşmesi kontrolü geçti. Bu, GPU üretim başarısı anlamına gelmez.',
    'Uygulama: 29 test geçti. İlk VoiceHub düzeltmeleri: 48 test ve 6 alt test geçti.',
    'Ek OpenVoice yükleme düzeltmesi: 12 test geçti.',
    'CosyVoice: 11 test ve 101 alt test; VibeVoice: 16 test geçti.',
    'Bark public generate düzeltmesi: 13 test geçti; repairs-en-05 içinde 24 ses puanlandı.',
    'ConversationTTS: sabit arşivden 187 tensörün adı ve şekli doğrulanarak Safetensors hazırlandı.',
    'repairs-en-06: ConversationTTS WER %17,62, CER %15,56; sayı metninin ASR çıktısı dinleme incelemesi istiyor.',
    'F5-TTS 24 örnekte WER %3,97, CER %3,42 ile tamamlandı.',
    'HiggsTTS 24 örnekte WER %4,84, CER %3,60 ile tamamlandı.',
    'Dia: dolgu düzeltmesi, KV cache ve public generate sınırı için 12 test geçti.',
    'Dia RTX 3090 karşılaştırması geçti: en büyük logit farkı 0,000012875; greedy tokenlar aynı.',
    'repairs-en-07 içindeki ilk Dia ses denemesinde use_cache API kontrolü hata verdi; bu kontrol düzeltildi.',
    'Fish codec: resmî dosyadaki altı sabit hesaplama tablosu ayrıldı; 535 ağırlık tensörü doğrulandı.',
    'Fish için 15 test ve 6 alt test geçti. repairs-en-08, Llasa sonrası Dia ve Fish seslerini sınar.',
    'Echo indirmesi tamamlandı ancak codec mimarisi ağırlıklarla uyuşmuyor; sorun açık.',
    'Echo indirmeleri artık değişebilir dalı commit ile sabitleyerek yeniden sürdürülebilir istemciyi kullanır.',
    'İndirme önbelleği, doğrulanmış sabit dosyaları aynı diskte hardlink ile paylaşır.',
    'SpeechT5 gerçek tokenizer’ı, 13 örnekte SentencePiece referansıyla eşleşti.', '',
    'İlk kısa tarama `all-models-en` eski ayarlarla iki metin kullandı. Düzeltilmiş ayarlarla',
    '`english-extended` tüm kaydı, sekiz metin ve üç seed ile tekrar değerlendirir.',
    'Kısa taramadaki zaman aşımı indirme süresini de içerir; kalite puanı değildir.', '',
    '`completed`, üretim ve puanlamanın bitmesini belirtir; kalite garantisi değildir.',
    'CosyVoice repairs-en-02: WER %17,62, iki 40,96 saniyelik eksik transkriptli çıktı.',
    'repairs-en-04: CosyVoice WER %10,71, üretim sınırına ulaşan örnek 0/24.',
    'VibeVoice WER %4,66, CER %3,97, medyan ilk ses süresi 104 ms; 24 örnek puanlandı.',
    'OuteTTS 24 örnekte WER %3,97 ile tamamlandı.',
    'Bark 24 örnekte WER %6,56, CER %4,48 ile tamamlandı.', '',
    'CSM 24 örnekte WER %5,35, CER %3,94 ile tamamlandı.', '',
    'WER, CER, MER, WIL/WIP, tam eşleşme, sözcük hataları, RTF, p50/p95 süre, GPU bellek',
    'tepesi, RMS/peak dBFS, clipping, sessizlik ve DC offset gerçek çıktılardan hesaplanır.',
    'VibeVoice için yeni Arena adaptörü ayrıca etiketlenir; tam referans dalga biçimi eşliği henüz doğrulanmadı.', '',
    '| Model | Durum | Ses / planlanan | Açıklama |', '|---|---|---:|---|',
]
for spec in cfg['catalog']:
    name = spec['model_type']
    path = run/name/'result.json'
    result = json.loads(path.read_text()) if path.exists() else {'status':'pending','rows':[]}
    audio = sum(bool(row.get('audio')) for row in result['rows'])
    error = result.get('error','').replace('|','/').replace('\n',' ')[:250]
    lines.append(f'| {name} | {result["status"]} | {audio} / {len(cfg["dataset"])*cfg["repeats"]} | {error} |')
for attempt in sorted((root/'runs').glob('repairs-*')):
    if not (attempt/'config.json').exists():
        continue
    lines += ['',f'## Ayrı doğrulama: {attempt.name}', '',
              '| Model | Durum | Puanlanan | WER | CER |', '|---|---|---:|---:|---:|']
    attempt_config = json.loads((attempt/'config.json').read_text())
    for spec in attempt_config['catalog']:
        path = attempt/spec['model_type']/'result.json'
        result = json.loads(path.read_text()) if path.exists() else {'status':'pending'}
        scores = result.get('summary',{})
        pct = lambda x: f'{x*100:.2f}%' if isinstance(x,(int,float)) else '—'
        lines.append(f'| {spec["model_type"]} | {result["status"]} | {scores.get("scored",0)} | {pct(scores.get("wer"))} | {pct(scores.get("cer"))} |')
latest = {}
for candidate in sorted((root/'runs').glob('*'), key=lambda p: p.stat().st_mtime):
    config_path = candidate/'config.json'
    if not config_path.exists():
        continue
    settings = json.loads(config_path.read_text())
    if len(settings.get('dataset', [])) != 8 or settings.get('repeats') != 3:
        continue
    for result_path in candidate.glob('*/result.json'):
        result = json.loads(result_path.read_text())
        if result.get('status') == 'completed' and result.get('summary', {}).get('scored') == 24:
            name = result_path.parent.name
            stamp = config_path.stat().st_mtime
            if name not in latest or stamp > latest[name][0]:
                latest[name] = (stamp, candidate.name, result)
lines += ['', '## Tamamlanan 24 örneklik son doğrulamalar', '',
          'Farklı koşuların en yeni tamamlanan ayarları gösterilir; ayarlar ve süre kapsamı README içindedir.', '',
          '| Model | Koşu | WER | CER | RTF |', '|---|---|---:|---:|---:|']
for name, (_, attempt, result) in sorted(latest.items()):
    scores = result['summary']
    lines.append(f'| {name} | {attempt} | {scores["wer"]*100:.2f}% | {scores["cer"]*100:.2f}% | {scores["rtf"]:.3f} |')
lines += ['', 'İşler Supervisor altında seri GPU kullanımıyla devam eder. Bu görevde 30 dakikalık',
    'kontrol ve düzeltme takibi etkindir; değişmeyen durumlarda bildirim göndermez.',
    'Ölçüm protokolü, hazırlama komutları ve yeniden üretme adımları README.md içindedir.', '']
(root/'STATUS_TR.md').write_text('\n'.join(lines))
