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
    'Uygulama: 17 test geçti. VoiceHub düzeltmeleri: 48 test ve 6 alt test geçti.',
    'SpeechT5 gerçek tokenizer’ı, 13 örnekte SentencePiece referansıyla eşleşti.', '',
    'İlk kısa tarama `all-models-en` eski ayarlarla iki metin kullandı. Düzeltilmiş ayarlarla',
    '`english-extended` tüm kaydı, sekiz metin ve üç seed ile tekrar değerlendirir.',
    'Kısa taramadaki zaman aşımı indirme süresini de içerir; kalite puanı değildir.', '',
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
lines += ['', 'İşler Supervisor altında seri GPU kullanımıyla devam eder. Bu görevde 30 dakikalık',
    'kontrol ve düzeltme takibi etkindir; değişmeyen durumlarda bildirim göndermez.',
    'Ölçüm protokolü, hazırlama komutları ve yeniden üretme adımları README.md içindedir.', '']
(root/'STATUS_TR.md').write_text('\n'.join(lines))
