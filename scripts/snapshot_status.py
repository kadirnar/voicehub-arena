"""Write a current Turkish status snapshot from durable manifests only."""
import datetime
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root/'runs'
read = lambda path: json.loads(path.read_text())
now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
lines = ['# VoiceHub Arena — durum', '', f'Yerel sonuç kopyasının rapor zamanı: {now}.',
         'Canlı arayüz: http://127.0.0.1:7860/', '',
         'RTX 3090 (24 GB), `vast-3090-voicehub` SSH bağlantısı ve ayrı VS Code uzak penceresi hazır.',
         'Bu rapor kaydedilmiş sonuçlardan üretilir; canlı iş ilerledikçe arayüz daha güncel olabilir.', '']
plan_path = runs/'public-english-v2/suite.json'
if plan_path.exists():
    plan = read(plan_path)
    counts = Counter(job['status'] for job in plan['jobs'])
    total = sum(dataset['total_samples'] for dataset in plan['datasets'])
    lines += ['## Yayımlanmış veri setleri ve Whisper large-v3', '',
              f"Kampanya durumu: **{plan['status']}**. Son etkin iş: `{plan.get('current_job', '—')}`.",
              f"{len(plan['catalog'])} İngilizce model; model başına {total:,} tam bölüm metni.",
              'Seçili modeller: ' + ', '.join(s.get('name', s['model_type']) for s in plan['catalog']) + '.',
              f"Tamamlanan model/bölüm parçaları: {counts['completed']} / {len(plan['jobs'])}; eksik veya hatalı: {counts['partial']+counts['failed']}.",
              'Parça tamamlanması tam veri setinin veya bütün kampanyanın tamamlandığı anlamına gelmez.', '',
              '| Veri seti | Tam bölüm | Üretilen ses | Puanlanan ses |', '|---|---:|---:|---:|']
    for dataset in plan['datasets']:
        generated = scored = 0
        for job in plan['jobs']:
            if job['dataset'] != dataset['id']:
                continue
            path = runs/job['run']/job['model']/'result.json'
            if path.exists():
                rows = read(path).get('rows', [])
                generated += sum(bool(row.get('audio')) for row in rows)
                scored += sum(row['status'] == 'ok' for row in rows)
        lines.append(f"| {dataset['id']} | {dataset['total_samples']:,} | {generated} | {scored} |")
    panel = sum(s['samples'] for d in plan['datasets'] for s in d['shards'] if s['phase'] in ('pilot','panel'))
    lines += ['', f"Etkin kapsam: **{plan.get('scope',{}).get('label','Public English')}**.",
              'Her model için önce 32 metin/bölüm pilotu, sonra 256 metin/bölüm paneli, ardından kalan bütün metinler çalışır.',
              f'Panel: {panel:,} metin/model; tam kapsam: {total:,} metin/model, toplam {total*len(plan["catalog"]):,} ses. Üretim seed42 ve tek tekrarlıdır.',
              'İlk sürüm yalnız seçili modellerde Seed-TTS-Eval İngilizce kapsamını çalıştırır. Diğer modeller ve veri setleri ertelenmiştir; önceki dosyalar korunur ve otomatik olarak yeniden başlatılmaz.',
              f"ASR: `{plan['asr']['checkpoint']}` @ `{plan['asr']['revision']}`; CUDA FP16.",
              'Bu tabloda ses sayıları modellerin toplamıdır; kaynak metin sayısıyla karıştırılmamalıdır.',
              'WER/CER ve güven aralıkları her veri seti ve kapsam için ayrı gösterilir. Eksik kapsam sıralamaya uygun değildir.',
              'Bu bir sabit ses ile anlaşılabilirlik testidir; Seed ses klonlama SIM veya Emergent duygu/doğallık değerlendirmesinin tekrarı değildir.',
              'LibriTTS ve LibriSpeech ortak kaynak içerir; sonuçlar tek bağımsız veri havuzu olarak birleştirilmez.',
              'LibriSpeech girişleri tüm modeller için küçük harfe dönüştürülür (librispeech_lowercase_v1); kaynak metin ve referanslar korunur. Eski büyük harfli denemeler ayrı arşivdir.',
              'StyleTTS2 NLTK tırnak hazırlama düzeltmesi, özgün tokenlarla doğrulanır; eski başarısız deneme korunur ve yeni sf1 koşuları ayrı izlenir.',
              'Yöntem ve kaynaklar: [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).', '']
latest = {}
for config_path in runs.glob('*/config.json'):
    cfg = read(config_path)
    if len(cfg.get('dataset', [])) != 8 or cfg.get('repeats') != 3:
        continue
    for spec in cfg['catalog']:
        path = config_path.parent/spec['model_type']/'result.json'
        if not path.exists():
            continue
        result = read(path)
        if result.get('status') != 'completed' or result.get('summary', {}).get('scored') != 24:
            continue
        stamp = cfg.get('started_at', 0)
        if stamp > latest.get(spec['model_type'], (-1,))[0]:
            latest[spec['model_type']] = (stamp, config_path.parent.name, result, cfg['asr']['checkpoint'])
lines += ['## Geçmiş tanı doğrulamaları', '',
          f'{len(latest)} İngilizce modelin sekiz metin × üç seed tanı üretimi ve puanlaması en az bir kez tamamlandı.',
          'Her modelin son tamamlanan tanı kaydı aşağıdadır. Bunlar yeni yayımlanmış-veri benchmark sonuçları değildir.', '',
          '| Model | Tanı koşusu | ASR | WER | CER |', '|---|---|---|---:|---:|']
for name, (_, run, result, asr) in sorted(latest.items()):
    summary = result['summary']
    lines.append(f"| {name} | {run} | {asr.split('/')[-1]} | {summary['wer']*100:.2f}% | {summary['cer']*100:.2f}% |")
lines += ['', 'NeuCodec erişimi doğrulanmış ve NeuTTS 24/24 tanı örneğini tamamlamıştır.',
          'repairs-en-13 içindeki Zonos2 tanı tekrarı, henüz ölçüm üretmeden yayımlanmış veri setlerine geçiş nedeniyle superseded olarak kapatıldı.',
          'Önceki tamamlanmış Zonos2 sonucu korunur; yeni kampanya Zonos2 modelini de içerir.',
          'Irodori-TTS Japonca ilan ettiği için İngilizce kapsamı dışındadır.', '',
          '## İşletim', '', 'Supervisor hizmeti: `voicehub-arena-public-suite`; plan: `runs/public-english-v2/suite.json`.',
          'Aynı GPU kilidiyle seri sentez ve ASR; model dosyaları için 32 GiB, ses yazımı için 4 GiB alan tabanı.',
          'Alan biterse sonuçlar silinmeden waiting_for_storage kaydı oluşur. Tam koşu devam eden uzun süreli bir kampanyadır.',
          '30 dakikalık takip, hata düzeltme ve sonuç kopyalama otomasyonu etkin. Değişmeyen durumda bildirim verilmez.',
          'Kimlik bilgileri proje dışında korunur.', '']
(root/'STATUS_TR.md').write_text('\n'.join(lines))
