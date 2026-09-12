import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voicehub_arena.benchmarks import build_jobs, public_report, public_overrides, restrict_public_scope
from voicehub_arena.metrics import errors, summarize
from voicehub_arena.server import create_app
from voicehub_arena.storage import write_json
from voicehub_arena.inputs import prepare_benchmark_text


def test_published_split_is_partitioned_without_duplication(tmp_path):
    path = Path(__file__).parents[1]/'scripts/prepare_public_datasets.py'
    spec = importlib.util.spec_from_file_location('prepare_public', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(id=f'test_{i}', text='Some text ' * (i % 40 + 1), reference='text',
                 category='question' if i % 2 else 'narration') for i in range(600)]
    destination = tmp_path/'datasets/public'
    manifest = module.write_dataset(destination, 'sample', rows, {}, 256, 256)
    assert [(s['phase'], s['samples']) for s in manifest['shards']] == [
        ('pilot',32), ('panel',224), ('expansion',256), ('expansion',88)]
    parts = [json.loads(line) for s in manifest['shards'] for line in (tmp_path/s['path']).read_text().splitlines()]
    assert len(parts) == len({r['id'] for r in parts}) == 600
    assert {r['id'] for r in parts} == {r['id'] for r in rows}
    assert [r['id'] for r in module.balanced_order(rows)] == [r['id'] for r in module.balanced_order(list(reversed(rows)))]


def test_first_release_scope_preserves_seed_results_and_defers_other_jobs():
    root = Path(__file__).parents[1]
    seed = json.loads((root/'datasets/public/seedtts_en/manifest.json').read_text())
    suite = {'datasets':[seed, {'id':'emergenttts', 'shards':[
        {'index':0, 'phase':'pilot', 'samples':32}]}]}
    catalog = [{'model_type':f'model{i}'} for i in range(33)]
    suite['jobs'] = build_jobs(suite, catalog)
    seed_job = next(j for j in suite['jobs'] if j['dataset']=='seedtts_en')
    seed_job.update(status='completed', finished_at=123)
    other_job = next(j for j in suite['jobs'] if j['dataset']!='seedtts_en')
    suite['current_job'] = other_job['run']
    scope = json.loads((root/'configs/public-scope.json').read_text())
    scoped = restrict_public_scope(suite, scope)
    assert len(scoped['datasets']) == 1
    assert scoped['datasets'][0]['total_samples'] * len(catalog) == 35904
    assert len(scoped['jobs']) == 198
    assert next(j for j in scoped['jobs'] if j['run']==seed_job['run']) == seed_job
    assert other_job in scoped['deferred_jobs']
    assert 'current_job' not in scoped
    assert len(suite['datasets']) == 2
    assert restrict_public_scope(scoped, scope) == scoped
    with pytest.raises(ValueError, match='explicit new plan'):
        restrict_public_scope(scoped, {'dataset_ids':['emergenttts']})


def test_new_normalization_and_character_edit_counts():
    score = errors(['She paid forty five dollars.'], ['She paid $45.'], normalization='whisper_english')
    assert score['wer'] == score['cer'] == 0
    assert errors(['cat'], ['cut'])['char_substitutions'] == 1
    assert errors(['cat'], ['cut'])['reference_chars'] == 3
    assert errors(['cat'], ['cut'])['cer'] == pytest.approx(1/3)
    audio = dict(latency_s=1, duration_s=1, peak_vram_mib=0, clipping_ratio=0,
                 silence_ratio=0, rms_dbfs=-10)
    with pytest.raises(ValueError, match='different scoring'):
        summarize([{**audio,'id':'a','status':'ok','normalization_id':mode} for mode in ['orthographic','whisper_english']])


def test_corpus_case_preparation_preserves_published_text_and_other_datasets():
    source = "OCEAN REIGNED SUPREME"
    assert prepare_benchmark_text(source,'librispeech_lowercase_v1') == 'ocean reigned supreme'
    assert source == 'OCEAN REIGNED SUPREME'
    assert prepare_benchmark_text('NASA launched at 9:30.') == 'NASA launched at 9:30.'
    with pytest.raises(ValueError,match='Unknown benchmark'):
        prepare_benchmark_text(source,'unknown')


def test_real_inflect_frontend_does_not_spell_corpus_words_after_case_preparation():
    pytest.importorskip('voicehub')
    from voicehub.models.inflecttts.source.inflect.inflect_nano_v2_frontend import normalize_text
    text = 'OCEAN REIGNED SUPREME'
    normalized = normalize_text(prepare_benchmark_text(text,'librispeech_lowercase_v1'))
    assert normalized == 'ocean reigned supreme'
    assert normalize_text(text) != normalized


def test_styletts_quoted_text_matches_released_cleaner_token_ids():
    pytest.importorskip('voicehub')
    from phonemizer import phonemize
    from nltk import word_tokenize
    from voicehub.architectures.styletts2.frontend import NativeStyleTTS2Frontend
    from voicehub.models.styletts2.source.styletts2.text_utils import TextCleaner
    from voicehub_arena.inputs import prepare_request
    texts = ['His son is "underground" publisher Adam Parfrey.',
             'He was known as "Roaring Bill".', 'Ordinary speech still works.']
    cleaner = TextCleaner()
    frontend = NativeStyleTTS2Frontend()
    for text in texts:
        original = phonemize(text, language='en-us', backend='espeak', strip=True,
                             preserve_punctuation=True, with_stress=True)
        original = ' '.join(word_tokenize(original, preserve_line=True))
        prepared, options = prepare_request('styletts2', text, {})
        if '"' in text:
            assert '``' in original
        else:
            assert prepared == original
        assert frontend.encode_phonemes(prepared, explicit=options['text_is_phonemes']).tolist()[0] == [0, *cleaner(original)]
    with pytest.raises(ValueError, match='outside the released'):
        frontend.encode_phonemes('həloʊ ☃', explicit=True)


def public_fixture(tmp_path):
    runs = tmp_path/'runs'
    source = tmp_path/'datasets/public/seedtts_en'
    source.mkdir(parents=True)
    row = dict(id='seed_a', category='read', text='Hello.', reference='Hello.')
    path = source/'shard-000.jsonl'
    path.write_text(json.dumps(row)+'\n')
    shard = {'path':str(path.relative_to(tmp_path)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'samples':1,'index':0,'phase':'pilot'}
    dataset = {'id':'seedtts_en','shards':[shard], 'total_samples':1}
    catalog = [{'model_type':'a','checkpoint':'a'}, {'model_type':'b','checkpoint':'b'}]
    plan = {'protocol_id':'public-english-v2','asr':{'checkpoint':'large-v3','revision':'pinned'},
            'datasets':[dataset],'catalog':catalog,'normalization_id':'whisper_english','repeats':1}
    plan['jobs'] = build_jobs(plan, catalog)
    write_json(runs/'public-english-v2/suite.json', plan)
    job = plan['jobs'][0]
    write_json(runs/job['run']/'config.json', {'catalog':[catalog[0]],'asr':plan['asr'],
        'dataset_part':shard,'normalization_id':plan['normalization_id'],'protocol':'public-english-v2'})
    write_json(runs/job['run']/'a/result.json', {'model_type':'a','status':'failed',
        'rows':[{**row,'repeat':0,'status':'failed','error':'test failure'}]})
    return runs, plan


def test_public_reports_keep_failed_coverage_and_audio_origin(tmp_path):
    runs, plan = public_fixture(tmp_path)
    data = public_report(runs, 'seedtts_en')
    assert data['results'][0]['summary']['scored'] == 0
    assert data['results'][0]['ranking_eligible'] is False
    assert 'wer' not in data['results'][0]['summary']
    assert data['results'][0]['rows'][0]['source_run'] == plan['jobs'][0]['run']
    assert data['results'][1]['planned_samples'] == 1
    client = TestClient(create_app(runs))
    assert client.get('/api/public-suite/seedtts_en/leaderboard.csv').status_code == 200
    assert client.get('/api/overview').json()['results'] == []
    assert client.get('/api/public-suite/unknown').status_code == 404
    # Mismatched scorers must never enter one comparison table.
    path = runs/plan['jobs'][0]['run']/'config.json'
    cfg = json.loads(path.read_text()); cfg['asr']['revision'] = 'changed'
    write_json(path,cfg)
    assert client.get('/api/public-suite/seedtts_en').status_code == 404
    cfg['asr'] = plan['asr']
    cfg['input_text_transform'] = 'librispeech_lowercase_v1'
    write_json(path,cfg)
    assert client.get('/api/public-suite/seedtts_en').status_code == 404
    cfg['input_text_transform'] = 'identity'
    cfg['frontend_protocols'] = {'a':'changed'}
    write_json(path,cfg)
    assert client.get('/api/public-suite/seedtts_en').status_code == 404


def test_campaign_pins_survive_shard_specific_frontend_preparation(tmp_path):
    frozen = {'a':{'config':{'revision':'frozen'},'generation':{'voice':'fixed'}, 'prepared_inputs':'old'}}
    path = tmp_path/'configs/public-models-lock.json'
    write_json(path, {'overrides':frozen})
    prepared = {'a':{'config':{'revision':'mutable'},'generation':{'voice':'changed'},'prepared_inputs':'new-shard'}}
    result, sha = public_overrides(tmp_path,[{'model_type':'a'}],prepared)
    assert result['a']['config']['revision'] == 'frozen'
    assert result['a']['generation']['voice'] == 'fixed'
    assert result['a']['prepared_inputs'] == 'new-shard'
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text())['overrides'] == frozen


def test_resume_retains_verified_audio_when_checkpoint_load_is_rejected(tmp_path, monkeypatch):
    pytest.importorskip('voicehub')
    import numpy as np
    import soundfile as sf
    import voicehub_arena.auth as auth
    import voicehub_arena.transport as transport
    import voicehub_arena.catalog as catalog
    from voicehub_arena.worker import generate
    monkeypatch.setattr(auth,'configure_hub_auth',lambda:None)
    monkeypatch.setattr(transport,'enable_verified_cache_reuse',lambda:None)
    monkeypatch.setattr(transport,'resolved_artifacts',lambda:[])
    monkeypatch.setattr(catalog,'declared_languages',lambda name:['en'])
    directory = tmp_path/'kokoro'; directory.mkdir()
    audio = directory/'a-0.wav'
    sf.write(audio,np.zeros(160),16000)
    row = {'id':'a','repeat':0,'status':'generated','audio':'kokoro/a-0.wav',
           'audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest()}
    write_json(directory/'result.json', {'status':'failed','rows':[row]})
    artifact = tmp_path/'checkpoint'; artifact.mkdir()
    (artifact/'data').write_bytes(b'changed checkpoint')
    cfg = {'dataset':[{'id':'a','text':'Hello.','category':'read'}], 'repeats':1, 'resume_samples':True,
           'catalog':[{'model_type':'kokoro','checkpoint':str(artifact)}],
           'overrides':{'kokoro':{'artifact_provenance':{'files_sha256':{'data':'0'*64}}}}}
    write_json(tmp_path/'config.json',cfg)
    generate(tmp_path/'config.json','kokoro')
    result = json.loads((directory/'result.json').read_text())
    assert result['status'] == 'blocked'
    assert 'differs from' in result['error']
    assert result['rows'] == [row]
