from fastapi.testclient import TestClient
from voicehub_arena.server import create_app
from voicehub_arena.storage import write_json


def test_catalog_and_file_boundaries(tmp_path):
    write_json(tmp_path/'trial/config.json',{'catalog':[]})
    client=TestClient(create_app(tmp_path))
    assert client.get('/').status_code==200
    assert client.get('/api/runs').json()[0]['id']=='trial'
    assert client.get('/api/runs/trial').json()['results']==[]
    assert client.get('/files/trial/config.json').status_code==200
    assert client.get('/files/%2e%2e/secret.json').status_code==404
    (tmp_path/'secret.key').write_text('not served')
    assert client.get('/files/secret.key').status_code==404


def test_overview_uses_latest_attempt_and_preserves_audio_origin(tmp_path):
    protocol = {'dataset': [{'id':'en_01','text':'hello'}], 'repeats':3, 'asr':{'revision':'pinned'}}
    models = [{'model_type':name,'checkpoint':name} for name in ('a','b','c')]
    write_json(tmp_path/'full/config.json', {**protocol, 'catalog':models, 'started_at':1})
    for spec in models:
        write_json(tmp_path/'full'/spec['model_type']/'result.json',
                   {**spec,'status':'blocked','error':'old failure','rows':[]})
    write_json(tmp_path/'repair/config.json', {**protocol,'catalog':models[:2],'started_at':2})
    write_json(tmp_path/'repair/state.json', {'status':'running','model':'b'})
    write_json(tmp_path/'repair/a/result.json', {**models[0],'status':'completed','rows':[]})
    # A later short scan must not override the full protocol's model status.
    write_json(tmp_path/'smoke/config.json', {**protocol,'repeats':1,'catalog':models,'started_at':3})
    write_json(tmp_path/'next/queue.json', {'models':['c'], 'status':'waiting'})
    data = TestClient(create_app(tmp_path)).get('/api/overview').json()
    results = {r['model_type']:r for r in data['results']}
    assert results['a']['status']=='completed'
    assert results['a']['source_run']=='repair'
    assert results['b']['status']=='pending'
    assert 'error' not in results['b']
    assert results['c']['error']=='old failure'
    assert results['c']['current_status']=='queued'
    assert results['c']['queued_run']=='next'
    assert results['a']['planned_samples']==3
    assert data['active_runs'][0]['run']=='repair'
