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
