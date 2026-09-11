import hashlib
from types import SimpleNamespace
from voicehub_arena.transport import verified_immutable_cache
import json
from pathlib import PurePosixPath
import pytest


def test_only_intact_immutable_cache_can_skip_network(tmp_path):
    path = tmp_path/'weights'
    path.write_bytes(b'original')
    cached = SimpleNamespace(path=path, sha256=hashlib.sha256(b'original').hexdigest())
    assert verified_immutable_cache('a'*40,cached)
    assert not verified_immutable_cache('main',cached)
    path.write_bytes(b'modified')
    assert not verified_immutable_cache('a'*40,cached)
    assert not verified_immutable_cache('a'*40,None)


@pytest.mark.parametrize('kind', ['git', 'lfs'])
@pytest.mark.parametrize('corrupt', [False, True])
def test_hub_copy_checks_content_before_publishing_metadata(tmp_path, monkeypatch, kind, corrupt):
    pytest.importorskip('voicehub')
    import huggingface_hub
    from voicehub_arena.transport import download_with_hub_client
    content = b'audited checkpoint'
    etag = (hashlib.sha256(content).hexdigest() if kind == 'lfs' else
            hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest())
    source = tmp_path/'source'
    source.write_bytes(b'altered checkpoint' if corrupt else content)
    def metadata(*args, **kwargs):
        assert kwargs['token'] is False
        return SimpleNamespace(commit_hash='a'*40, etag=etag, size=len(content))
    monkeypatch.setattr(huggingface_hub, 'get_hf_file_metadata', metadata)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **k: str(source))
    record = tmp_path/'metadata.json'
    options = dict(repo_id='test/model', revision='a'*40, token=None,
                   relative_file=PurePosixPath('model.safetensors'),
                   repository_cache=tmp_path/'native', metadata_path=record)
    if corrupt:
        with pytest.raises((ValueError, OSError)):
            download_with_hub_client(options)
        assert not record.exists()
    else:
        path = download_with_hub_client(options)
        assert path.read_bytes() == content
        assert path.samefile(source)
        saved = json.loads(record.read_text())
        assert saved['commit'] == 'a'*40
        assert saved['sha256'] == hashlib.sha256(content).hexdigest()
        assert 'token' not in saved


def test_blob_publication_preserves_an_existing_immutable_snapshot(tmp_path):
    pytest.importorskip('voicehub')
    from voicehub_arena.transport import _publish_verified_blob
    source, destination = tmp_path/'source', tmp_path/'native'
    source.write_bytes(b'new')
    destination.write_bytes(b'old')
    with pytest.raises(ValueError, match='conflicts'):
        _publish_verified_blob(source, destination, size=3,
                               sha256=hashlib.sha256(b'new').hexdigest())
    assert destination.read_bytes() == b'old'
    destination.write_bytes(b'new')
    _publish_verified_blob(source, destination, size=3,
                           sha256=hashlib.sha256(b'new').hexdigest())
    assert destination.read_bytes() == b'new'


def test_blob_publication_falls_back_on_cross_device_cache(tmp_path, monkeypatch):
    pytest.importorskip('voicehub')
    import errno
    import voicehub_arena.transport as transport
    source, destination = tmp_path/'source', tmp_path/'native'
    source.write_bytes(b'audited')
    def cross_device(*args):
        raise OSError(errno.EXDEV, 'different devices')
    monkeypatch.setattr(transport.os, 'link', cross_device)
    transport._publish_verified_blob(source, destination, size=7,
                                      sha256=hashlib.sha256(b'audited').hexdigest())
    assert destination.read_bytes() == b'audited'
    assert not destination.samefile(source)


def test_hub_copy_rejects_wrong_commit_before_download(tmp_path, monkeypatch):
    pytest.importorskip('voicehub')
    import huggingface_hub
    from voicehub_arena.transport import download_with_hub_client
    monkeypatch.setattr(huggingface_hub, 'get_hf_file_metadata', lambda *a, **k:
                        SimpleNamespace(commit_hash='b'*40))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda *a, **k: pytest.fail('must not download'))
    with pytest.raises(ValueError, match='immutable requested commit'):
        download_with_hub_client(dict(repo_id='test/model', revision='a'*40,
            token=None, relative_file=PurePosixPath('weights')))


@pytest.mark.parametrize('status, expected', [(404, FileNotFoundError), (401, PermissionError), (403, PermissionError)])
def test_hub_errors_preserve_provider_fallback_contract(monkeypatch, status, expected):
    import requests
    from huggingface_hub.errors import HfHubHTTPError
    import voicehub_arena.transport as transport
    response = requests.Response()
    response.status_code = status
    def fail(options):
        raise HfHubHTTPError('upstream request error', response=response)
    monkeypatch.setattr(transport, '_download_with_hub_client', fail)
    with pytest.raises(expected):
        transport.download_with_hub_client(dict(repo_id='test/model', revision='a'*40,
            relative_file=PurePosixPath('model.safetensors.index.json')))
