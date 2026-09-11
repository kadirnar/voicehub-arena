import hashlib
from types import SimpleNamespace
from voicehub_arena.transport import verified_immutable_cache


def test_only_intact_immutable_cache_can_skip_network(tmp_path):
    path = tmp_path/'weights'
    path.write_bytes(b'original')
    cached = SimpleNamespace(path=path, sha256=hashlib.sha256(b'original').hexdigest())
    assert verified_immutable_cache('a'*40,cached)
    assert not verified_immutable_cache('main',cached)
    path.write_bytes(b'modified')
    assert not verified_immutable_cache('a'*40,cached)
    assert not verified_immutable_cache('a'*40,None)
