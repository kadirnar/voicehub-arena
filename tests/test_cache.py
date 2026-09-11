import os
from pathlib import Path
import time
import pytest
from voicehub_arena.cache import cleanup_abandoned_downloads, prune_native_cache
from voicehub_arena.cache import cleanup_dead_download_locks


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor inventory required')
def test_only_closed_old_native_partial_downloads_are_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    cache = tmp_path/'hub/voicehub/repos/example'
    cache.mkdir(parents=True)
    names = ['.old.incomplete', '.open.incomplete', '.recent.incomplete', 'model.safetensors']
    for name in names:
        (cache/name).write_bytes(b'1234')
    old = time.time()-600
    for name in names:
        if name != '.recent.incomplete':
            os.utime(cache/name, (old,old))
    with (cache/'.open.incomplete').open('rb'):
        result = cleanup_abandoned_downloads()
        assert result == {'files':1, 'bytes':4}
        assert not (cache/'.old.incomplete').exists()
        assert all((cache/name).exists() for name in names[1:])


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor inventory required')
def test_cache_budget_preserves_open_and_next_model(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv('HF_HOME',str(tmp_path))
    root = tmp_path/'hub/voicehub/repos'
    repos = []
    for name in ('old','next','open'):
        repo = root/hashlib.sha256(name.encode()).hexdigest()
        repo.mkdir(parents=True)
        (repo/'model').write_bytes(b'1234')
        repos.append(repo)
    token = tmp_path/'token'
    token.write_text('unrelated')
    with (repos[2]/'model').open('rb'):
        assert prune_native_cache(4,['next']) == {'repositories':1,'bytes':4}
    assert not repos[0].exists()
    assert repos[1].exists() and repos[2].exists() and token.exists()


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux process inventory required')
def test_dead_locks_only_reclaimed_after_owner_exits(tmp_path, monkeypatch):
    import subprocess
    import sys
    child = subprocess.Popen([sys.executable, '-c', 'pass'])
    child.wait()
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    root = tmp_path/'hub/voicehub/locks'
    root.mkdir(parents=True)
    old = time.time()-20
    for name, owner in [('dead', child.pid), ('live', os.getpid())]:
        path = root/f'{name}.lock'
        path.write_text(f'{owner}:'+('a'*32))
        os.utime(path, (old, old))
    (root/'invalid.lock').write_text('invalid')
    assert cleanup_dead_download_locks() == 1
    assert not (root/'dead.lock').exists()
    assert (root/'live.lock').exists() and (root/'invalid.lock').exists()
