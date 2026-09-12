import os
from pathlib import Path
import time
import pytest
from voicehub_arena.cache import cleanup_abandoned_downloads, prune_native_cache
from voicehub_arena.cache import cleanup_dead_download_locks
from voicehub_arena.cache import prune_checkpoint_caches


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


def _paired_repo(root, name, content=b'1234'):
    import hashlib
    native = root/'hub/voicehub/repos'/hashlib.sha256(name.encode()).hexdigest()
    native.mkdir(parents=True)
    (native/'model.safetensors').write_bytes(content)
    official = root/'hub'/('models--'+name.replace('/', '--'))
    blob_name = hashlib.sha256(content).hexdigest()
    blob = official/'blobs'/blob_name
    blob.parent.mkdir(parents=True)
    os.link(native/'model.safetensors', blob)
    revision = hashlib.sha1(name.encode()).hexdigest()
    snapshot = official/'snapshots'/revision
    snapshot.mkdir(parents=True)
    (snapshot/'model.safetensors').symlink_to('../../blobs/'+blob_name)
    (official/'refs').mkdir()
    (official/'refs/main').write_text(revision)
    return native, official, blob


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor inventory required')
def test_paired_cache_counts_hardlinks_once_and_reclaims_both_references(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    native, official, _ = _paired_repo(tmp_path, 'org/old')
    (tmp_path/'token').write_text('unrelated')
    # Four payload bytes plus forty bytes in refs/main, despite two hardlinks.
    result = prune_checkpoint_caches(44)
    assert result['repositories'] == 0 and result['cache_bytes'] == 44
    result = prune_checkpoint_caches(1)
    assert result['repositories'] == 1 and result['cache_bytes'] == 0
    assert not native.exists() and not official.exists()
    assert (tmp_path/'token').read_text() == 'unrelated'


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor inventory required')
def test_paired_cache_preserves_open_next_and_resumable_repositories(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    groups = {name: _paired_repo(tmp_path, 'org/'+name) for name in ('old', 'next', 'open', 'partial')}
    partial = groups['partial'][1]/'blobs/download.incomplete'
    partial.write_bytes(b'resume me')
    with groups['open'][2].open('rb'):
        result = prune_checkpoint_caches(1, ['org/next'])
    assert result['repositories'] == 1
    assert not groups['old'][0].exists() and not groups['old'][1].exists()
    for name in ('next', 'open', 'partial'):
        assert groups[name][0].exists() and groups[name][1].exists()
    assert partial.read_bytes() == b'resume me'


@pytest.mark.skipif(not Path('/proc/self/fd').is_dir(), reason='Linux descriptor inventory required')
def test_paired_cache_reserves_free_space_even_below_the_cache_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    native, official, _ = _paired_repo(tmp_path, 'org/old')
    monkeypatch.setattr('voicehub_arena.cache.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=1000 if official.exists() else 5000))
    result = prune_checkpoint_caches(10**9, min_free_bytes=4000)
    assert result['repositories'] == 1 and result['freed_bytes'] == 4000
    assert not native.exists() and not official.exists()
