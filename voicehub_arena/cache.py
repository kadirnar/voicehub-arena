"""Reclaim abandoned temporary native downloads after their worker has stopped."""
import os
import hashlib
import shutil
import re
from pathlib import Path
import time


def cleanup_dead_download_locks():
    """Reclaim Linux locks whose recorded process no longer exists.

    The controller calls this while holding Arena's GPU lock, before starting a
    worker. Live owners, recent/invalid lock files and shared caches are retained.
    """
    if not Path('/proc/self/fd').is_dir() or not os.environ.get('HF_HOME'):
        return 0
    root = Path(os.environ['HF_HOME'])/'hub/voicehub/locks'
    removed = 0
    for path in root.glob('*.lock'):
        try:
            stat = path.stat()
            owner = path.read_text()
            match = re.fullmatch(r'([1-9][0-9]*):[a-f0-9]{32}', owner)
            if not match or time.time()-stat.st_mtime < 5:
                continue
            try:
                os.kill(int(match[1]), 0)
            except ProcessLookupError:
                if path.stat().st_ino == stat.st_ino and path.read_text() == owner:
                    path.unlink()
                    removed += 1
            except PermissionError:
                pass
        except (FileNotFoundError, PermissionError, UnicodeError):
            continue
    return removed


def open_paths():
    if not Path('/proc/self/fd').is_dir():
        return None
    opened = set()
    for directory in Path('/proc').glob('[0-9]*/fd'):
        try:
            for fd in directory.iterdir():
                try:
                    opened.add(fd.resolve())
                except OSError:
                    pass
        except FileNotFoundError:
            continue
        except PermissionError:
            return None
    return opened


def cleanup_abandoned_downloads():
    if not Path("/proc/self/fd").is_dir():
        return {"files": 0, "bytes": 0}
    cache = Path(os.environ.get("HF_HOME", Path.home()/".cache/huggingface"))/"hub/voicehub"
    # Completed checkpoints and the official Hub client's resumable downloads
    # are deliberately excluded. Native random temporary files cannot resume.
    opened = open_paths()
    if opened is None:
        return {"files": 0, "bytes": 0}
    result = {"files": 0, "bytes": 0}
    for path in cache.rglob(".*.incomplete"):
        try:
            stat = path.stat()
            if path.resolve() in opened or time.time()-stat.st_mtime < 300:
                continue
            path.unlink()
            result["files"] += 1
            result["bytes"] += stat.st_size
        except FileNotFoundError:
            pass
    return result


def prune_native_cache(budget_bytes, protected_repos=()):
    """Bound only the dedicated native download cache, between GPU workers.

    Must be explicitly enabled with a dedicated HF_HOME. Original artifacts,
    credentials, official Hub cache, source, audio and results are out of scope.
    """
    if budget_bytes <= 0 or not os.environ.get('HF_HOME'):
        raise ValueError('Cache pruning requires a positive budget and explicit HF_HOME')
    root = Path(os.environ['HF_HOME'])/'hub/voicehub/repos'
    opened = open_paths()
    if opened is None or not root.is_dir():
        return {'repositories':0,'bytes':0}
    protected = {hashlib.sha256(repo.encode()).hexdigest() for repo in protected_repos}
    candidates = []
    for repo in root.iterdir():
        if repo.is_symlink() or not repo.is_dir() or len(repo.name) != 64:
            continue
        files = [p for p in repo.rglob('*') if p.is_file() and not p.is_symlink()]
        if not files:
            continue
        size = sum(p.stat().st_size for p in files)
        age = max(p.stat().st_mtime for p in files)
        in_use = any(repo.resolve() == path or repo.resolve() in path.parents for path in opened)
        candidates.append((age,repo,size,in_use or repo.name in protected))
    total = sum(item[2] for item in candidates)
    result = {'repositories':0,'bytes':0}
    for _,repo,size,protected in sorted(candidates):
        if total <= budget_bytes:
            break
        if protected:
            continue
        shutil.rmtree(repo)
        total -= size
        result['repositories'] += 1
        result['bytes'] += size
    return result
