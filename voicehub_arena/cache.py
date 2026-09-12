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


def prune_checkpoint_caches(budget_bytes, protected_repos=(), min_free_bytes=0):
    """Reclaim paired native/Hub model caches between workers under the GPU lock.

    Count hardlinked files once. Protect the next model, ASR and open handles in
    either cache. Hub deletion uses its revision API, retaining resumable partial
    downloads and unrelated datasets. Credentials and prepared artifacts are
    outside the managed hub directory.
    """
    if budget_bytes <= 0 or min_free_bytes < 0 or not os.environ.get('HF_HOME'):
        raise ValueError('Cache management requires positive budget and explicit HF_HOME')
    from huggingface_hub import scan_cache_dir

    root = Path(os.environ['HF_HOME'])/'hub'
    opened = open_paths()
    if opened is None or not root.is_dir():
        return {'repositories': 0, 'freed_bytes': 0, 'skipped': 'No safe cache inventory'}
    protected = {hashlib.sha256(repo.encode()).hexdigest() for repo in protected_repos}
    groups = {}
    native_root = root/'voicehub/repos'
    if native_root.is_dir():
        for native in native_root.iterdir():
            if not native.is_symlink() and native.is_dir() and re.fullmatch('[a-f0-9]{64}', native.name):
                groups[native.name] = {'native': native, 'paths': [native], 'revisions': []}
    info = scan_cache_dir(root)
    for repo in info.repos:
        if repo.repo_type != 'model' or repo.repo_path.is_symlink():
            continue
        key = hashlib.sha256(repo.repo_id.encode()).hexdigest()
        group = groups.setdefault(key, {'paths': [], 'revisions': []})
        group['paths'].append(repo.repo_path)
        group['revisions'].extend(rev.commit_hash for rev in repo.revisions)

    def physical_file_bytes():
        seen = set()
        total = 0
        for path in root.rglob('*'):
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                seen.add(inode)
                total += stat.st_size
        return total

    candidates = []
    for key, group in groups.items():
        paths = [p.resolve() for p in group['paths']]
        if key in protected or any(p == used or p in used.parents for p in paths for used in opened):
            continue
        # Deleting the last Hub revision can delete the whole repository, so
        # exclude any group with resumable partials before asking Hub to evict it.
        if any(next(folder.rglob('*.incomplete'), None) is not None for folder in paths):
            continue
        stamps = [p.stat().st_mtime for folder in paths for p in folder.rglob('*')
                  if p.is_file() and not p.is_symlink()]
        candidates.append((max(stamps, default=0), key, group))
    before_free = shutil.disk_usage(root).free
    result = {'repositories': 0, 'freed_bytes': 0}
    for _, key, group in sorted(candidates):
        if physical_file_bytes() <= budget_bytes and shutil.disk_usage(root).free >= min_free_bytes:
            break
        # Recheck handles immediately before deleting this repository group.
        current = open_paths()
        if current is None:
            break
        if any(p.resolve() == used or p.resolve() in used.parents
               for p in group['paths'] for used in current):
            continue
        if any(next(p.rglob('*.incomplete'), None) is not None for p in group['paths']):
            continue
        if group.get('native') is not None:
            shutil.rmtree(group['native'])
        if group['revisions']:
            info.delete_revisions(*group['revisions']).execute()
        result['repositories'] += 1
    result['freed_bytes'] = max(0, shutil.disk_usage(root).free-before_free)
    result['free_bytes'] = shutil.disk_usage(root).free
    result['cache_bytes'] = physical_file_bytes()
    return result
