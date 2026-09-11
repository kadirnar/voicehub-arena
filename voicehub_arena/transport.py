"""Reuse digest-verified immutable VoiceHub downloads without another HTTP GET."""
import hashlib
import json
import re

_resolved = {}


def download_with_hub_client(options):
    """Use resumable Hub/Xet transport, then validate the native cache copy."""
    from pathlib import Path
    import voicehub.hub_transport as transport
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    repo, revision = options['repo_id'], options['revision']
    filename = options['relative_file'].as_posix()
    token = options['token'] if options['token'] is not None else False
    metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision), token=token)
    if metadata.commit_hash != revision:
        raise ValueError('Hub metadata does not match the immutable requested commit')
    etag = metadata.etag.strip('"') if metadata.etag else ''
    if not re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}', etag):
        raise ValueError('Hub file is missing a verifiable content digest')
    source = Path(hf_hub_download(repo, filename, revision=revision, token=token))
    if len(etag) == 40:
        digest = hashlib.sha1(b'blob '+str(source.stat().st_size).encode()+b'\0')
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*2**20), b''):
                digest.update(chunk)
        if digest.hexdigest() != etag:
            raise ValueError('Hub Git object digest does not match the downloaded file')
    snapshot_key = transport._snapshot_key(revision, metadata.commit_hash)
    destination = transport._safe_join(options['repository_cache'], 'snapshots',
                                       snapshot_key, *options['relative_file'].parts)
    with source.open('rb') as handle:
        size, sha256 = transport._download_atomic(handle, destination,
            expected_size=metadata.size, expected_sha256=etag if len(etag)==64 else None,
            immutable=True)
    transport._atomic_write_json(options['metadata_path'], {
        'version': 1, 'repo_id': repo, 'revision': revision, 'commit': metadata.commit_hash,
        'relative_file': filename, 'snapshot_key': snapshot_key, 'etag': etag,
        'size': size, 'sha256': sha256,
    })
    return destination


def resolved_artifacts():
    return list(_resolved.values())


def verified_immutable_cache(revision, cached):
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision) or cached is None or not cached.sha256:
        return False
    digest = hashlib.sha256()
    with cached.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8*2**20), b""):
            digest.update(chunk)
    return digest.hexdigest() == cached.sha256


def enable_verified_cache_reuse():
    import voicehub.hub_transport as transport
    original = transport._download_or_reuse
    if getattr(original, "arena_verified_cache", False):
        return

    def resolve(**options):
        # VoiceHub has already checked the metadata's repo/revision/file identity
        # and byte count. We additionally check every byte before local reuse.
        if verified_immutable_cache(options["revision"], options["cached"]):
            path = options["cached"].path
        elif re.fullmatch(r'[a-f0-9]{40}', options['revision']):
            path = download_with_hub_client(options)
        else:
            path = original(**options)
        try:
            metadata = json.loads(options['metadata_path'].read_text())
            fields = ('repo_id','revision','commit','relative_file','sha256','size')
            record = {key:metadata.get(key) for key in fields}
            _resolved[(record['repo_id'],record['commit'],record['relative_file'])] = record
        except (OSError, ValueError):
            pass
        return path

    resolve.arena_verified_cache = True
    transport._download_or_reuse = resolve
