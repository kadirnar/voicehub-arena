"""Reuse digest-verified immutable VoiceHub downloads without another HTTP GET."""
import hashlib
import json
import re

_resolved = {}


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
