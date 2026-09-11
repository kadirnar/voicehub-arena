"""Convert only the digest-pinned official ConversationTTS archive on CPU."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from voicehub.architectures.conversationtts import metadata
from voicehub.architectures.conversationtts.checkpoint import (
    _normalize_legacy_state, _validate_inventory, exportable_state_dict,
)
from voicehub.architectures.conversationtts.modeling import (
    ConversationTTSModel, ConversationTTSArchitectureConfig,
)
from voicehub.checkpointing import save_safetensors
from voicehub.hub import resolve_pretrained_file
from voicehub.models.conversationtts.configuration_conversationtts import ConversationTTSConfig
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.transport import enable_verified_cache_reuse


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


configure_hub_auth()
enable_verified_cache_reuse()
torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
output = root/'artifacts/conversationtts'
output.mkdir(parents=True, exist_ok=True)
target = output/'model.safetensors'
source = resolve_pretrained_file(metadata.CONVERSATIONTTS_CHECKPOINT_REPOSITORY,
    metadata.CONVERSATIONTTS_CHECKPOINT_FILENAME, revision=metadata.CONVERSATIONTTS_CHECKPOINT_REVISION)
if (source.stat().st_size != metadata.CONVERSATIONTTS_CHECKPOINT_SIZE or
        sha256(source) != metadata.CONVERSATIONTTS_CHECKPOINT_SHA256):
    raise ValueError('ConversationTTS archive differs from the immutable audit')
allowed_names = {'datetime.timedelta', 'numpy.dtype', 'numpy.core.multiarray.scalar'}
unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(source)) - allowed_names
if unexpected:
    raise ValueError('Unexpected checkpoint types; refusing conversion: '+', '.join(sorted(unexpected)))
# The pinned archive stores NumPy scalar metrics and a duration outside model
# state. No unrestricted unpickler or dynamically expanded allowlist is used.
allowed = [datetime.timedelta, np.dtype,
           (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
           type(np.dtype('float64')), type(np.dtype('float32'))]
with torch.serialization.safe_globals(allowed):
    payload = torch.load(source, map_location='cpu', weights_only=True)
state = _normalize_legacy_state(payload['model'])
with torch.device('meta'):
    graph = ConversationTTSModel(ConversationTTSArchitectureConfig(**ConversationTTSConfig().model_args))
expected = exportable_state_dict(graph)
_validate_inventory(expected, {name: tuple(tensor.shape) for name, tensor in state.items()}, path=source)
save_safetensors(state, target, metadata={'format': metadata.NATIVE_CONVERSATIONTTS_FORMAT,
                                        'producer': 'voicehub-arena-audited-conversion'})
provenance = {'repo_id': metadata.CONVERSATIONTTS_CHECKPOINT_REPOSITORY,
    'revision': metadata.CONVERSATIONTTS_CHECKPOINT_REVISION,
    'source_filename': metadata.CONVERSATIONTTS_CHECKPOINT_FILENAME,
    'source_sha256': metadata.CONVERSATIONTTS_CHECKPOINT_SHA256,
    'sha256': sha256(target), 'tensor_count': len(state),
    'parameter_count': sum(value.numel() for value in state.values()),
    'conversion': 'Pinned digest; restricted weights-only NumPy scalar/timedelta metadata; exact native inventory'}
(output/'provenance.json').write_text(json.dumps(provenance, indent=2))
path = root/'configs/models.json'
models = json.loads(path.read_text())
models.setdefault('conversationtts', {}).update(checkpoint=str(target), artifact_provenance=provenance)
path.write_text(json.dumps(models, indent=2))
print('ConversationTTS tensor-only checkpoint verified:', len(state), 'tensors', flush=True)
