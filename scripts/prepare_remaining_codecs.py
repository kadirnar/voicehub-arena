"""Explicit CPU conversions of pinned VoxCPM2 and XTTS release artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from voicehub.hub import resolve_pretrained_file
from voicehub.checkpointing import SafeTensorReader
from voicehub_arena.auth import configure_hub_auth
from voicehub_arena.storage import write_json
from voicehub_arena.transport import enable_verified_cache_reuse, resolved_artifacts

ROOT = Path(__file__).resolve().parents[1]
XTTS_LEGACY_SHA256 = 'c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187'
XTTS_CONFIG_RECORDS = {
    'TTS.tts.configs.xtts_config.XttsConfig',
    'TTS.tts.models.xtts.XttsArgs',
    'TTS.tts.models.xtts.XttsAudioConfig',
    'TTS.config.shared_configs.BaseDatasetConfig',
}


class LegacyConfigurationRecord:
    """Inert storage for discarded checkpoint metadata; no upstream methods."""
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified_file(repo, revision, filename):
    path = resolve_pretrained_file(repo, filename, revision=revision)
    origin = next(item for item in reversed(resolved_artifacts())
                  if item['repo_id'] == repo and item['relative_file'] == filename)
    if origin['commit'] != revision or path.stat().st_size != origin['size'] or sha256(path) != origin['sha256']:
        raise ValueError('Pinned source verification failed before conversion')
    return path, origin


def prepare_voxcpm():
    from voicehub.architectures.voxcpm2 import metadata as meta
    from voicehub.architectures.voxcpm2.configuration import VoxCPM2ArchitectureConfig
    from voicehub.architectures.voxcpm2.codec import VoxCPMAudioVAE
    from voicehub.architectures.voxcpm2.checkpoint import convert_legacy_voxcpm_codec, validate_voxcpm_checkpoint

    repo, revision = meta.VOXCPM2_CHECKPOINT_REPOSITORY, meta.VOXCPM2_CHECKPOINT_REVISION
    config_path, config_origin = verified_file(repo, revision, 'config.json')
    legacy, origin = verified_file(repo, revision, meta.VOXCPM2_CODEC_LEGACY_FILE)
    config = VoxCPM2ArchitectureConfig.from_mapping(json.loads(config_path.read_text()))
    codec = VoxCPMAudioVAE(config.audio_vae_config, device='cpu', dtype=torch.float32)
    destination = ROOT/'artifacts/voxcpm2/audiovae.safetensors'
    destination.parent.mkdir(parents=True, exist_ok=True)
    convert_legacy_voxcpm_codec(codec, legacy, destination, trust_legacy_pickle=True,
                               verify_official_integrity=True)
    report = validate_voxcpm_checkpoint(codec, destination, require_official_inventory=True)
    provenance = {'source': origin, 'config': config_origin, 'converted_sha256': sha256(destination),
                  'config_files_sha256': {'codec_path': sha256(destination)},
                  'tensor_count': report.tensor_count, 'header_fingerprint': report.header_fingerprint,
                  'conversion': 'Pinned digest; weights_only=True; strict native AudioVAE inventory'}
    settings_path = ROOT/'configs/models.json'
    settings = json.loads(settings_path.read_text())
    settings['voxcpm']['config']['codec_path'] = str(destination)
    settings['voxcpm']['artifact_provenance'] = provenance
    write_json(settings_path, settings)
    write_json(ROOT/'runs/validations/voxcpm2-codec-conversion.json', provenance)
    print('VoxCPM codec conversion verified:', json.dumps(provenance), flush=True)


def prepare_xtts():
    from voicehub.architectures.xtts2 import XTTS2Config, XTTS2Model, XTTS2Tokenizer
    from voicehub.architectures.xtts2 import metadata as meta
    from voicehub.architectures.xtts2.checkpoint import convert_trusted_legacy_xtts2_checkpoint, inspect_xtts2_checkpoint

    repo, revision = meta.XTTS2_CHECKPOINT_REPOSITORY, meta.XTTS2_CHECKPOINT_REVISION
    legacy, origin = verified_file(repo, revision, 'model.pth')
    if origin['sha256'] != XTTS_LEGACY_SHA256 or origin['size'] != 1_867_929_118:
        raise ValueError('XTTS archive differs from the reviewed metadata conversion input')
    if set(torch.serialization.get_unsafe_globals_in_checkpoint(legacy)) != XTTS_CONFIG_RECORDS:
        raise ValueError('XTTS archive contains unexpected serialized configuration types')
    destination = ROOT/'artifacts/xtts2'
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in [('config.json', meta.XTTS2_CONFIG_SHA256), ('vocab.json', meta.XTTS2_VOCAB_SHA256)]:
        source, _ = verified_file(repo, revision, filename)
        if sha256(source) != expected:
            raise ValueError('XTTS config/tokenizer differs from the audited source')
        shutil.copyfile(source, destination/filename)
    checkpoint = destination/'model.safetensors'
    # These four classes are dataclass configuration records in the reviewed
    # source. Restore their data into inert objects rather than importing TTS or
    # executing configuration methods. The converter only reads payload['model'].
    with torch.serialization.safe_globals([(LegacyConfigurationRecord, name) for name in XTTS_CONFIG_RECORDS]):
        convert_trusted_legacy_xtts2_checkpoint(legacy, checkpoint, trust_legacy_pickle=True)
    config = XTTS2Config.from_json(destination/'config.json')
    tokenizer = XTTS2Tokenizer.from_file(destination/'vocab.json')
    with torch.device('meta'):
        model = XTTS2Model(config, start_text_token=tokenizer.start_id, stop_text_token=tokenizer.stop_id)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    with SafeTensorReader(checkpoint) as reader:
        actual = {name: reader.tensor_shape(name) for name in reader.keys()}
    if actual != expected:
        raise ValueError('Converted XTTS tensor names/shapes do not match the full native graph')
    report = inspect_xtts2_checkpoint(checkpoint)
    if report.tensor_count != meta.XTTS2_NATIVE_TENSOR_COUNT or report.parameter_count != meta.XTTS2_NATIVE_PARAMETER_COUNT:
        raise ValueError('Converted XTTS inventory differs from the audited release')
    provenance = {'source': origin, 'converted_sha256': sha256(checkpoint),
                  'files_sha256': {name: sha256(destination/name) for name in ('model.safetensors', 'config.json', 'vocab.json')},
                  'tensor_count': report.tensor_count, 'header_fingerprint': report.header_fingerprint,
                  'configuration_records': sorted(XTTS_CONFIG_RECORDS),
                  'conversion': 'Pinned digest; weights_only=True; four inert discarded config records; official trainer tensors removed; exact native graph inventory'}
    settings_path = ROOT/'configs/models.json'
    settings = json.loads(settings_path.read_text())
    settings.setdefault('xtts', {})['checkpoint'] = str(destination)
    settings['xtts']['artifact_provenance'] = provenance
    write_json(settings_path, settings)
    write_json(ROOT/'runs/validations/xtts2-conversion.json', provenance)
    print('XTTS conversion verified:', json.dumps(provenance), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['voxcpm', 'xtts'])
    args = parser.parse_args()
    configure_hub_auth()
    enable_verified_cache_reuse()
    torch.set_num_threads(2)
    {'voxcpm': prepare_voxcpm, 'xtts': prepare_xtts}[args.model]()
