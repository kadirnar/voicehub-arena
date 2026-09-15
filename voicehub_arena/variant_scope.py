"""User-selected execution/display scope, separate from immutable run provenance."""
from pathlib import Path
import hashlib
import json


def load_selection(cfg, manifest='configs/variant-campaign.json'):
    manifest = Path(manifest)
    selected = json.loads((manifest.parent/'variant-selection.json').read_text())
    if selected['campaign'] != cfg['campaign'] or selected['frozen_manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Selection does not match the frozen campaign')
    by_id = {m['id']:m for m in cfg['models']}
    ids = selected['active_model_ids']
    if not ids or len(set(ids)) != len(ids) or set(ids)-set(by_id):
        raise ValueError('Invalid selected models')
    return selected, [by_id[k] for k in ids]


def require_active(model_id, cfg, manifest='configs/variant-campaign.json'):
    selected, _ = load_selection(cfg, manifest)
    if model_id not in selected['active_model_ids']:
        raise ValueError('Model removed from active scope by user: '+model_id)
