"""Fetch the author's pinned LibriTTS release for the native StyleTTS2 loader."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
repo = "yl4579/StyleTTS2-LibriTTS"
revision = "3aa7ba7f8f275ec13dce21682a61494c35089e2a"
paths = {}
artifact = root/"artifacts/styletts2"
artifact.mkdir(parents=True, exist_ok=True)
for name in ("Models/LibriTTS/config.yml", "Models/LibriTTS/epochs_2nd_00020.pth"):
    cached = Path(hf_hub_download(repo, name, revision=revision)).resolve()
    target = artifact/Path(name).name
    # Native loaders inspect suffixes after resolve(); Hub symlinks resolve to
    # extensionless blobs. A hard link preserves the filename without a copy.
    if not target.exists():
        try:
            os.link(cached, target)
        except OSError:
            shutil.copyfile(cached, target)
    paths[Path(name).name] = str(target)
checkpoint = paths["epochs_2nd_00020.pth"]
with open(checkpoint, "rb") as stream:
    sha = hashlib.file_digest(stream, "sha256").hexdigest()
config_path = root / "configs/models.json"
config = json.loads(config_path.read_text())
config["styletts2"] = {
    "checkpoint": checkpoint,
    "artifact_provenance": {"repo": repo, "revision": revision, "sha256": sha},
    # The reviewed VoiceHub loader uses torch.load(weights_only=True) and
    # verifies every deployable parameter name and shape before assignment.
    "config": {"config_path": paths["config.yml"], "trust_pickle_checkpoint": True, "dtype": "float32"},
    "generation": {"speaker_audio_path": "datasets/reference/emily.wav"},
}
config_path.write_text(json.dumps(config, indent=2)+"\n")
print("StyleTTS2 official checkpoint prepared:", revision, sha)
