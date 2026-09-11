"""Install one real CMU Arctic x-vector from the published example dataset."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from huggingface_hub import hf_hub_download

root = Path(__file__).resolve().parents[1]
repo = "Matthijs/cmu-arctic-xvectors"
revision = "5c1297a9eb6c91714ea77c0d4ac5aca9b6a952e5"
archive = hf_hub_download(repo, "spkrec-xvect.zip", repo_type="dataset", revision=revision)
with zipfile.ZipFile(archive) as package:
    names = sorted(n for n in package.namelist() if "slt" in n and n.endswith(".npy"))
    if not names:
        raise ValueError("Published archive has no SLT speaker vectors")
    selected = names[0]
    vector = np.load(io.BytesIO(package.read(selected)), allow_pickle=False).squeeze()
if vector.shape != (512,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
    raise ValueError("Invalid real speaker vector")
output = root/"datasets/reference/speecht5_slt.npy"
np.save(output, vector, allow_pickle=False)
provenance = {"repo": repo, "revision": revision, "entry": selected,
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "purpose": "Published CMU Arctic SLT speaker preset; not the Emily reference voice."}
output.with_suffix(".json").write_text(json.dumps(provenance, indent=2)+"\n")
path = root/"configs/models.json"
config = json.loads(path.read_text())
config["speecht5"] = {"generation": {"speaker_embedding_path": str(output.relative_to(root))}}
path.write_text(json.dumps(config, indent=2)+"\n")
print("SpeechT5 real speaker vector prepared:", selected)
