"""Convert Microsoft's pinned example voice cache into tensor-only storage."""
import hashlib
import json
from pathlib import Path
import urllib.request
import torch
from safetensors.torch import save_file
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

root = Path(__file__).resolve().parents[1]
revision = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
url = f"https://raw.githubusercontent.com/microsoft/VibeVoice/{revision}/demo/voices/streaming_model/en-Emma_woman.pt"
expected = "75b15c481e0d848991f1789620aa9929c583ec2c5f701f8152362cf74498bbf8"
source = root/"artifacts/vibevoice/en-Emma_woman.pt"
source.parent.mkdir(parents=True, exist_ok=True)
if not source.exists():
    source.write_bytes(urllib.request.urlopen(url, timeout=60).read(5_000_000))
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise ValueError("Official voice cache digest mismatch")
# Only these two inspected, trusted Transformers container classes are admitted.
# The runtime never unpickles this archive: it consumes the tensor-only export.
with torch.serialization.safe_globals([DynamicCache, BaseModelOutputWithPast]):
    data = torch.load(source, map_location="cpu", weights_only=True)
tensors = {}
for section in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
    value = data[section]
    tensors[f"{section}.hidden"] = value.last_hidden_state.contiguous()
    cache = vars(value.past_key_values)
    keys, values = cache["key_cache"], cache["value_cache"]
    if len(keys) != len(values) or len(keys) not in (4,20):
        raise ValueError("Unexpected official cache topology")
    for index,(key,item) in enumerate(zip(keys,values)):
        if key.shape != item.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError("Invalid cached key/value shape")
        tensors[f"{section}.key.{index}"] = key.contiguous()
        tensors[f"{section}.value.{index}"] = item.contiguous()
if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
    raise ValueError("Non-finite official cache tensor")
output = root/"datasets/reference/vibevoice_emma.safetensors"
save_file(tensors, output, metadata={"source_url":url,"source_sha256":expected})
output.with_suffix(".json").write_text(json.dumps({"url":url,"source_sha256":expected,
    "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"tensors":len(tensors)},indent=2)+"\n")
print("VibeVoice official cache converted:",len(tensors),"verified tensors")
