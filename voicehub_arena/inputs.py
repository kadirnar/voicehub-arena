"""Explicit text preparation for providers whose native graphs accept phonemes."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=128)
def prepared_features(directory, text):
    import numpy as np
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text())
    key = hashlib.sha256(text.encode()).hexdigest()
    entry = manifest["entries"][key]
    path = root/entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Prepared linguistic input digest mismatch")
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    if manifest["model"] == "gptsovits":
        with np.load(root/"reference.npz", allow_pickle=False) as data:
            result.update({key: data[key] for key in data.files})
    return result


def prepare_request(model_type, text, generation, *, prepared_inputs=None, model=None):
    options = dict(generation)
    if prepared_inputs:
        options.update(prepared_features(prepared_inputs, text))
        ssl = options.pop("ssl_features", None)
        if ssl is not None and model is not None:
            import torch
            if not hasattr(model, "_arena_prompt_semantic"):
                s2 = model.model.s2
                parameter = next(s2.parameters())
                with torch.no_grad():
                    codes = s2.extract_latent(torch.as_tensor(ssl, device=parameter.device, dtype=parameter.dtype))
                model._arena_prompt_semantic = codes[0, 0].unsqueeze(0)
            options["prompt_semantic_ids"] = model._arena_prompt_semantic
    if model_type == "inflecttts":
        from voicehub.models.inflecttts.source.inflect.inflect_vits_frontend import run_vits_frontend
        options["phoneme_text"] = run_vits_frontend(text).phoneme_text
    elif model_type == "styletts2":
        from phonemizer import phonemize
        from nltk import word_tokenize
        text = phonemize(text, language="en-us", backend="espeak", strip=True,
                         preserve_punctuation=True, with_stress=True)
        text = " ".join(word_tokenize(text, preserve_line=True))
        options["text_is_phonemes"] = True
    elif model_type == "zonos":
        from voicehub.models.zonos.source.zonos.conditioning import phonemize
        options["phonemes"] = phonemize([text], [options.get("language", "en-us")])[0]
    return text, options
