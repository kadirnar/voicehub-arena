"""Convert the three immutable, audited official CosyVoice3 weights on CPU."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import shutil

import torch
import voicehub
from transformers import Qwen2TokenizerFast
from huggingface_hub import hf_hub_download
from voicehub.architectures.cosyvoice_native.checkpoint import (
    convert_audited_cosyvoice_legacy_checkpoint, validate_cosyvoice_checkpoint,
)
from voicehub.architectures.cosyvoice_native.configuration import CosyVoiceArchitectureConfig
from voicehub.architectures.cosyvoice_native.metadata import (
    COSYVOICE3_MODEL_ID, COSYVOICE3_MODEL_REVISION, COSYVOICE3_LEGACY_FILES,
)
from voicehub.architectures.cosyvoice_native.modeling import CosyVoiceNativeModel
from voicehub.architectures.cosyvoice_native.tokenization import CosyVoiceTextTokenizer


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b''):
            value.update(chunk)
    return value.hexdigest()


root = Path(__file__).resolve().parents[1]
output = root/'artifacts/cosyvoice3'
output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
config = CosyVoiceArchitectureConfig()
with torch.device('meta'):
    model = CosyVoiceNativeModel(config, initialize=False)
manifest = {'repo_id': COSYVOICE3_MODEL_ID, 'revision': COSYVOICE3_MODEL_REVISION,
            'conversion': 'VoiceHub audited size/SHA256, restricted weights-only, exact inventory',
            'source_files': COSYVOICE3_LEGACY_FILES, 'files_sha256': {}}
for component, identity in COSYVOICE3_LEGACY_FILES.items():
    target = output/f'{component}.safetensors'
    if not target.exists():
        source = hf_hub_download(COSYVOICE3_MODEL_ID, identity['filename'], revision=COSYVOICE3_MODEL_REVISION)
        convert_audited_cosyvoice_legacy_checkpoint(
            getattr(model, component), source, target, component=component)
    report = validate_cosyvoice_checkpoint(getattr(model, component), target,
                                           component=component, require_official_inventory=True)
    manifest['files_sha256'][target.name] = digest(target)
    print(component, report.tensor_count, 'audited tensors verified', flush=True)
    gc.collect()

(output/'cosyvoice_config.json').write_text(json.dumps(config.to_dict(), indent=2))
for filename in ('vocab.json', 'merges.txt', 'tokenizer_config.json'):
    source = hf_hub_download(COSYVOICE3_MODEL_ID, 'CosyVoice-BlankEN/'+filename,
                             revision=COSYVOICE3_MODEL_REVISION)
    shutil.copyfile(source, output/filename)
# The published BlankEN tokenizer is extended at construction by the released
# CosyVoice3 wrapper. Read only its literal dictionary; never execute source code.
source_path = Path(voicehub.__file__).parent/'models/cosyvoice/source/cosyvoice/tokenizer/tokenizer.py'
tree = ast.parse(source_path.read_text())
provider = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CosyVoice3Tokenizer')
initializer = next(node for node in provider.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
assignment = next(node for node in initializer.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'special_tokens' for target in node.targets))
special = ast.literal_eval(assignment.value)
reference = Qwen2TokenizerFast.from_pretrained(str(output), local_files_only=True)
reference.add_special_tokens(special)
reference.save_pretrained(output)
native = CosyVoiceTextTokenizer.from_files(output/'vocab.json', output/'merges.txt',
                                          output/'tokenizer_config.json', validate_published_ids=True)
texts = [json.loads(line)['text'] for line in (root/'datasets/english.jsonl').read_text().splitlines() if line]
texts += ['You are a helpful assistant. Speak clearly.<|endofprompt|>', ' '.join(special['additional_special_tokens'])]
for text in texts:
    if list(native.encode(text).input_ids) != reference.encode(text, add_special_tokens=False):
        raise ValueError('CosyVoice native tokenizer differs from the prepared Qwen2 reference')
manifest['tokenizer_preparation'] = {'source_sha256': digest(source_path),
    'source_class': 'CosyVoice3Tokenizer', 'special_tokens': special,
    'reference': 'Transformers Qwen2TokenizerFast', 'parity_examples': len(texts)}
for filename in ('cosyvoice_config.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json'):
    manifest['files_sha256'][filename] = digest(output/filename)
(output/'provenance.json').write_text(json.dumps(manifest, indent=2))
models_path = root/'configs/models.json'
models = json.loads(models_path.read_text())
models.setdefault('cosyvoice', {}).update(checkpoint=str(output), artifact_provenance=manifest)
models_path.write_text(json.dumps(models, indent=2))
print('Native CosyVoice3 checkpoint and tokenizer ready.', flush=True)
