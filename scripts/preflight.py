"""Validate request contracts before downloading or allocating model weights."""
import json
from pathlib import Path
from voicehub_arena.auth import configure_hub_auth
configure_hub_auth()
from voicehub import AutoModelForTextToSpeech
from voicehub_arena.catalog import discover, declared_languages
from voicehub_arena.inputs import prepare_request

overrides = json.loads(Path('configs/models.json').read_text())
text = json.loads(Path('datasets/english.jsonl').read_text().splitlines()[0])['text']
for spec in discover():
    name = spec['model_type']
    settings = overrides.get(name,{})
    try:
        languages = declared_languages(name)
        if languages and not any(x.lower().startswith('en') for x in languages):
            print(name, 'UNSUPPORTED_ENGLISH', flush=True)
            continue
        checkpoint = settings.get('checkpoint',spec['checkpoint'])
        if name == 'vibevoice' and settings.get('runtime_adapter'):
            from voicehub_arena.vibevoice_adapter import ArenaVibeVoice
            model = ArenaVibeVoice.from_pretrained(checkpoint, device='cpu', **settings.get('config',{}))
        else:
            model = AutoModelForTextToSpeech.from_pretrained(checkpoint, model_type=name,device='cpu',**settings.get('config',{}))
        prepared_text, options = prepare_request(name,settings.get('text_prefix','')+text,
            settings.get('generation',{}),prepared_inputs=settings.get('prepared_inputs'))
        defaults = model.generation_config.to_dict()
        defaults.update(seed=42,**options)
        prepared = model.prepare_inputs_for_generation(prepared_text,**defaults)
        model._validate_model_kwargs(prepared)
        model._validate_common_generation_inputs(prepared)
        model._validate_generation_inputs(prepared)
        print(name, 'CONTRACT_OK', flush=True)
    except Exception as error:
        print(name, type(error).__name__, str(error),flush=True)
