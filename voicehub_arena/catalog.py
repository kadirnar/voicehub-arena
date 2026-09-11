import dataclasses
import re
from pathlib import Path
import voicehub
from voicehub import list_model_specs


def declared_languages(model_type):
    """Use the audited model card in the installed source checkout, if present.

    Absence is unknown, not evidence that English is unsupported.
    """
    card=Path(voicehub.__file__).parent.parent/'docs/models/providers'/f'{model_type}.md'
    if not card.is_file():
        return None
    marker='<summary>Supported language abbreviations</summary>'
    source=card.read_text()
    if marker not in source:
        return None
    section=source.split(marker,1)[1].split('</details>',1)[0]
    languages=re.findall(r'`([a-zA-Z][a-zA-Z_-]*)`',section)
    return languages or None


def discover():
    rows = []
    for s in list_model_specs():
        if s.task.value != "text-to-speech":
            continue
        rows.append(dict(model_type=s.model_type, name=s.display_name,
                         checkpoint=s.default_model_path, capabilities=list(s.capabilities),
                         declared_languages=declared_languages(s.model_type),
                         module=s.module, class_name=s.class_name))
    return sorted(rows, key=lambda r:r["model_type"])
