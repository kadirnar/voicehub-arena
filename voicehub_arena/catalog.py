import dataclasses
from voicehub import list_model_specs


def discover():
    rows = []
    for s in list_model_specs():
        if s.task.value != "text-to-speech":
            continue
        rows.append(dict(model_type=s.model_type, name=s.display_name,
                         checkpoint=s.default_model_path, capabilities=list(s.capabilities),
                         module=s.module, class_name=s.class_name))
    return sorted(rows, key=lambda r:r["model_type"])
