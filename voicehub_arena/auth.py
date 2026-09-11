"""Bridge the protected HF credential store to VoiceHub's environment transport."""
import os


def configure_hub_auth():
    # VoiceHub's native HTTP transport reads the environment, while the official
    # Hub client also reads HF_HOME/token. Never serialize this into run config.
    if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").lower() in {"1", "true", "yes", "on"}:
        return
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    from huggingface_hub import get_token
    token = get_token()
    if token:
        os.environ["HF_TOKEN"] = token
