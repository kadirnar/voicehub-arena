from voicehub_arena.auth import configure_hub_auth


def test_protected_store_bridges_to_native_transport(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "test-credential")
    configure_hub_auth()
    import os
    assert os.environ["HF_TOKEN"] == "test-credential"
    assert capsys.readouterr() == ("", "")


def test_explicit_disable_prevents_implicit_credentials(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "true")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "test-credential")
    configure_hub_auth()
    import os
    assert "HF_TOKEN" not in os.environ
