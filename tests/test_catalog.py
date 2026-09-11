from voicehub_arena import catalog


def test_language_metadata_is_complete_and_absence_is_unknown(tmp_path,monkeypatch):
    monkeypatch.setattr(catalog.voicehub,'__file__',str(tmp_path/'voicehub/__init__.py'))
    assert catalog.declared_languages('example') is None
    folder=tmp_path/'docs/models/providers'
    folder.mkdir(parents=True)
    (folder/'example.md').write_text('<summary>Supported language abbreviations</summary>\n`ja`, `zh`, `en-us`\n</details>')
    assert catalog.declared_languages('example')==['ja','zh','en-us']
    (folder/'example.md').write_text('Language coverage is unknown.')
    assert catalog.declared_languages('example') is None
