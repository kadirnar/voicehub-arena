import hashlib
import json
import numpy as np
import pytest
from voicehub_arena.inputs import prepared_features


def test_prepared_features_require_exact_text_and_verified_artifact(tmp_path):
    text = 'The target sentence.'
    file = tmp_path/'features.npz'
    np.savez(file, input_ids=np.array([2,3,4]))
    key = hashlib.sha256(text.encode()).hexdigest()
    manifest = {'model':'melotts','entries':{key:{'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    np.testing.assert_array_equal(prepared_features(str(tmp_path),text)['input_ids'],[2,3,4])
    with pytest.raises(KeyError):
        prepared_features(str(tmp_path),'A different sentence.')
    prepared_features.cache_clear()
    file.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='digest mismatch'):
        prepared_features(str(tmp_path),text)


def test_xtts_uses_release_number_currency_and_abbreviation_cleaner():
    pytest.importorskip('voicehub')
    from voicehub_arena.inputs import prepare_request
    generation = {'language':'en', 'speaker_audio_path':'reference.wav'}
    text, options = prepare_request('xtts', 'Dr. Smith paid $12.50 for 3 books.', generation)
    assert text == 'doctor smith paid twelve dollars, fifty cents for three books.'
    assert options['text_is_normalized'] is True
    assert options['speaker_audio_path'] == 'reference.wav'
    assert 'text_is_normalized' not in generation
