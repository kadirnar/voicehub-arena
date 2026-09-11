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
