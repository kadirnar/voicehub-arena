"""Fish categorical sampling must never select a token with zero probability."""
from unittest.mock import patch

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('voicehub')
from voicehub.architectures.fishtts.sampling import sample_exponential_race


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_zero_uniform_draw_still_selects_the_only_possible_token(dtype):
    probabilities = torch.tensor([0., 0., 1., 0.], dtype=dtype)
    uniform = torch.tensor([.5, .5, 0., .5], dtype=dtype)
    with patch.object(torch, 'rand_like', return_value=uniform) as draw:
        token = sample_exponential_race(probabilities)
    assert token.item() == 2
    assert token.dtype == torch.long
    draw.assert_called_once_with(probabilities)


def test_tied_zero_races_never_choose_a_masked_token():
    probabilities = torch.tensor([0., .4, 0., .6])
    uniform = torch.tensor([.4, 0., .8, 0.])
    with patch.object(torch, 'rand_like', return_value=uniform):
        token = sample_exponential_race(probabilities)
    assert probabilities[token] > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_ordinary_choices_and_rng_consumption_are_preserved(dtype):
    probabilities = torch.tensor([0., .1, .6, 0., .3], dtype=dtype)
    for seed in range(20):
        torch.manual_seed(seed)
        uniform = torch.rand_like(probabilities)
        expected = (probabilities / -uniform.log()).argmax()
        expected_state = torch.get_rng_state()
        assert probabilities[expected] > 0
        torch.manual_seed(seed)
        assert torch.equal(sample_exponential_race(probabilities), expected)
        assert torch.equal(torch.get_rng_state(), expected_state)
