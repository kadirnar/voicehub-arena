"""Numerical regression tests for the bundled native VITS runtime patch."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('voicehub')
from voicehub.architectures.vits.modeling import (
    VitsGenerationError,
    _rational_quadratic_spline,
    _unconstrained_rational_quadratic_spline,
)


def test_low_precision_inverse_recovers_valid_monotonic_spline():
    fixture = json.loads((Path(__file__).parent / 'fixtures/vits-spline-low-precision.json').read_text())
    values = [torch.tensor(x, dtype=getattr(torch, fixture['dtype'])) for x in fixture['values']]
    # This valid in-domain input raised a negative-discriminant exception in
    # bfloat16. Float64 inversion and a forward round trip establish the target.
    reference, ref_logdet = _rational_quadratic_spline(*(v.double() for v in values), **fixture['options'])
    output, logdet = _rational_quadratic_spline(*values, **fixture['options'])
    assert output.dtype == logdet.dtype == values[0].dtype
    assert torch.isfinite(output).all() and torch.isfinite(logdet).all()
    assert torch.equal(output, reference.to(output.dtype))
    assert torch.equal(logdet, ref_logdet.to(logdet.dtype))
    forward, forward_logdet = _rational_quadratic_spline(reference, *(v.double() for v in values[1:]),
                                                       **{**fixture['options'], 'reverse': False})
    torch.testing.assert_close(forward, values[0].double(), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(forward_logdet + ref_logdet, torch.zeros_like(ref_logdet), rtol=0, atol=1e-7)


def test_spline_identity_tails_and_interior_round_trip():
    generator = torch.Generator().manual_seed(201)
    values = torch.tensor([-7., -5., -1., 0., 2., 5., 7.], dtype=torch.float32)
    params = [torch.randn(7, n, generator=generator) * .3 for n in (10, 10, 9)]
    forward, logdet = _unconstrained_rational_quadratic_spline(values, *params)
    recovered, reverse_logdet = _unconstrained_rational_quadratic_spline(forward, *params, reverse=True)
    torch.testing.assert_close(recovered, values, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logdet + reverse_logdet, torch.zeros_like(logdet), rtol=0, atol=1e-5)
    assert torch.equal(forward[[0, 6]], values[[0, 6]])
    assert torch.equal(logdet[[0, 6]], torch.zeros(2))


def test_invalid_inverse_is_not_silently_clamped():
    with pytest.raises(VitsGenerationError):
        _rational_quadratic_spline(torch.tensor([float('nan')]), torch.zeros(1, 10),
                                  torch.zeros(1, 10), torch.zeros(1, 11), reverse=True,
                                  tail_bound=5., min_bin_width=.001, min_bin_height=.001,
                                  min_derivative=.001)
