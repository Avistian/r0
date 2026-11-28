"""Source random affine stage against literal independent Gaussian draws."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import priors as p


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
@pytest.mark.parametrize("individual", [False, True])
def test_random_scale_source_laws(engine: Any, individual: bool) -> None:
    rng = np.random.Generator(engine(28))
    oracle = np.random.Generator(engine(28))
    count = 3 if individual else 1
    a = oracle.normal(size=count)
    b = oracle.normal(size=count)
    activation = p.sample_tabicl_random_scale(3, individual=individual, rng=rng)
    x = np.array([[-3.0, 0.0, 1.0], [2.0, 4.0, -1.0]])
    np.testing.assert_allclose(
        activation.apply(torch.tensor(x)).numpy(), np.exp(2 * a) * (x + b), rtol=1e-14, atol=1e-14
    )
    assert dict(activation.parameters)["normal_scale_0"] == a[0]
    assert dict(activation.parameters)["normal_shift_0"] == b[0]
    assert p.CausalActivation(**asdict(activation)) == activation
    with pytest.raises(ValueError, match="width|channels"):
        activation.apply(torch.ones((2, 4), dtype=torch.float64))


def test_random_scale_invalid_width_and_limits_preserve_stream() -> None:
    rng = np.random.default_rng(1)
    before = str(rng.bit_generator.state)
    for width in (0, -1):
        with pytest.raises(ValueError, match="width|channels"):
            p.sample_tabicl_random_scale(width, rng=rng)
    with pytest.raises(ValueError, match="limit"):
        p.sample_tabicl_random_scale(3, rng=rng, limits=p.CausalLimits(parameters=1))
    assert str(rng.bit_generator.state) == before


def test_random_scale_retains_representable_extremes_and_rejects_unbounded_metadata() -> None:
    import math

    activation = p.CausalActivation(
        "tabicl-random-scale-v1",
        (
            ("channels", 1),
            ("individual", 0),
            ("normal_scale_0", -math.log(1e308) / 2),
            ("normal_shift_0", 1e308),
        ),
    )
    torch.testing.assert_close(
        activation.apply(torch.tensor([[1e308]], dtype=torch.float64)),
        torch.tensor([[2.0]], dtype=torch.float64),
        rtol=1e-13,
        atol=1e-13,
    )
    with pytest.raises(ValueError, match="draw"):
        p.CausalActivation("tabicl-random-scale-v1", (("channels", 2**62), ("individual", 1)))
    with pytest.raises(ValueError, match="nonfinite"):
        p.CausalActivation(
            "tabicl-random-scale-v1",
            (
                ("channels", 1),
                ("individual", 0),
                ("normal_scale_0", 1e308),
                ("normal_shift_0", 0.0),
            ),
        )


def test_network_validates_random_scale_channel_geometry_before_evaluation() -> None:
    activation = p.sample_tabicl_random_scale(3, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="width|channels"):
        p.NetworkAffine(((1.0,),), (0.0,), activation=activation)
