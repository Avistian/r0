"""Corrected source scaling against realized masks, with legacy P1 kept separate."""

import json
import math
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
@pytest.mark.parametrize(
    ("profile", "family"),
    [("tabicl-individual", "individual-sparse"), ("tabicl-block", "block-sparse")],
)
@pytest.mark.parametrize("exponent", [0.0, 0.5])
def test_tabicl_mask_scale_raw_replay(
    engine: Any, profile: str, family: str, exponent: float
) -> None:
    request = p.NetworkSamplingRequest(
        5,
        (7, 3),
        families=(family,),
        fan_in=False,
        mask_profile=profile,
        scale_exponent=exponent,
        weight_scale=2.0,
        dropout=1.0,
    )
    sample = p.sample_network(
        request, named_rng=np.random.Generator(engine(28)), raw_rng=np.random.Generator(engine(29))
    )
    for index, block in enumerate(sample.function.blocks):
        affine = block.experts[0]
        mask = np.array(affine.mask)
        if family == "individual-sparse":
            keep = 1.0 if index == 0 else 0.01
            if index == 0:
                assert mask.all()
            raw = next(r for r in sample.raw if r.role == f"block:{index}-expert:0-mask")
            np.testing.assert_array_equal(
                mask, np.array(raw.values).reshape(mask.shape) >= (0 if index == 0 else 0.99)
            )
        else:
            count_draw = next(r for r in sample.raw if r.role == f"block:{index}-expert:0-blocks")
            count = int(count_draw.values[0]) + 1
            assert 1 <= count <= math.ceil(math.sqrt(min(mask.shape)))
            h, w = mask.shape[0] // count, mask.shape[1] // count
            row, column = np.indices(mask.shape)
            expected = (row // h == column // w) & (row < count * h) & (column < count * w)
            np.testing.assert_array_equal(mask, expected)
            keep = float(mask.sum()) / mask.size
            assert keep > 0
        assert affine.multiplier == pytest.approx(2.0 / (math.sqrt(keep) if exponent else 1.0))
        assert all(abs(b) <= 1 / math.sqrt(affine.input_width) for b in affine.bias)
    recipe = p.NetworkRecipe(
        p.LinearRecipe(
            rows=16,
            context_rows=8,
            features=5,
            signal_features=5,
            active_features=5,
            task_type="multiclass",
            classes=3,
        ),
        sample,
    )
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    task = p.generate_task(recipe, seed=28, task_id=17)
    assert p.explain_generation(task, [15, 0, 15])["signal_max_error"] < 1e-12


def test_tabicl_profiles_reject_conflicting_families_and_non_source_exponents() -> None:
    request = p.NetworkSamplingRequest(
        3, (2,), families=("individual-sparse",), fan_in=False, mask_profile="tabicl-individual"
    )
    with pytest.raises(ValueError, match="profile|scaling|laws"):
        replace(request, scale_exponent=1.0)
    with pytest.raises(ValueError, match="profile|family"):
        replace(request, families=("dense",))
    with pytest.raises(ValueError, match="profile|fan-in"):
        replace(request, fan_in=True)


def test_tabicl_source_uniform_bias_cannot_have_an_unused_hyperprior() -> None:
    template = p.NetworkSamplingRequest(
        3, (2,), families=("block-sparse",), fan_in=False, mask_profile="tabicl-block"
    )
    with pytest.raises(ValueError, match="unused"):
        p.NetworkHyperpriorRequest(template, bias_scale=p.NetworkScalarPrior())
