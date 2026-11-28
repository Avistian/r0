"""Public functional decoding rejects aggregate resource breaches before numeric arrays."""

import json
from dataclasses import asdict
from functools import partial
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors


@pytest.mark.parametrize(
    "cap", [{"node_channels": 1}, {"total_channels": 1}, {"working_bytes": 8192}]
)
@pytest.mark.parametrize("limit_owner", ["caller", "payload"])
def test_covariance_graph_admission_precedes_array_allocation(
    monkeypatch: pytest.MonkeyPatch,
    cap: dict[str, int],
    limit_owner: str,
) -> None:
    base = priors.JointRootBase(
        "covariance",
        2,
        gaussian_factor=((1.0, 0.0), (0.0, 1.0)),
        weights=priors.UniformWeightVector((0.5, 0.5)),
        factor=((0.5, 0.0), (0.0, 0.5)),
        covariance_base="gaussian",
    )
    recipe = priors.FunctionalCausalRecipe(
        priors.LinearRecipe(
            rows=16, features=1, signal_features=1, active_features=1, context_rows=8
        ),
        (
            priors.FunctionalCausalNode(
                (), priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), root=base
            ),
        ),
        ((0, 0),),
        ((0, 1),),
    )
    payload = json.loads(json.dumps(asdict(recipe)))
    if limit_owner == "payload":
        payload["limits"].update(cap)
    allocated: list[tuple[int, ...]] = []
    original = np.asarray

    def observe(*args: Any, **kwargs: Any) -> NDArray[Any]:
        array = original(*args, **kwargs)
        allocated.append(array.shape)
        return array

    # Observe the numerical-library allocation boundary without changing its result.
    monkeypatch.setattr(np, "asarray", observe)
    decoder = (
        partial(priors.functional_recipe_from_dict, limits=priors.CausalLimits(**cap))
        if limit_owner == "caller"
        else priors.recipe_from_dict
    )
    with pytest.raises(ValueError, match="limit"):
        decoder(payload)
    assert allocated == []


@pytest.mark.parametrize(
    ("value", "weight", "expected"),
    [
        (2.0, 0.5, 1.0),
        (-1e308, 0.5, -5e307),
        (5e-324, 0.5, 0.0),
        (1e-300, 1e-300, 0.0),
        (3.0, 0.0, 0.0),
    ],
)
def test_array_free_root_validation_preserves_binary64_factor_equation(
    value: float,
    weight: float,
    expected: float,
) -> None:
    base = priors.JointRootBase(
        "covariance",
        1,
        gaussian_factor=((value,),),
        weights=priors.UniformWeightVector((weight,)),
        factor=((expected,),),
        covariance_base="gaussian",
    )
    assert base.factor == ((expected,),)
    wrong = float(np.nextafter(expected, np.inf))
    with pytest.raises(ValueError, match="column-weighted construction"):
        priors.JointRootBase(
            "covariance",
            1,
            gaussian_factor=((value,),),
            weights=priors.UniformWeightVector((weight,)),
            factor=((wrong,),),
            covariance_base="gaussian",
        )
