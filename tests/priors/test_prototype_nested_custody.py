"""Every nested base dataclass field is mandatory inside prototype custody."""

import copy
import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors as p


def cases():
    law = p.SourceRootLaw((p.SourceRootCoordinate("gaussian"),))
    return (
        p.JointRootBase(
            "covariance",
            1,
            gaussian_factor=((2.0,),),
            weights=p.UniformWeightVector((0.5,)),
            factor=((1.0,),),
            covariance_base="gaussian",
        ),
        p.JointRootBase(
            "covariance",
            1,
            gaussian_factor=((1.0,),),
            weights=p.PolynomialWeightVector((0.0,)),
            factor=((1.0,),),
            covariance_base="uniform",
        ),
        p.SourceRootBase(law),
        p.SourceRootBase(law, p.SourceRootCalibration(law, ("standardize",), 8, (0.0,), (1.0,))),
    )


def objects(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    if isinstance(value, dict):
        item = cast(dict[str, Any], value)
        yield path, item
        for key, child in item.items():
            yield from objects(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(cast(list[Any], value)):
            yield from objects(child, (*path, index))


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("batch", [False, True])
def test_all_nested_base_fields_are_required(index: int, batch: bool):
    plan = p.PrototypeRootPlan(cases()[index], 8, 0.5)
    record = (
        p.sample_prototype_batch_reference(
            plan, prototype_rng=np.random.default_rng(1), weights_rng=np.random.default_rng(2)
        )
        if batch
        else p.sample_prototype_root(plan, rng=np.random.default_rng(1))
    )
    decode = p.prototype_batch_reference_from_dict if batch else p.sampled_prototype_root_from_dict
    raw = json.loads(json.dumps(asdict(record)))
    assert decode(raw) == record
    for path, value in objects(raw["plan"]["base"]):
        for field in value:
            edited = copy.deepcopy(raw)
            parent = edited["plan"]["base"]
            for key in path:
                parent = parent[key]
            del parent[field]
            with pytest.raises(ValueError, match="."):
                decode(edited)
