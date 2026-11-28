"""Every prototype public decoder fails closed with ValueError for malformed records."""

import copy
import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors as p


def records():
    law = p.PrototypeWeightLaw()
    plan = p.PrototypeRootPlan(p.JointRootBase("gaussian", 2), 8, 0.5)
    sampled = p.sample_prototype_root(plan, rng=np.random.default_rng(1))
    batch = p.sample_prototype_batch_reference(
        plan, prototype_rng=np.random.default_rng(1), weights_rng=np.random.default_rng(2)
    )
    return [
        (p.prototype_weight_law_from_dict, asdict(law)),
        (p.prototype_root_plan_from_dict, asdict(plan)),
        (p.frozen_prototype_root_from_dict, asdict(sampled.root)),
        (p.sampled_prototype_root_from_dict, asdict(sampled)),
        (p.prototype_batch_reference_from_dict, asdict(batch)),
    ]


@pytest.mark.parametrize("index", range(5))
def test_all_top_level_missing_and_wrong_fields(index: int):
    decode, valid = records()[index]
    assert decode(valid) is not None
    for key in valid:
        missing = copy.deepcopy(valid)
        del missing[key]
        with pytest.raises(ValueError, match="prototype|fields|invalid|malformed"):
            decode(missing)
    invalid_values: list[Any] = [None, [], 1, "bad", False]
    for invalid in invalid_values:
        with pytest.raises(ValueError, match="prototype|object|record|metadata"):
            decode(invalid)


@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_nested_required_fields_and_malformed_container_types(index: int):
    decode, valid = records()[index]
    valid = json.loads(json.dumps(valid))

    def dictionaries(
        value: Any, path: tuple[Any, ...] = ()
    ) -> Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
        if isinstance(value, dict):
            yield path, cast(dict[str, Any], value)
            for key, child in cast(dict[str, Any], value).items():
                yield from dictionaries(child, (*path, key))
        elif isinstance(value, (tuple, list)):
            for key, child in enumerate(cast(list[Any], value)):
                yield from dictionaries(child, (*path, key))

    for path, child in dictionaries(valid):
        if not path:
            continue
        replacements: tuple[Any, ...] = (
            None,
            [],
            "bad",
            {},
            *({k: v for k, v in child.items() if k != omitted} for omitted in child),
        )
        for replacement in replacements:
            data = copy.deepcopy(valid)
            parent = data
            for key in path[:-1]:
                parent = parent[key]
            # Serialized records use mutable JSON arrays for mutation.
            parent[path[-1]] = replacement
            with pytest.raises(ValueError, match="."):
                decode(data)
