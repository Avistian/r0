"""Registry JSON must not coerce malformed legacy ledger containers."""

import copy
import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors


@pytest.mark.parametrize("events", [{}, "", [0]])
def test_registry_rejects_malformed_empty_ledger_events(events: Any) -> None:
    context = priors.JointSamplingContext(np.random.default_rng(1))
    with context.transaction("event-free"):
        pass
    payload = json.loads(json.dumps(asdict(context.freeze_registry())))
    payload["decisions"]["events"] = events
    with pytest.raises(ValueError, match="ledger|array|object"):
        priors.joint_operation_registry_from_dict(payload)


def _paths(
    value: Any, kind: type[Any], prefix: tuple[str | int, ...] = ()
) -> Iterator[tuple[str | int, ...]]:
    if type(value) is kind:
        yield prefix
    if type(value) is dict:
        for key, child in cast(dict[str, Any], value).items():
            yield from _paths(child, kind, (*prefix, key))
    elif type(value) is list:
        for key, child in enumerate(cast(list[Any], value)):
            yield from _paths(child, kind, (*prefix, key))


def _payload() -> dict[str, Any]:
    context = priors.JointSamplingContext(np.random.default_rng(11))
    with context.transaction("all-modes") as scope:
        for mode in ("local", "global", "meta"):
            scope.numerical(mode, priors.NumericalParameter(f"scale-{mode}", 0.0, 1.0, mode=mode))
            scope.categorical(
                f"category-{mode}",
                priors.CategoricalParameter(f"type-{mode}", ("a", "b"), mode=mode),
            )
    payload = json.loads(json.dumps(asdict(context.freeze_registry())))
    assert priors.joint_operation_registry_from_dict(payload) == context.freeze_registry()
    return payload


@pytest.mark.parametrize("replacement", [{}, "", 0, None])
def test_registry_rejects_wrong_json_types_for_every_ledger_array(replacement: Any) -> None:
    original = _payload()
    for path in _paths(original["decisions"], list, ("decisions",)):
        payload = copy.deepcopy(original)
        parent: Any = payload
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = replacement
        with pytest.raises(ValueError, match="array"):
            priors.joint_operation_registry_from_dict(payload)


def test_registry_rejects_nonobject_records_everywhere_in_ledger() -> None:
    original = _payload()
    for path in _paths(original["decisions"], dict, ("decisions",)):
        payload = copy.deepcopy(original)
        parent: Any = payload
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = []
        with pytest.raises(ValueError, match="object"):
            priors.joint_operation_registry_from_dict(payload)
