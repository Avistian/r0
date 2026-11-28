"""Saved root realizations must bind raw draws, calibration and frozen outputs."""

import copy
import json
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from r0.modules import priors


def _root(
    family: str = "covariance", function: str = "tree", vector: str = "uniform", seed: int = 110
) -> tuple[priors.JointRootRealization, priors.JointSamplingRecord]:
    context = priors.JointSamplingContext(np.random.default_rng(seed), mode="local")
    plan = priors.sample_joint_root(
        context,
        "saved-root",
        width=2,
        calibration_rows=4,
        prior=priors.JointRootPrior(
            families=(family,),
            function=priors.JointFunctionPrior(
                functions=function,
                matrix=priors.JointMatrixPrior(families=("gaussian",), vector_family=vector),
            ),
        ),
    )
    decisions = context.freeze()
    root = priors.materialize_joint_root(
        plan,
        decisions,
        rng=np.random.default_rng(111),
        calibration_rng=np.random.default_rng(112),
        fit_rng=np.random.default_rng(113),
        calibration_id="saved-root-fit",
    )
    return root, decisions


@pytest.mark.parametrize(
    "path",
    [
        ("calibration", "base", "gaussian_factor", 0, 0),
        ("calibration", "base", "factor", 0, 0),
        ("calibration", "base", "weights", "draws", 0),
        ("calibration", "draws", 0, 0),
        ("calibration", "values", 0, 0),
        ("outputs", 0, 0),
    ],
)
def test_saved_root_rejects_independently_tampered_numeric_construction(
    path: tuple[str | int, ...],
) -> None:
    root, decisions = _root()
    payload = json.loads(json.dumps(asdict(root)))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] += 0.125
    with pytest.raises(ValueError, match="differ"):
        priors.joint_root_realization_from_dict(payload, decisions=decisions)


def test_saved_root_rejects_fit_identity_and_foreign_owned_function() -> None:
    root, decisions = _root()
    payload = json.loads(json.dumps(asdict(root)))
    payload["calibration_id"] = "some-other-fit"
    with pytest.raises(ValueError, match="fit differs"):
        priors.joint_root_realization_from_dict(payload, decisions=decisions)
    other, _ = _root(function="lin")
    payload = json.loads(json.dumps(asdict(root)))
    payload["function"] = json.loads(json.dumps(asdict(other.function)))
    with pytest.raises(
        ValueError, match="request|decision|owned|operation|family|parameters|events"
    ):
        priors.joint_root_realization_from_dict(payload, decisions=decisions)


def test_saved_root_rejects_missing_and_unknown_record_fields() -> None:
    root, decisions = _root()
    original = json.loads(json.dumps(asdict(root)))
    for path in ((), ("calibration",), ("calibration", "base"), ("calibration", "base", "weights")):
        node = original
        for name in path:
            node = node[name]
        for field in (*node, "unexpected"):
            payload = copy.deepcopy(original)
            target = payload
            for name in path:
                target = target[name]
            if field == "unexpected":
                target[field] = 0
            else:
                del target[field]
            with pytest.raises(ValueError, match="required fields"):
                priors.joint_root_realization_from_dict(payload, decisions=decisions)


def test_saved_root_rejects_boolean_numeric_fields_in_root_records() -> None:
    root, decisions = _root()
    original = json.loads(json.dumps(asdict(root)))

    def paths(value: Any, prefix: tuple[str | int, ...] = ()) -> Iterator[tuple[str | int, ...]]:
        if type(value) in (int, float):
            yield prefix
        elif type(value) is dict:
            for key, child in cast(dict[str, Any], value).items():
                yield from paths(child, (*prefix, key))
        elif type(value) is list:
            for key, child in enumerate(cast(list[Any], value)):
                yield from paths(child, (*prefix, key))

    for path in (
        *paths(original["calibration"], ("calibration",)),
        *paths(original["outputs"], ("outputs",)),
    ):
        payload = copy.deepcopy(original)
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = True
        with pytest.raises(ValueError, match="finite|owned plan|requires|require"):
            priors.joint_root_realization_from_dict(payload, decisions=decisions)


def test_saved_root_zero_covariance_factor_remains_valid_without_jitter() -> None:
    root, decisions = _root(function="lin")
    payload = json.loads(json.dumps(asdict(root)))
    base = payload["calibration"]["base"]
    base["weights"]["draws"] = [0.0, 0.0]
    base["factor"] = [[0.0, 0.0], [0.0, 0.0]]
    payload["calibration"]["values"] = [[0.0, 0.0]] * 4
    payload["outputs"] = [[0.0, 0.0]] * 4
    restored = priors.joint_root_realization_from_dict(payload, decisions=decisions)
    assert restored.outputs == ((0.0, 0.0),) * 4
    assert restored.calibration.base.factor == ((0.0, 0.0),) * 2


def test_saved_root_ball_zero_radius_valid_but_zero_direction_rejected() -> None:
    root, decisions = _root(family="ball", function="lin")
    payload = json.loads(json.dumps(asdict(root)))
    payload["calibration"]["radial_uniform"][0] = 0.0
    payload["calibration"]["values"][0] = [0.0, 0.0]
    payload["outputs"][0] = [0.0, 0.0]
    restored = priors.joint_root_realization_from_dict(payload, decisions=decisions)
    assert restored.outputs[0] == (0.0, 0.0)
    payload["calibration"]["draws"][0] = [0.0, 0.0]
    with pytest.raises(ValueError, match="zero root ball direction"):
        priors.joint_root_realization_from_dict(payload, decisions=decisions)


def test_saved_root_honors_caller_owned_resource_limits() -> None:
    root, decisions = _root()
    payload = json.loads(json.dumps(asdict(root)))
    with pytest.raises(ValueError, match="limit"):
        priors.joint_root_realization_from_dict(
            payload, decisions=decisions, limits=priors.CausalLimits(parameters=16)
        )


@pytest.mark.parametrize(("seed", "base"), [(0, "uniform"), (1, "gaussian")])
@pytest.mark.parametrize("vector", ["uniform", "polynomial"])
def test_saved_covariance_root_retains_both_base_and_weight_laws(
    seed: int, base: str, vector: str
) -> None:
    root, decisions = _root(function="lin", vector=vector, seed=seed)
    assert root.plan.covariance_base == base
    assert root.plan.weights is not None
    assert root.plan.weights.family == vector
    restored = priors.joint_root_realization_from_dict(
        json.loads(json.dumps(asdict(root))), decisions=decisions
    )
    assert restored == root
