"""Explicit typed drafts retain the complete archived base inside guarded execution."""

import json
from dataclasses import MISSING, asdict, fields, is_dataclass
from typing import Any, cast

import numpy as np

from r0.modules import priors as p


def _identity(value: Any) -> int:
    if type(value) is str and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("typed policy identities require exact nonnegative decimal integers")
    return value


def construct_typed_policy(
    record: dict[str, Any], draft: dict[str, Any], *, limits: p.CausalLimits
) -> p.SampledTypedInterventionPolicy:
    controls = dict(draft)
    raw_streams = controls.pop("streams", None)
    if type(raw_streams) is not list or len(cast(list[Any], raw_streams)) != 3:
        raise ValueError("typed policy requires three exact stream seeds")
    seeds = tuple(_identity(seed) for seed in cast(list[Any], raw_streams))
    if any(key in controls for key in ("base", "limits", "version")):
        raise ValueError("typed policy drafts cannot replace the retained base or caller limits")
    base = record["recipe"]
    runtime = base
    while "nodes" not in runtime:
        runtime = runtime.get("runtime", runtime.get("recipe", runtime.get("graph")))
        if type(runtime) is not dict:
            raise ValueError("typed policy requires a retained causal graph")
        runtime = cast(dict[str, Any], runtime)
    version = runtime["version"]
    target = runtime["target"]
    labels = (
        [target["node"]]
        if version == "converted-causal-v1"
        else [target[0]]
        if version in ("native-causal-v1", "native-vector-causal-v1")
        else sorted({node for node, _ in target})
    )
    defaults = {
        field.name: asdict(field.default)
        if is_dataclass(field.default) and not isinstance(field.default, type)
        else field.default
        for field in fields(p.TypedInterventionPolicyRequest)
        if field.default is not MISSING
    }
    defaults.update(
        base=base,
        label_nodes=labels,
        limits=asdict(limits),
        reference_seed=record["seed"],
        reference_task_id=record["task_id"],
    )
    if controls.get("bounds_lifecycle") == "source-reference-population":
        defaults["reference_rows"] = runtime["task"]["rows"]
    for key in ("reference_seed", "reference_task_id", "population"):
        if key in controls:
            controls[key] = _identity(controls[key])
    for key, kind in (
        ("locations", p.InterventionPolicyLocation),
        ("parameter_laws", p.InterventionParameterLaw),
    ):
        if key in controls:
            if type(controls[key]) is not list:
                raise ValueError("typed policy locations and laws require arrays")
            normalized: list[dict[str, Any]] = []
            for item in cast(list[Any], controls[key]):
                if type(item) is not dict:
                    raise ValueError("typed policy location/law entries must be objects")
                values = dict(cast(dict[str, Any], item))
                for field in ("channels", "probabilities", "path", "choices"):
                    if field in values and type(values[field]) is list:
                        values[field] = tuple(cast(list[Any], values[field]))
                normalized.append(asdict(kind(**values)))
            controls[key] = normalized
    # Complete request admission occurs before any original sampled/fitted replay.
    request = p.typed_intervention_policy_request_from_dict(
        json.loads(json.dumps({**defaults, **controls}, allow_nan=False)), limits=limits
    )
    streams = [np.random.default_rng(seed) for seed in seeds]
    result = p.sample_typed_intervention_policy(
        request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
    )
    p.recipe_from_dict(asdict(result), limits=limits)
    return result
