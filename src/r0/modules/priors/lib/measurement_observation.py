"""Measurement receipt construction and decoding for observation tasks."""

from dataclasses import asdict
from typing import Any, cast

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.measurement_pipeline import MeasurementStage, draw_measurements
from r0.modules.priors.lib.training_features import TrainingFeature


def measurement_stages_from_dict(data: Any) -> tuple[MeasurementStage, ...]:
    if type(data) not in (tuple, list) or not 1 <= len(data) <= 64:
        raise ValueError("measurement stages require a bounded array")
    result: list[MeasurementStage] = []
    for item in cast(list[Any], data):
        if type(item) is not dict:
            raise ValueError("measurement stage requires an object")
        raw: dict[str, Any] = dict(cast(dict[str, Any], item))
        if not {"kind", "columns"} <= raw.keys() or raw.keys() - {"kind", "columns", "parameters"}:
            raise ValueError("measurement stage fields differ")
        if type(raw["columns"]) not in (list, tuple) or type(raw.get("parameters", ())) not in (
            list,
            tuple,
        ):
            raise ValueError("measurement columns and parameters must be arrays")
        raw["columns"] = tuple(raw["columns"])
        parameters = raw.get("parameters", ())
        if any(type(pair) not in (list, tuple) for pair in parameters):
            raise ValueError("measurement parameters must be pairs")
        raw["parameters"] = tuple(tuple(pair) for pair in parameters)
        result.append(MeasurementStage(**raw))
    return tuple(result)


def measure_observation(
    stages: tuple[MeasurementStage, ...],
    raw: torch.Tensor,
    schema: tuple[TrainingFeature, ...],
    *,
    seed: int,
    source: dict[str, Any],
    limits: CausalLimits,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if (
        type(stages) is not tuple
        or not 1 <= len(stages) <= 64
        or any(type(s) is not MeasurementStage for s in stages)
    ):
        raise ValueError("observation measurements require bounded frozen stages")
    if any(max(s.columns) >= raw.shape[1] for s in stages):
        raise ValueError("measurement column lies outside input schema")
    if schema and any(schema[c].kind != "continuous" for s in stages for c in s.columns):
        raise ValueError("measurement maps currently require declared continuous columns")
    keys = (seed, source["seed"], source["task_id"])
    if any(type(v) is not int or not 0 <= v < 2**63 for v in keys):
        raise ValueError("measurement identities require exact integers")
    sampled = draw_measurements(
        stages,
        np.asarray(raw.detach().numpy(), dtype=np.float64),
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4D454153, *keys]))),
        limits=limits,
    )
    return torch.tensor(sampled.values), {
        "version": sampled.version,
        "stages": [asdict(s) for s in stages],
        "binding": sampled.binding,
        "initial": asdict(sampled.initial),
        "final": asdict(sampled.final),
    }
