"""Standalone network construction stays inside the guarded generation worker."""

from dataclasses import asdict
from typing import Any

import numpy as np

from r0.modules import priors


def construct_network(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits, authored: bool
) -> priors.NetworkRecipe:
    fields = dict(fields)
    task = fields.pop("task")
    root = fields.pop("root", None)
    if authored:
        network = fields.pop("network")
        if network.get("version") != "frozen-structural-network-v1":
            raise ValueError("authored network controls require a frozen architecture")
    else:
        raw = dict(fields.pop("network_prior"))
        declared = raw.pop("limits", asdict(limits))
        if declared != asdict(limits):
            raise ValueError("network construction limits must match caller exploration limits")
        if "activations" in raw:
            raw["activations"] = tuple(priors.CausalActivation(**a) for a in raw["activations"])
        if "raw_limits" in raw:
            raw["raw_limits"] = priors.RawDrawLimits(**raw["raw_limits"])
        request = priors.NetworkSamplingRequest(**raw, limits=limits)
        named, weights = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
        network = asdict(priors.sample_network(request, named_rng=named, raw_rng=weights))
    if fields:
        raise ValueError("unknown standalone network construction controls")
    result = priors.recipe_from_dict(
        {
            "version": "standalone-network-v1",
            "task": task,
            "network": network,
            "root": root,
            "limits": asdict(limits),
        },
        limits=limits,
    )
    if not isinstance(result, priors.NetworkRecipe):
        raise ValueError("standalone network construction requires its recipe")
    return result
