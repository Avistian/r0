"""Source-profiled network sampling runs only in the guarded construction worker."""

from dataclasses import asdict
from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_hyperprior_network(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.NetworkRecipe:
    if set(fields) != {"task", "root", "hyperprior"}:
        raise ValueError("profiled network construction requires task, root and hyperprior")
    if type(fields["hyperprior"]) is not dict:
        raise ValueError("profiled network hyperprior must be a JSON object")
    profile = dict(cast(dict[str, Any], fields["hyperprior"]))
    if type(profile.get("template")) is not dict:
        raise ValueError("profiled network template must be a JSON object")
    template = dict(cast(dict[str, Any], profile["template"]))
    declared = template.pop("limits", asdict(limits))
    if declared != asdict(limits):
        raise ValueError("profiled network limits must match caller exploration limits")
    profile["template"] = {**template, "limits": asdict(limits)}
    request = priors.hyperprior_request_from_dict(profile, limits=limits)
    task = priors.recipe_from_dict(fields["task"])
    if type(task) is not priors.LinearRecipe:
        raise ValueError("profiled network requires ordinary task controls")
    outputs = task.classes if task.task_type == "multiclass" else 1
    if request.template.input_width != task.features or request.template.widths != (outputs,):
        raise ValueError("profiled network fixed input/output geometry differs from task")
    hyper, named, weights = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    network = priors.sample_hyperprior_network(
        request, hyper_rng=hyper, named_rng=named, raw_rng=weights
    )
    result = priors.recipe_from_dict(
        {
            "version": "standalone-network-v1",
            "task": asdict(task),
            "network": asdict(network),
            "root": fields["root"],
            "limits": asdict(limits),
        },
        limits=limits,
    )
    if not isinstance(result, priors.NetworkRecipe):
        raise ValueError("profiled network construction requires its standalone recipe")
    return result
