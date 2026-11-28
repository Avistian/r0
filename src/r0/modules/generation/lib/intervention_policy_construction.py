"""Explicit sampled-policy drafts are realized only inside the guarded pair worker."""

import json
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from r0.modules import priors as p


def construct_intervention_policy(
    record: dict[str, Any], draft: str, *, limits: p.CausalLimits
) -> p.SampledInterventionPolicy | p.SampledTypedInterventionPolicy:
    fields = json.loads(draft)
    if type(fields) is not dict:
        raise ValueError("sampled policy settings require an object")
    fields = dict(cast(dict[str, Any], fields))
    if fields.get("profile") == "explicit-typed-policy-r0-v1":
        from r0.modules.generation.lib.typed_policy_construction import construct_typed_policy

        return construct_typed_policy(record, fields, limits=limits)
    seeds: Any = fields.pop("streams", None)
    if (
        type(seeds) is not list
        or len(cast(list[Any], seeds)) != 3
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in cast(list[Any], seeds))
    ):
        raise ValueError("policy streams require three exact nonnegative integer seeds")
    if any(key in fields for key in ("base", "limits", "version")):
        raise ValueError("policy draft cannot replace its retained base or caller limits")
    if record["recipe"].get("version") not in ("functional-causal-v1", "functional-causal-v2"):
        raise ValueError("sampled policies currently require an authored functional graph")
    # Admission and retained archive reconstruction have distinct roles.
    p.recipe_from_dict(record["recipe"], limits=limits)
    base = p.recipe_from_dict(record["recipe"], limits=p.CausalLimits(**record["recipe"]["limits"]))
    if type(base) is not p.FunctionalCausalRecipe:
        raise ValueError("sampled policies currently require an authored functional graph")
    labels = tuple(sorted({node for node, _ in base.target}))
    fields.setdefault("label_nodes", labels)
    fields.setdefault("reference_seed", record["seed"])
    fields.setdefault("reference_task_id", record["task_id"])
    if fields.get("bounds_lifecycle") == "source-reference-population":
        fields.setdefault("reference_rows", base.task.rows)
    for key in ("label_nodes", "eligibility"):
        if key in fields:
            if type(fields[key]) not in (tuple, list):
                raise ValueError("policy node indices require arrays")
            fields[key] = tuple(fields[key])
    laws: list[p.InterventionParameterLaw] = []
    for raw in fields.pop("parameter_laws", []):
        law = dict(raw)
        for key in ("path", "choices"):
            if key in law:
                law[key] = tuple(law[key])
        laws.append(p.InterventionParameterLaw(**law))
    if "raw_limits" in fields:
        fields["raw_limits"] = p.RawDrawLimits(**fields["raw_limits"])
    request = p.InterventionPolicyRequest(base, parameter_laws=tuple(laws), limits=limits, **fields)
    streams = [np.random.default_rng(seed) for seed in cast(list[int], seeds)]
    result = p.sample_intervention_policy(
        request, selection_rng=streams[0], value_rng=streams[1], calibration_rng=streams[2]
    )
    # The same strict archive path used for reopen must accept the complete result.
    p.recipe_from_dict(asdict(result), limits=limits)
    return result
