"""Source-qualified observation sampling and receipt inspection inside the guard."""

import json
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_observations(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.NetworkObservationRecipe:
    from r0.modules.generation.lib.construction import construct_graph

    if set(fields) != {"source", "task", "observations", "selection_seed"}:
        raise ValueError("observations require source, task, policy and separate selection seed")
    source_controls = cast(dict[str, Any], fields["source"])
    if type(source_controls) is not dict or source_controls.get("topology") not in (
        "network",
        "network-authored",
        "network-hyperprior",
        "source-root-network",
        "prototype-root-network",
        "network-innovations",
    ):
        raise ValueError("observations require an explicit supported standalone source")
    selection_seed = fields["selection_seed"]
    if type(selection_seed) is str and selection_seed.isdecimal():
        selection_seed = int(selection_seed)
    if type(selection_seed) is not int or not 0 <= selection_seed < 2**63:
        raise ValueError("observation selection seed must be an exact integer in [0, 2**63)")
    task = priors.recipe_from_dict(fields["task"], limits=limits)
    if type(task) is not priors.LinearRecipe:
        raise ValueError("observation task requires linear task adapter controls")
    source = construct_graph(source_controls, seed=seed, limits=limits)
    if not isinstance(source, (priors.NetworkRecipe, priors.NetworkInnovationRecipe)):
        raise ValueError("observation source must retain a standalone network")
    policy = cast(dict[str, Any], fields["observations"])
    if type(policy) is not dict or set(policy) != {
        "profile",
        "selection",
        "target",
        "feature_order",
        "raw_limits",
    }:
        raise ValueError("observation policy requires all explicit fields")
    policy = dict(policy)
    policy["raw_limits"] = priors.RawDrawLimits(**policy["raw_limits"])
    request = priors.NetworkObservationRequest(
        source,
        task.features,
        outputs=task.classes if task.task_type == "multiclass" else 1,
        **policy,
    )
    rng = np.random.default_rng(np.random.SeedSequence([selection_seed, 0x4F425356]))
    return priors.NetworkObservationRecipe(
        task, priors.sample_network_observations(request, rng=rng)
    )


def observation_editor(data: dict[str, Any]) -> dict[str, Any]:
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("observation editor requires bounded output bytes")
    if len(json.dumps(data, allow_nan=False).encode()) > maximum:
        raise ValueError("observation editor request exceeds byte cap")
    recipe = priors.recipe_from_dict(
        data["recipe"], limits=priors.CausalLimits(**data.get("causal_limits", {}))
    )
    draft = data.get("validation") == "observation-draft"
    if draft:
        if type(recipe) is not priors.FunctionalCausalRecipe:
            raise ValueError("observation edits require an authored functional graph")
        runtime = recipe
        selection = None
    else:
        if not isinstance(recipe, priors.NetworkObservationRecipe):
            raise ValueError("observation editor requires its complete retained envelope")
        runtime = recipe.runtime
        selection = recipe.selection_card(runtime)
    result = {
        "kind": "mechanism-edit",
        "construction": recipe.version,
        "editable_recipe": asdict(runtime),
        "geometry": None,
        "observation_selection": selection,
        "notice": (
            "Selected internal/final network channels, with explicit target aliases for overlap. "
            "The receipt and original source remain in the before archive. Editing this derived "
            "graph creates linked authored equations; it does not resample source selections or "
            "claim source-profile provenance for the edited graph. Use interventions for explicit "
            "state/readout changes. Variable-width, skip, residual and gated selection policies "
            "remain required extensions, not silently substituted architectures."
        ),
        "nodes": [
            {
                "index": i,
                "parents": node.parents,
                "width": node.function.output_width,
                "function": node.function.version,
                "observed": [c for n, c in runtime.observed if n == i],
                "target": [c for n, c in runtime.target if n == i],
            }
            for i, node in enumerate(runtime.nodes)
        ],
    }
    if len(json.dumps(result, allow_nan=False).encode()) > maximum:
        raise ValueError("observation editor response exceeds byte cap")
    return result
