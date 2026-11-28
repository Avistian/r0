"""Per-task ensemble draws retain each member's complete construction receipt."""

import json
import math
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.generation.lib.tree_ensemble_prior import (
    ensemble_prior_from_dict,
    sample_ensemble_prior,
)
from r0.modules.priors import (
    CausalLimits,
    FunctionalCausalRecipe,
    sample_tree_extension,
    score_ensemble_recipe,
    tree_extension_request_from_dict,
)


def construct_tree_ensemble(
    construction: GraphConstruction, *, limits: CausalLimits, seed: int, task_id: int
) -> tuple[FunctionalCausalRecipe, dict[str, Any]]:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("ensemble construction requires exact integer identities")
    controls = json.loads(construction.controls_json)
    if (
        set(controls)
        not in (
            {"topology", "members", "weights", "reduction"},
            {"topology", "members", "weights", "reduction", "prior"},
        )
        or controls["topology"] != "tree-ensemble"
    ):
        raise ValueError("ensemble construction fields differ")
    if type(controls["members"]) is not list or not 1 <= len(controls["members"]) <= 128:
        raise ValueError("ensemble requires 1..128 member requests")
    raw_weights = controls["weights"]
    if type(raw_weights) is not list:
        raise ValueError("ensemble weights must be an array")
    raw_weights = cast(list[Any], raw_weights)
    if len(raw_weights) not in (0, len(controls["members"])):
        raise ValueError("ensemble weights must match member count or be empty")
    if any(type(w) not in (float, int) or not math.isfinite(w) for w in raw_weights):
        raise ValueError("ensemble weights must be finite numbers")
    weights = tuple(cast(list[float], raw_weights))
    reduction = controls["reduction"]
    if reduction not in ("mean", "sum"):
        raise ValueError("ensemble reduction must be mean or sum")
    if reduction == "mean" and weights:
        try:
            total = math.fsum(weights)
        except OverflowError as exc:
            raise ValueError("ensemble weight sum overflows") from exc
        if total == 0 or not math.isfinite(total):
            raise ValueError("ensemble mean requires nonzero finite weight sum")
    requests = tuple(
        tree_extension_request_from_dict(raw, limits=limits) for raw in controls["members"]
    )
    if any(r.task != requests[0].task or r.root != requests[0].root for r in requests):
        raise ValueError("ensemble member requests must share task and root controls")
    prior = ensemble_prior_from_dict(controls["prior"]) if "prior" in controls else None
    if prior is not None:
        if weights:
            raise ValueError("sampled ensemble prior requires empty authored weights")
        prior.validate_pool(len(requests), reduction)
    # Admit the full declared support before any size, selection or weight draw.
    costs = [3 + 3 * 2**r.max_depth for r in requests]
    worst = sum(costs) if prior is None else prior.max_members * max(costs)
    if 2 + worst > limits.nodes:
        raise ValueError("ensemble worst-case node count exceeds limits")
    selection = None
    if prior is not None:
        selection, weights = sample_ensemble_prior(
            prior, len(requests), construction_seed=construction.seed, seed=seed, task_id=task_id
        )
        requests = tuple(requests[index] for index in selection)
    sampled = tuple(
        sample_tree_extension(
            request,
            limits=limits,
            rng=np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([0x454E534D, construction.seed, seed, task_id, index])
                )
            ),
        )
        for index, request in enumerate(requests)
    )
    recipe = score_ensemble_recipe(
        tuple(s.recipe for s in sampled), weights=weights, reduction=reduction, limits=limits
    )
    receipt: dict[str, Any] = {
        "version": "r0-tree-ensemble-construction-v1"
        if prior is None
        else "r0-tree-ensemble-construction-v2",
        "construction_seed": construction.seed,
        "seed": seed,
        "task_id": task_id,
        "weights": list(weights),
        "reduction": reduction,
        "members": [asdict(s) for s in sampled],
    }
    if prior is not None:
        receipt["prior"] = asdict(prior)
        receipt["selected_templates"] = list(selection or ())
    return recipe, json.loads(json.dumps(receipt, allow_nan=False))
