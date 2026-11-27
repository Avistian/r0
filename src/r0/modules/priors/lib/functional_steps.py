"""Replay functional archives for original-row equation inspection."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_decoding import functional_recipe_from_dict
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng, functional_values
from r0.modules.priors.lib.functional_target_calibration import target_calibration_from_dict
from r0.modules.priors.lib.representatives import record_fingerprint


def functional_steps(
    record: dict[str, Any],
    card: dict[str, Any],
    indices: Sequence[int],
    add: Callable[..., None],
    *,
    limits: CausalLimits | None = None,
) -> torch.Tensor:
    limits = CausalLimits() if limits is None else limits
    raw_recipe = record["recipe"]
    if raw_recipe.get("version") == "network-observation-recipe-v1":
        from r0.modules.priors.lib.network_observation_recipe_decoding import (
            network_observation_recipe_from_dict,
        )
        from r0.modules.priors.lib.representatives import encode

        envelope = network_observation_recipe_from_dict(raw_recipe, limits=limits)
        recipe = envelope.runtime
        if (
            encode(card.get("observation_selection")) != encode(envelope.selection_card(recipe))
            or encode(card.get("observed")) != encode(recipe.observed)
            or encode(card.get("target")) != encode(recipe.target)
            or encode(card.get("nodes")) != encode([asdict(n) for n in recipe.nodes])
        ):
            raise ValueError("observation inspection metadata differs from retained selection")
    elif raw_recipe.get("version") == "sampled-geometry-graph-v1":
        from r0.modules.priors.lib.geometry_graph_decoding import geometry_graph_from_dict

        recipe = geometry_graph_from_dict(raw_recipe, limits=limits).runtime
    elif raw_recipe.get("version") == "sampled-topology-graph-v1":
        from r0.modules.priors.lib.topology_graph_decoding import topology_graph_from_dict

        recipe = topology_graph_from_dict(raw_recipe, limits=limits).runtime
    elif raw_recipe.get("version") == "sampled-parent-graph-v1":
        from r0.modules.priors.lib.parent_graph_decoding import parent_graph_from_dict

        recipe = parent_graph_from_dict(raw_recipe, limits=limits).runtime
    elif raw_recipe.get("version") == "sampled-functional-graph-v1":
        from r0.modules.priors.lib.sampled_graph_decoding import sampled_graph_from_dict

        recipe = sampled_graph_from_dict(raw_recipe, limits=limits).runtime
    else:
        recipe = functional_recipe_from_dict(raw_recipe, limits=limits)
    values: list[torch.Tensor] = []
    functional_values(
        recipe,
        lambda *keys: functional_rng(record["seed"], record["task_id"], *keys),
        "cpu",
        add=add,
        indices=indices,
        readouts=values,
    )
    scores = torch.stack([values[i][list(indices), j] for i, j in recipe.target], dim=1)
    parameters: dict[str, Any] = {"target": recipe.target}
    if "observation_selection" in card:
        parameters["observation_selection"] = card["observation_selection"]
    if recipe.target_calibration_rows:
        calibration = target_calibration_from_dict(card.get("calibration"), len(recipe.target))
        if (
            calibration.rows != recipe.target_calibration_rows
            or calibration.prefix_fingerprint
            != record_fingerprint(([asdict(n) for n in recipe.nodes], recipe.target))
        ):
            raise ValueError("retained target calibration differs from its frozen graph")
        scores = calibration.apply(scores)
        parameters["calibration"] = card["calibration"]
    add(
        "signal",
        "Graph target channels",
        list(dict.fromkeys(f"readout-{i}" for i, _ in recipe.target)),
        "(declared target channels - fixed center) / fixed scale"
        if recipe.target_calibration_rows
        else "select declared target channels",
        scores,
        parameters,
    )
    return scores
