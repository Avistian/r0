"""Discarded authored templates validate complete graph semantics before source replay."""

from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_decoding import functional_node_from_dict
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.hyperprior_graph import HyperpriorGraph
from r0.modules.priors.lib.hyperprior_graph_request import HyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.topology_hyperprior import TopologyHyperpriorRequest
from r0.modules.priors.lib.topology_hyperprior_decoding import (
    complete_topology_record,
    sampled_topology_hyperprior_metadata,
)


def authored_topology_root_preview(value: Any) -> Any:
    if type(value) is dict:
        raw = cast(dict[str, Any], value)
        if raw.get("version") == "sampled-source-root-base-v1":
            return {
                "law": raw["law"],
                "calibration": raw["calibration"],
                "version": "source-root-base-v1",
            }
        if raw.get("version") == "sampled-prototype-root-v1":
            return authored_topology_root_preview(raw["root"])
        return {key: authored_topology_root_preview(item) for key, item in raw.items()}
    if type(value) in (list, tuple):
        return tuple(
            authored_topology_root_preview(item)
            for item in cast(list[Any] | tuple[Any, ...], value)
        )
    return value


def preflight_hyperprior_graph(
    outer: dict[str, Any],
    request: dict[str, Any],
    task: LinearRecipe,
    prior: TopologyHyperpriorRequest,
    declared: CausalLimits,
    raw_limits: RawDrawLimits,
    *,
    limits: CausalLimits,
) -> None:
    """Original metadata was already charged; this view is never returned as a recipe."""
    topology = sampled_topology_hyperprior_metadata(outer["topology"], limits=limits)
    templates: list[FunctionalCausalNode] = []
    for key in ("root_template", "edge_template"):
        raw = complete_topology_record(request[key], FunctionalCausalNode)
        preview = authored_topology_root_preview(recursive_authored_preview(raw))
        templates.append(functional_node_from_dict(preview, limits=limits))
    preview_request = HyperpriorGraphRequest(
        task,
        prior,
        templates[0],
        templates[1],
        declared,
        raw_limits,
        request["layout"],
        request["mechanism_sharing"],
        request["version"],
    )
    # The same constructors validate versions, template ownership, geometry,
    # request/receipt identity, caps, derived readouts and runtime compatibility.
    HyperpriorGraph(preview_request, topology, outer["version"])
