"""Strict whole-task preview before topology or nested mechanism snapshot restore."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import decode_admitted_function
from r0.modules.priors.lib.functional_causal_decoding import functional_node_from_dict
from r0.modules.priors.lib.hyperprior_graph import HyperpriorGraph
from r0.modules.priors.lib.hyperprior_graph_admission import admit_hyperprior_graph
from r0.modules.priors.lib.hyperprior_graph_request import HyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.topology_hyperprior_decoding import (
    complete_topology_record,
    sampled_topology_hyperprior_from_dict,
    topology_hyperprior_from_dict,
)

_DEFAULT_LIMITS = CausalLimits()


def hyperprior_graph_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> HyperpriorGraph:
    if type(limits) is not CausalLimits:
        raise ValueError("topology graph decoding requires caller-owned limits")
    observation_metadata_cost(data, limits)
    try:
        outer = complete_topology_record(data, HyperpriorGraph)
        raw = complete_topology_record(outer["request"], HyperpriorGraphRequest)
        declared = CausalLimits(**complete_topology_record(raw["limits"], CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(declared, f.name), getattr(limits, f.name))
                for f in fields(CausalLimits)
            }
        )
        observation_metadata_cost(data, effective)
        raw_task = complete_topology_record(raw["task"], LinearRecipe)
        if type(raw_task["class_offsets"]) not in (tuple, list):
            raise ValueError("topology task class offsets require an explicit array")
        raw_task["class_offsets"] = tuple(raw_task["class_offsets"])
        task = LinearRecipe(**raw_task)
        prior = topology_hyperprior_from_dict(raw["topology"])
        caps = RawDrawLimits(**complete_topology_record(raw["raw_limits"], RawDrawLimits))
        # Authored geometry preview strips only nested sampled receipts, not source equations.
        root_function = decode_admitted_function(
            recursive_authored_preview(raw["root_template"]["function"]), limits=effective
        )
        admit_hyperprior_graph(
            task,
            prior,
            raw["root_template"],
            raw["edge_template"],
            root_function.output_width,
            effective,
            caps,
            retained=outer["topology"],
        )
        from r0.modules.priors.lib.hyperprior_graph_preflight import preflight_hyperprior_graph

        preflight_hyperprior_graph(outer, raw, task, prior, declared, caps, limits=effective)
        request = HyperpriorGraphRequest(
            task,
            prior,
            functional_node_from_dict(raw["root_template"], limits=effective),
            functional_node_from_dict(raw["edge_template"], limits=effective),
            declared,
            caps,
            raw["layout"],
            raw["mechanism_sharing"],
            raw["version"],
        )
        topology = sampled_topology_hyperprior_from_dict(outer["topology"], limits=effective)
        result = HyperpriorGraph(request, topology, outer["version"])
        if encode(asdict(result)) != encode(data):
            raise ValueError("topology graph archive differs from complete exact construction")
        return result
    except (
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        RecursionError,
        IndexError,
    ) as error:
        raise ValueError("malformed topology graph archive") from error
