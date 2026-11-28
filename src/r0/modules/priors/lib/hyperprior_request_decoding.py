"""Complete request validation precedes restoration of retained sampled templates."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_decoding import functional_node_from_dict
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.hyperprior_graph_preflight import authored_topology_root_preview
from r0.modules.priors.lib.hyperprior_graph_request import HyperpriorGraphRequest
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches
from r0.modules.priors.lib.topology_hyperprior_decoding import (
    complete_topology_record,
    topology_hyperprior_from_dict,
)


def hyperprior_graph_request_from_dict(
    data: Any, *, limits: CausalLimits
) -> HyperpriorGraphRequest:
    observation_metadata_cost(data, limits)
    raw = complete_topology_record(data, HyperpriorGraphRequest)
    declared = CausalLimits(**complete_topology_record(raw["limits"], CausalLimits))
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("topology request limits exceed caller limits")
    task_raw = complete_topology_record(raw["task"], LinearRecipe)
    task_raw["class_offsets"] = tuple(task_raw["class_offsets"])
    task = LinearRecipe(**task_raw)
    prior = topology_hyperprior_from_dict(raw["topology"])
    caps = RawDrawLimits(**complete_topology_record(raw["raw_limits"], RawDrawLimits))
    templates = [
        complete_topology_record(raw[k], FunctionalCausalNode)
        for k in ("root_template", "edge_template")
    ]

    def build(nodes: list[FunctionalCausalNode]) -> HyperpriorGraphRequest:
        return HyperpriorGraphRequest(
            task,
            prior,
            nodes[0],
            nodes[1],
            declared,
            caps,
            raw["layout"],
            raw["mechanism_sharing"],
            raw["version"],
        )

    preview = build(
        [
            functional_node_from_dict(
                authored_topology_root_preview(recursive_authored_preview(node)), limits=declared
            )
            for node in templates
        ]
    )
    for key, value in asdict(preview).items():
        if key not in ("root_template", "edge_template") and not construction_matches(
            value, raw[key]
        ):
            raise ValueError("topology request metadata must retain exact types and values")
    # Charge original receipts as well as discarded authored geometry before replay.
    from r0.modules.priors.lib.hyperprior_graph_admission import admit_hyperprior_graph

    admit_hyperprior_graph(
        task,
        prior,
        templates[0],
        templates[1],
        preview.root_template.function.output_width,
        declared,
        caps,
    )
    from r0.modules.priors.lib.topology_template_preflight import preflight_topology_templates

    preflight_topology_templates(templates, limits=declared, state_bytes=caps.state_bytes)
    result = build([functional_node_from_dict(node, limits=declared) for node in templates])
    if not construction_matches(asdict(result), data):
        raise ValueError("topology request must retain every exact field")
    return result
