"""Whole GP intervention lineage and future population reservation before replay."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_population_metadata_cost import (
    population_metadata_cost,
    record_cost,
    retained_cost,
)
from r0.modules.priors.lib.gp_root_decoding import preflight_gp_root_dag
from r0.modules.priors.lib.gp_root_intervention import GPRootIntervention
from r0.modules.priors.lib.gp_root_query_metadata import (
    GPQueryMetadata,
    admit_query_envelope,
    query_source_metadata,
)
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.representatives import record_fingerprint


def intervention_source_metadata(data: Any, limits: CausalLimits) -> GPQueryMetadata:
    if type(data) is dict and cast(dict[str, Any], data).get("version") == "gp-root-dag-v1":
        data = cast(dict[str, Any], data)
        cap, base, bindings, request = preflight_gp_root_dag(data, limits=limits)
        source = data["source"]
        return GPQueryMetadata(
            data,
            cap,
            base,
            bindings,
            request,
            frozenset({(source["seed"], source["task_id"])}),
            ((request, base),),
        )
    return query_source_metadata(data, limits=limits)


def intervention_plan(base: Any, change: Any, *, limits: CausalLimits):
    if type(limits) is not CausalLimits:
        raise ValueError("GP intervention requires explicit caller limits")
    retained_cost((base, change), limits)
    metadata = intervention_source_metadata(base, limits)
    raw = gp_fields(change, CausalInterventionRecipe)
    if not gp_payload_equal(raw["base"], metadata.current["base"]):
        raise ValueError("GP intervention must retain its exact original authored base graph")
    preflight_calibration_custody(change, metadata.limits)
    preview = intervention_from_dict(
        authored_semantic_preview(recursive_authored_preview(change)), limits=metadata.limits
    )
    runtime = preview.effective_runtime()
    for binding in metadata.bindings:
        node = runtime.nodes[binding.node]
        equation = node if isinstance(node, FunctionalCausalNode) else node.equation
        if equation.function.input_width != len(binding.columns):
            raise ValueError("GP intervention changed the bound root input geometry")
        for item in raw["interventions"]:
            if item["node"] == binding.node and item.get("root") is not None:
                raise ValueError("GP-bound root laws require an explicit GP population edit")
    original = metadata.current["source"]
    seed, task_id = original["seed"], original["task_id"]
    if preview.mode == "population":
        seed = (
            int(
                record_fingerprint(
                    (
                        "gp-root-intervention-population-v1",
                        seed,
                        task_id,
                        preview.population,
                    )
                ),
                16,
            )
            % 2**63
        )
        if (seed, task_id) in metadata.identities:
            raise ValueError("GP intervention population identity collides with retained lineage")
    cost = record_cost(
        GPRootIntervention,
        {
            "base": retained_cost(base, limits),
            "change": retained_cost(change, limits),
            "population": population_metadata_cost(metadata.request),
            "limits": retained_cost(asdict(metadata.limits), limits),
            "version": retained_cost("gp-root-intervention-v1", limits),
        },
    )
    admit_query_envelope(
        cost, metadata.populations + ((metadata.request, runtime),), metadata.limits
    )
    return metadata, preview, seed, task_id
