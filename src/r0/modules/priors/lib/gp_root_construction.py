"""Complete authored GP-root construction admission before finite population draws."""

from dataclasses import replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.gp_column_decoding import gp_column_request_from_dict, gp_fields
from r0.modules.priors.lib.gp_column_metadata import gp_sequence
from r0.modules.priors.lib.gp_column_population import realize_gp_columns
from r0.modules.priors.lib.gp_root_dags import GPRootBinding, GPRootDAGRecipe
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview


def construct_gp_root_dag(
    data: dict[str, Any],
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits,
) -> GPRootDAGRecipe:
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.gp_population_metadata_cost import gp_root_dag_metadata_cost

    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("GP root construction requires exact nonnegative identities")
    observation_metadata_cost(data, limits)
    if set(data) != {"request", "base", "bindings"}:
        raise ValueError("GP root construction requires complete population, graph and bindings")
    request = gp_column_request_from_dict(data["request"], limits=limits)
    preflight_calibration_custody(data["base"], limits)
    preview = recipe_from_dict(
        authored_semantic_preview(recursive_authored_preview(data["base"])),
        limits=limits,
    )
    if type(preview) not in (
        FunctionalCausalRecipe,
        NativeCausalRecipe,
        NativeVectorRecipe,
        ConvertedCausalRecipe,
    ):
        raise ValueError("GP roots require an authored functional or typed graph")
    assert isinstance(
        preview,
        (FunctionalCausalRecipe, NativeCausalRecipe, NativeVectorRecipe, ConvertedCausalRecipe),
    )
    gp_sequence(data["bindings"])
    bindings: list[GPRootBinding] = []
    for item in data["bindings"]:
        raw = gp_fields(item, GPRootBinding)
        gp_sequence(raw["columns"])
        bindings.append(GPRootBinding(raw["node"], tuple(raw["columns"]), raw["version"]))
    frozen = tuple(bindings)
    if (
        not bindings
        or len({b.node for b in bindings}) != len(bindings)
        or request.rows != preview.task.rows
    ):
        raise ValueError("GP root construction requires unique root bindings and matching rows")
    for binding in bindings:
        if binding.node >= len(preview.nodes):
            raise ValueError("GP root binding exceeds graph")
        node = preview.nodes[binding.node]
        equation = node if isinstance(node, FunctionalCausalNode) else node.equation
        if (
            equation.parents
            or equation.root is None
            or len(binding.columns) != equation.function.input_width
            or any(column >= request.columns for column in binding.columns)
        ):
            raise ValueError("GP root binding geometry differs from population or graph")

    for cap in (limits, request.limits, preview.limits):
        gp = request.admit(cap)
        graph = replace(preview, limits=cap).admit()
        future = gp_root_dag_metadata_cost(request, preview, frozen, limits=limits)
        # Include submitted custody alongside the canonical preview: omitted
        # defaults and sampled receipts must not undercount future graph metadata.
        raw_count, raw_storage = observation_metadata_cost(data["base"], cap)
        count = future["numbers"] + raw_count
        storage = future["storage"] + raw_storage
        if (
            count + gp["parameters"] + graph["parameters"] > cap.parameters
            or 4 * storage + 4 * (gp["estimated_working_bytes"] + graph["estimated_working_bytes"])
            > cap.working_bytes
        ):
            raise ValueError("GP root construction aggregate source/graph metadata exceeds limits")
    base = recipe_from_dict(data["base"], limits=limits)
    assert isinstance(
        base,
        (FunctionalCausalRecipe, NativeCausalRecipe, NativeVectorRecipe, ConvertedCausalRecipe),
    )
    source = realize_gp_columns(request, seed=seed, task_id=task_id)
    return GPRootDAGRecipe(source, base, frozen, limits)
