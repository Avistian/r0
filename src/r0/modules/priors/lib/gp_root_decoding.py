"""Whole retained GP/DAG binding admission before either source is replayed."""

from dataclasses import asdict, fields, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import (
    gp_column_population_from_dict,
    gp_fields,
    gp_payload_equal,
    gp_population_metadata,
)
from r0.modules.priors.lib.gp_column_metadata import gp_sequence
from r0.modules.priors.lib.gp_root_dags import GPRootBinding, GPRootDAGRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview

_DEFAULT_LIMITS = CausalLimits()


def preflight_gp_root_dag(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS):
    from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
    from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
    from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.native_causal import NativeCausalRecipe
    from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe

    observation_metadata_cost(data, limits)
    try:
        raw = gp_fields(data, GPRootDAGRecipe)
        if raw["version"] != "gp-root-dag-v1":
            raise ValueError("unsupported GP root DAG version")
        cap = CausalLimits(**gp_fields(raw["limits"], CausalLimits))
        if any(getattr(cap, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
            raise ValueError("GP root DAG declared limits exceed caller limits")
        request = gp_population_metadata(raw["source"], limits=cap)
        preflight_calibration_custody(raw["base"], cap)
        preview = recipe_from_dict(
            authored_semantic_preview(recursive_authored_preview(raw["base"])), limits=cap
        )
        if type(preview) not in (
            FunctionalCausalRecipe,
            NativeCausalRecipe,
            NativeVectorRecipe,
            ConvertedCausalRecipe,
        ):
            raise ValueError("GP root DAG requires an authored functional or typed runtime")
        assert isinstance(
            preview,
            (FunctionalCausalRecipe, NativeCausalRecipe, NativeVectorRecipe, ConvertedCausalRecipe),
        )
        gp_sequence(raw["bindings"])
        bindings: list[GPRootBinding] = []
        for item in raw["bindings"]:
            binding = gp_fields(item, GPRootBinding)
            gp_sequence(binding["columns"])
            bindings.append(
                GPRootBinding(binding["node"], tuple(binding["columns"]), binding["version"])
            )
        if (
            not bindings
            or len({b.node for b in bindings}) != len(bindings)
            or preview.task.rows != request.rows
        ):
            raise ValueError("GP root DAG row identity or unique binding count differs")
        for b in bindings:
            if b.node >= len(preview.nodes):
                raise ValueError("GP root node outside runtime")
            n = preview.nodes[b.node]
            equation = n if isinstance(n, FunctionalCausalNode) else n.equation
            if (
                equation.parents
                or equation.root is None
                or len(b.columns) != equation.function.input_width
                or any(i >= request.columns for i in b.columns)
            ):
                raise ValueError("GP root binding geometry differs")
        for ceiling in (cap, request.limits, preview.limits):
            source = request.admit(ceiling)
            graph = replace(preview, limits=ceiling).admit()
            count, storage = observation_metadata_cost(data, ceiling)
            if (
                count + source["parameters"] + graph["parameters"] > ceiling.parameters
                or 4 * storage
                + 4 * (source["estimated_working_bytes"] + graph["estimated_working_bytes"])
                > ceiling.working_bytes
            ):
                raise ValueError("GP root DAG aggregate source/graph metadata exceeds limits")
        return cap, preview, tuple(bindings), request
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed GP root DAG archive") from exc


def gp_root_dag_from_dict(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS) -> GPRootDAGRecipe:
    from typing import cast

    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.gp_root_dags import GPRootRuntime

    cap, _, bindings, _ = preflight_gp_root_dag(data, limits=limits)
    raw = gp_fields(data, GPRootDAGRecipe)
    base = cast(GPRootRuntime, recipe_from_dict(raw["base"], limits=cap))
    result = GPRootDAGRecipe(
        gp_column_population_from_dict(raw["source"], limits=cap),
        base,
        bindings,
        cap,
        raw["version"],
    )
    if not gp_payload_equal(asdict(result), data):
        raise ValueError("GP root DAG differs from complete canonical replay")
    return result
