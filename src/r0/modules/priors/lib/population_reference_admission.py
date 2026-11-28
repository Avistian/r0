"""Array-free original-row reference controls and complete source/working-set reservation."""

import math
from dataclasses import asdict, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalRecipe
from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.population_graph_intervention_decoding import (
    admit_population_graph_comparison,
)
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.population_reference_execution import (
    observed_schema,
    overlay_reference,
    reference_clamps,
)
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.recursive_population_ancestors import ANCESTOR_TYPES, PopulationAncestors
from r0.modules.priors.lib.recursive_population_decoding import (
    complete_population_record,
    population_raw_cost,
)
from r0.modules.priors.lib.recursive_population_graph_decoding import admit_population_graph_request
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_population_records import population_identity


def admit_population_reference(
    source: Any,
    *,
    row: int,
    feature: int,
    values: tuple[float, ...],
    capture_location: str,
    limits: CausalLimits,
):
    from r0.modules.priors.lib.generate import recipe_from_dict

    if type(limits) is not CausalLimits:
        raise ValueError("population reference requires explicit causal limits")
    admit_function_metadata(source, limits)
    if type(source) is not dict:
        raise ValueError("population reference source must be a complete recipe object")
    source = cast(dict[str, Any], source)
    version = source.get("version")
    if version not in (
        "recursive-population-graph-v1",
        "recursive-population-graph-intervention-v1",
    ):
        raise ValueError("population reference requires an original or retained changed population")
    baseline = source if version == "recursive-population-graph-v1" else source.get("base")
    baseline = complete_population_record(baseline, RecursivePopulationGraph)
    population_identity(baseline["seed"], baseline["task_id"])
    request = cast(dict[str, Any], baseline["request"])
    if type(request) is not dict or type(request.get("base")) is not dict:
        raise ValueError("population reference requires its complete request")
    task = cast(dict[str, Any], request["base"].get("task"))
    if type(task) is not dict:
        raise ValueError("population reference requires its task controls")
    rows, features = task.get("rows"), task.get("features")
    if (
        type(rows) is not int
        or type(features) is not int
        or type(row) is not int
        or not 0 <= row < rows
        or type(feature) is not int
        or not 0 <= feature < features
        or type(values) is not tuple
        or not 1 <= len(values) <= 129
        or any(type(v) is not float or not math.isfinite(v) for v in values)
        or rows * len(values) > 20000
        or type(capture_location) is not str
        or capture_location not in ("capture", "representation_and_capture")
    ):
        raise ValueError("population reference requires bounded original row/feature/grid controls")
    declared = CausalLimits(**request["limits"])
    effective = recursive_effective_limits(declared, limits)
    # Sequential original-shape execution: one full population at a time plus
    # retained grid output and original anchor. Bound total row work separately.
    cells = len(values) * (features + 4 * max(1, task.get("classes", 0)) + 8)
    numbers, storage = population_raw_cost(source, effective)
    numbers += 8 * cells
    storage = 8 * storage + 2048 * cells
    if numbers >= effective.parameters or storage >= effective.working_bytes:
        raise ValueError("population reference source/grid aggregate exceeds limits")
    nested = replace(
        effective,
        parameters=effective.parameters - numbers,
        working_bytes=effective.working_bytes - storage,
    )
    if version == "recursive-population-graph-intervention-v1":
        admit_population_graph_comparison(source, limits=nested)
    else:
        admit_population_graph_request(request, nested)
    raw_base = request["base"]
    preview = authored_semantic_preview(raw_base)
    preview["limits"] = asdict(nested)
    runtime = recipe_from_dict(preview, limits=nested)
    if type(runtime) not in ANCESTOR_TYPES:
        raise ValueError("population reference requires one of the four declared graph families")
    runtime = cast(PopulationAncestors, runtime)
    prior = None
    if version == "recursive-population-graph-intervention-v1":
        raw_change = authored_semantic_preview(source["change"])
        raw_change["base"] = preview
        prior = intervention_from_dict(raw_change, limits=nested)
        runtime = prior.effective_runtime()
    schema = observed_schema(runtime)
    has_captures = isinstance(runtime, ConvertedCausalRecipe) and any(
        ref.capture is not None for ref in runtime.observed
    )
    if not has_captures and capture_location != "capture":
        raise ValueError("population reference has no observed converter representation to change")
    # Reject category geometry and ambiguous cuts before any construction replay.
    neutral = [0.0] * features
    if prior and isinstance(runtime, ConvertedCausalRecipe):
        for i, ref in enumerate(runtime.observed):
            for cut in prior.interventions:
                if (
                    i != feature
                    and isinstance(cut, ConverterHardIntervention)
                    and (cut.node, cut.capture) == (ref.node, ref.capture)
                ):
                    neutral[i] = float(schema[i].vocabulary.index(cut.category))
    for value in values:
        dummy = tuple(value if i == feature else neutral[i] for i in range(features))
        overlay_reference(runtime, prior, reference_clamps(runtime, dummy, capture_location))
    return effective, nested, schema
