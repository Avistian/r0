"""Complete graph comparison custody is reserved before nested RNG restoration."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.population_graph_change_preview import population_change_preview
from r0.modules.priors.lib.population_graph_semantic_preflight import preflight_population_change
from r0.modules.priors.lib.recursive_forest_decoding import sampled_recursive_forest_from_dict
from r0.modules.priors.lib.recursive_population_decoding import (
    complete_population_record,
    population_raw_cost,
)
from r0.modules.priors.lib.recursive_population_graph_decoding import (
    admit_population_graph_request,
    recursive_population_graph_from_dict,
)
from r0.modules.priors.lib.recursive_population_graph_intervention import (
    RecursivePopulationGraphIntervention,
)
from r0.modules.priors.lib.recursive_population_graph_records import RecursivePopulationGraph
from r0.modules.priors.lib.recursive_task_preflight import admit_recursive_tasks
from r0.modules.priors.lib.representatives import encode


def preflight_population_graph_intervention(data: Any, *, limits: CausalLimits):
    admit_function_metadata(data, limits)
    raw = complete_population_record(data, RecursivePopulationGraphIntervention)
    baseline = complete_population_record(raw["base"], RecursivePopulationGraph)
    change = complete_population_record(raw["change"], CausalInterventionRecipe)
    request = baseline["request"]
    if encode(change["base"]) != encode(request["base"]):
        raise ValueError("population counterfactual must retain the exact original DAG")
    numbers, storage = population_raw_cost(data, limits)
    # Two independent live graph/receipt/capture reservations plus outer metadata.
    if numbers >= limits.parameters or 8 * storage >= limits.working_bytes:
        raise ValueError("population comparison metadata exceeds caller limits")
    nested = replace(
        limits,
        parameters=(limits.parameters - numbers) // 2,
        working_bytes=(limits.working_bytes - 8 * storage) // 2,
    )
    _, base_raw, _, _, effective, plans = admit_population_graph_request(request, nested)
    after = population_change_preview(base_raw, change["interventions"])
    disabled = {
        c["node"]
        for c in change["interventions"]
        if c.get("version") == "soft-causal-intervention-v1" and c.get("function") is not None
    }
    geometry = {
        p.node: (
            p.forest.input_width,
            p.forest.classes or p.forest.output_width,
            p.forest.classes,
        )
        for p in plans
        if p.node not in disabled
    }
    after_report = admit_recursive_tasks(
        [after],
        storage=storage,
        numbers=numbers,
        replay_work=0,
        limits=effective,
        retain_node_stages=True,
        population_geometry=geometry,
    )
    preflight_population_change(raw, baseline, change, limits=effective)
    return raw, baseline, change, effective, after_report[0]


def population_graph_intervention_from_dict(
    data: Any, *, limits: CausalLimits
) -> RecursivePopulationGraphIntervention:
    try:
        raw, baseline, change, effective, _ = preflight_population_graph_intervention(
            data, limits=limits
        )
        # Geometry and outer semantics pass before either receipt decoder.
        # The whole archive is already charged above, with a separate nested
        # workspace reserved for each population. Passing that reduced workspace
        # to the standalone decoder would charge the baseline archive a second
        # time. Decode under the original caller ceiling after aggregate admission.
        base = recursive_population_graph_from_dict(baseline, limits=limits)
        decoded_change = intervention_from_dict(
            {**change, "base": {**change["base"], "limits": asdict(effective)}}, limits=effective
        )
        decoded_change = replace(decoded_change, base=base.request.base)
        samples = tuple(
            sampled_recursive_forest_from_dict(s, limits=effective)
            for s in fitted_array(raw["samples"], len(base.bindings))
        )
        inputs: list[tuple[int, tuple[tuple[float, ...], ...] | None]] = []
        for item in fitted_array(raw["inputs"], len(base.bindings)):
            pair = fitted_array(item, 2)
            if len(pair) != 2:
                raise ValueError("population input binding must be a node and matrix")
            node, matrix = pair
            binding = next((b for b in base.bindings if b.node == node), None)
            if type(node) is not int or binding is None:
                raise ValueError("population input binding node differs")
            rows = (
                None
                if matrix is None
                else tuple(
                    tuple(fitted_array(row, binding.sample.plan.input_width))
                    for row in fitted_array(matrix, base.task.rows)
                )
            )
            inputs.append((node, rows))
        result = RecursivePopulationGraphIntervention(
            base, decoded_change, samples, tuple(inputs), raw["version"]
        )
        result.validate_replay(limits=limits)
        if encode(asdict(result)) != encode(data):
            raise ValueError("population comparison differs from complete canonical archive")
        return result
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError, IndexError) as exc:
        raise ValueError("invalid complete population DAG intervention archive") from exc
