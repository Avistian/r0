"""One metadata-only reference/comparison/raw planner for typed sampling and decode."""

from dataclasses import fields, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import decode_admitted_function
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.intervention_policy_laws import numeric_leaves, parameter_group
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.recursive_function_admission import admit_recursive_payloads
from r0.modules.priors.lib.recursive_task_preflight import (
    admit_recursive_tasks,
    recursive_authored_preview,
)
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan


def _frozen(value: Any) -> Any:
    if type(value) is dict:
        return tuple((key, _frozen(item)) for key, item in cast(dict[str, Any], value).items())
    if type(value) in (list, tuple):
        return tuple(_frozen(v) for v in cast(list[Any] | tuple[Any, ...], value))
    return value


def admit_typed_record(request: dict[str, Any], limits: CausalLimits) -> dict[str, int]:
    """Preview authored geometry only; retain and charge every original wrapper."""
    admit_function_metadata(request, limits)
    try:
        return _plan(request, limits)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed typed policy admission metadata") from error


def _plan(request: dict[str, Any], limits: CausalLimits) -> dict[str, int]:
    declared = CausalLimits(**request["limits"])
    limits = CausalLimits(
        **{f.name: min(getattr(limits, f.name), getattr(declared, f.name)) for f in fields(limits)}
    )
    raw_limits = RawDrawLimits(**request["raw_limits"])
    ceiling = min(limits.working_bytes, raw_limits.working_bytes)
    if 6 * raw_limits.state_bytes > ceiling:
        raise ValueError("typed policy initial state reservation exceeds declared limits")
    base = cast(dict[str, Any], request["base"])
    graph: dict[str, Any] = base
    while "nodes" not in graph:
        nested = graph.get("runtime", graph.get("recipe", graph.get("graph")))
        if type(nested) is not dict:
            raise ValueError("typed policy requires retained causal runtime metadata")
        graph = cast(dict[str, Any], nested)
    if graph["version"] in (
        "support-population-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
    ):
        raise ValueError("population/support refitting policies require a separate lifecycle")
    graph_limits = CausalLimits(**graph["limits"])
    effective = CausalLimits(
        **{
            f.name: min(getattr(limits, f.name), getattr(graph_limits, f.name))
            for f in fields(limits)
        }
    )
    nodes = graph["nodes"]
    equations = [node.get("equation", node) for node in nodes]
    functions = [
        decode_admitted_function(recursive_authored_preview(n["function"]), limits=effective)
        for n in equations
    ]
    widths = sum(f.input_width + f.output_width for f in functions)
    locations, laws = request["locations"], request["parameter_laws"]
    eligible = request["eligibility"] or [
        i for i in range(len(nodes)) if i not in request["label_nodes"]
    ]
    rows = request["reference_rows"] if request["kind"] == "hard" else graph["task"]["rows"]
    if type(rows) is not int or not 1 <= rows <= 20000:
        raise ValueError("typed reference rows must be a bounded exact integer")
    roots = sum(
        function_record_cost(_frozen(n["root"]), limits)[0]
        for n in equations
        if n["root"] is not None
    )
    numbers = 64 * rows * (widths + roots + len(locations)) if request["kind"] == "hard" else 0
    bounded_noise = sum(
        f.output_width
        for n, f in zip(equations, functions, strict=True)
        if n["exogenous"]["inputs"]["bound"] is not None and request["kind"] == "hard"
    )
    numbers += 256 * max(rows, 256) * bounded_noise
    numbers += sum(
        numeric_leaves(parameter_group(equations[law["node"]]["function"], tuple(law["path"]))[2])
        for law in laws
    )
    numbers += len(eligible) + 1 + sum(len(a["channels"]) or 1 for a in locations)
    draws = 64 * (len(nodes) + widths + len(locations)) + len(laws) + 2 + 256 * bounded_noise
    _, storage = function_record_cost(_frozen(request), limits)
    graph_numbers, graph_storage = function_record_cost(_frozen(graph), effective)
    forest: list[dict[str, Any]] = []
    convolution: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if type(value) is dict:
            raw = cast(dict[str, Any], value)
            if raw.get("version") == "frozen-recursive-forest-function-v1":
                forest.append(raw)
            if raw.get("version") == "sampled-feature-conv1d-r0-v1":
                convolution.append(raw)
            for child in raw.values():
                visit(child)
        elif type(value) in (tuple, list):
            for child in cast(tuple[Any, ...] | list[Any], value):
                visit(child)

    visit(graph)
    replay = admit_recursive_payloads(
        forest, storage=0, numbers=0, limits=effective, graphs=[], convolutions=convolution
    )
    fit_work = 0
    fit_rows = 0
    envelope_rows = 0
    # Retained sampled graph constructors also replay their original calibration
    # population. Do not substitute the smaller emitted task population.
    envelope = base
    construction_work = 0
    construction_parameters = 0
    while "nodes" not in envelope:
        source_request = envelope.get("request", {})
        if source_request.get("version") in (
            "typed-topology-request-r0-v1",
            "typed-parent-topology-request-r0-v1",
        ):
            from r0.modules.priors.lib.typed_topology_decoding import (
                typed_topology_request_from_dict,
            )

            support = typed_topology_request_from_dict(source_request, limits=limits).admit()
            construction_work += support["estimated_working_bytes"]
            construction_parameters += support["parameters"]
        envelope_rows = max(envelope_rows, source_request.get("calibration_rows", 0))
        raw_cap = source_request.get("raw_limits")
        if raw_cap is not None:
            # Same staged snapshot reservation as joint/native construction.
            fit_work += 32 * min(RawDrawLimits(**raw_cap).state_bytes, 1024**2)
        envelope = cast(
            dict[str, Any],
            envelope.get("runtime", envelope.get("recipe", envelope.get("graph"))),
        )
    for fit in graph.get("fits", ()):
        realization = fit["realization"]
        plan = realization["plan"]
        fit_rows = max(fit_rows, plan["rows"])
        if plan["version"] == "independent-xgboost-hist-fit-plan-v1":
            fit_work += XGBoostFitPlan(**plan).admit(
                effective, RawDrawLimits(**realization["raw_limits"])
            ) + 12 * len(realization["model_json"])
    view = graph
    if graph["version"] in (
        "fitted-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "xgboost-fitted-functional-causal-v1",
    ):
        view = {**graph, "version": "functional-causal-v1"}
    # Three live graph views: complete reference, original comparison, effective
    # comparison. Use the largest row population, including retained ancestor fits.
    count = max(graph["task"]["rows"], rows, fit_rows)
    view = {
        **view,
        "task": {**graph["task"], "rows": count},
        "calibration_rows": max(view.get("calibration_rows", 0), envelope_rows),
    }
    report = admit_recursive_tasks(
        [view],
        storage=graph_storage,
        numbers=graph_numbers + replay["parameters"],
        replay_work=replay["estimated_working_bytes"] + fit_work,
        limits=effective,
    )[0]
    # The shared graph planner already includes each typed encoded state, native
    # emission and converter workspace. Reserve pair difference buffers as well.
    channels = sum(f.input_width + 12 * f.output_width for f in functions)
    if graph["version"].startswith("native-"):
        channels += (
            sum(
                len(d.get("vocabulary", ()))
                for node in nodes
                for d in node.get("domains", (node.get("domain", {}),))
            )
            * 4
        )
    working = (
        storage
        + construction_work
        + 3 * report["estimated_working_bytes"]
        + 32 * graph["task"]["rows"] * channels
        + 128 * numbers
        + 2048 * draws
        + 6 * raw_limits.state_bytes
    )
    if (
        numbers > raw_limits.numbers
        or draws > raw_limits.draws
        or working > ceiling
        or construction_parameters > 0
        and construction_parameters + numbers + graph_numbers > limits.parameters
    ):
        raise ValueError("typed policy aggregate reference/comparison/raw admission exceeds limits")
    RawDrawLedger(limits=replace(raw_limits, working_bytes=ceiling)).admit_specs(
        (RawDrawSpec(("typed-policy",), "aggregate-reserve", "normal", (numbers,)),)
    )

    return {
        "parameters": construction_parameters + numbers + graph_numbers,
        "estimated_working_bytes": working,
    }
