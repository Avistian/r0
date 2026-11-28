"""Aggregate observation source/custody, construction and emission before replay."""

from dataclasses import fields, is_dataclass
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_innovation_recipe import with_network_innovations
from r0.modules.priors.lib.network_observation_request import NetworkObservationRequest
from r0.modules.priors.lib.network_observation_runtime import compile_observation_runtime
from r0.modules.priors.lib.observation_source_preview import (
    field,
    observation_source_preview,
    values,
)
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


def observation_metadata_cost(data: Any, limits: CausalLimits) -> tuple[int, int]:
    numbers = storage = seen = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal numbers, storage, seen
        seen += 1
        if depth > 32 or seen > 16 * limits.parameters + 64 * limits.nodes:
            raise ValueError("observation envelope metadata traversal limit exceeded")
        if type(value) in (int, float, bool):
            numbers += 1
            storage += 64
        elif type(value) is str:
            if len(value) > limits.working_bytes:
                raise ValueError("observation envelope text limit exceeded")
            storage += len(value.encode("utf-8"))
        elif value is None:
            pass
        elif type(value) in (tuple, list, dict) or is_dataclass(value):
            storage += 32
        else:
            raise ValueError("observation envelope requires frozen records or JSON")
        if numbers > limits.parameters or storage > limits.working_bytes:
            raise ValueError("observation envelope metadata limit exceeded")
        if type(value) is dict:
            for key, item in cast(dict[Any, Any], value).items():
                if type(key) is not str:
                    raise ValueError("observation envelope keys must be strings")
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif type(value) in (tuple, list):
            for item in cast(tuple[Any, ...], value):
                visit(item, depth + 1)
        elif is_dataclass(value) and not isinstance(value, type):
            for f in fields(value):
                visit(f.name, depth + 1)
                visit(getattr(value, f.name), depth + 1)

    visit(data, 0)
    return numbers, storage


def admit_observation_envelope(
    task: LinearRecipe,
    receipt: Any,
    limits: CausalLimits,
    *,
    evaluation_rows: int | None = None,
) -> dict[str, int]:
    if type(task) is not LinearRecipe or type(limits) is not CausalLimits:
        raise ValueError("observation envelope requires task controls and caller limits")
    if evaluation_rows is not None and (
        type(evaluation_rows) is not int or not 1 <= evaluation_rows < 2**63
    ):
        raise ValueError("observation evaluation rows must be positive bounded integers")
    numbers, storage = observation_metadata_cost((task, receipt), limits)
    reserve = 8 * storage + 16384
    if reserve > limits.working_bytes or 4 * numbers > limits.parameters:
        raise ValueError("observation envelope aggregate retained storage limit exceeded")
    raw_request = field(receipt, "request")
    source, plan, replay_parameters, replay = observation_source_preview(
        field(raw_request, "source"), limits
    )
    raw_caps = field(raw_request, "raw_limits")
    caps = (
        raw_caps
        if type(raw_caps) is RawDrawLimits
        else RawDrawLimits(**values(raw_caps, RawDrawLimits))
    )
    kwargs = values(raw_request, NetworkObservationRequest)
    kwargs["source"] = (
        with_network_innovations(source, plan)
        if field(field(raw_request, "source"), "version") == "standalone-network-innovations-v1"
        else source
    )
    kwargs["raw_limits"] = caps
    request = NetworkObservationRequest(**kwargs)
    features, targets = tuple(field(receipt, "features")), tuple(field(receipt, "targets"))
    runtime = compile_observation_runtime(request, features, targets, task)
    rows = max(task.rows, evaluation_rows or 0)
    report = admit_functional_graph(
        runtime.nodes,
        limits,
        rows=rows,
        metadata=(task, runtime.nodes, runtime.observed, runtime.target),
        features=task.features,
        outputs=len(runtime.target),
    )
    report["parameters"] += 4 * numbers + replay_parameters
    report["estimated_working_bytes"] += reserve + replay + 2 * caps.state_bytes
    if (
        report["parameters"] > limits.parameters
        or report["estimated_working_bytes"] > limits.working_bytes
    ):
        raise ValueError("observation envelope aggregate construction/emission limit exceeded")
    return report
