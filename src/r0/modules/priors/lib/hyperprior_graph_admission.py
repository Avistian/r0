"""Whole worst-size topology/template graph admission without replaying any stream."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_function_admission import admit_recursive_payloads
from r0.modules.priors.lib.recursive_task_preflight import admit_recursive_tasks
from r0.modules.priors.lib.topology_hyperprior import TopologyHyperpriorRequest
from r0.modules.priors.lib.topology_hyperprior_admission import admit_topology_hyperprior


def hyperprior_readouts(task: LinearRecipe, count: int, width: int):
    outputs = task.classes if task.task_type == "multiclass" else 1
    if count * width < task.features + outputs:
        raise ValueError("minimum topology size cannot hold distinct features and targets")
    return (
        tuple(divmod(i, width) for i in range(task.features)),
        tuple(divmod(i, width) for i in range(count * width - outputs, count * width)),
    )


def admit_hyperprior_graph(
    task: LinearRecipe,
    topology: TopologyHyperpriorRequest,
    root: dict[str, Any],
    edge: dict[str, Any],
    width: int,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
    *,
    retained: Any = None,
) -> dict[str, int]:
    """Sum all-root and chain bounds: this dominates every admitted forest mixture.

    Metadata is charged per node, even when immutable templates share Python objects.
    Both synthetic graphs are only dictionaries; sampled function/root restoration is
    deferred until all geometry, replay and topology reservations pass together.
    """
    top = admit_topology_hyperprior(topology, limits, raw_limits)
    n = topology.maximum_nodes
    if type(width) is not int or not 1 <= width <= limits.node_channels:
        raise ValueError("topology graph width exceeds channel limit")
    if n * width > limits.total_channels:
        raise ValueError("topology graph worst total width exceeds limit")
    hyperprior_readouts(task, topology.minimum_nodes, width)
    numeric, storage = observation_metadata_cost((task, topology, root, edge), limits)
    if retained is not None:
        retained_numbers, retained_storage = observation_metadata_cost(retained, limits)
        if (
            retained_numbers > top["parameters"]
            or 8 * retained_storage > top["estimated_working_bytes"]
        ):
            raise ValueError("retained topology exceeds its preallocated reservation")
    parameters = top["parameters"] + 4 * n * (numeric + 64)
    working = top["estimated_working_bytes"] + 16 * n * (storage + 1024)
    if parameters > limits.parameters or working > limits.working_bytes:
        raise ValueError("topology graph worst retained metadata exceeds limits")
    observed, target = hyperprior_readouts(task, n, width)
    graphs: list[dict[str, Any]] = []
    for chain in (False, True):
        nodes = [
            {**(edge if chain and i else root), "parents": (i - 1,) if chain and i else ()}
            for i in range(n)
        ]
        graphs.append(
            dict(
                task=asdict(task),
                nodes=nodes,
                observed=observed,
                target=target,
                limits=asdict(limits),
                version="functional-causal-v1",
            )
        )
    envelopes: list[dict[str, Any]] = []
    convolutions: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if type(value) is dict:
            value = cast(dict[str, Any], value)
            if value.get("version") == "frozen-recursive-forest-function-v1":
                envelopes.append(value)
            if value.get("version") == "sampled-feature-conv1d-r0-v1":
                convolutions.append(value)
            for item in value.values():
                collect(item)
        elif type(value) in (tuple, list):
            for item in value:
                collect(item)

    collect(graphs)
    replay = admit_recursive_payloads(
        envelopes,
        storage=storage,
        numbers=numeric,
        limits=limits,
        graphs=[],
        convolutions=convolutions,
    )
    reports = admit_recursive_tasks(
        graphs,
        storage=storage * n,
        numbers=numeric * n,
        replay_work=replay["estimated_working_bytes"],
        limits=limits,
    )
    parameters += sum(r["parameters"] for r in reports)
    working += sum(r["estimated_working_bytes"] for r in reports)
    if parameters > limits.parameters or working > limits.working_bytes:
        raise ValueError("aggregate topology graph construction/emission budget exceeded")
    return dict(
        maximum_nodes=n,
        total_channels=n * width,
        parameters=parameters,
        estimated_working_bytes=working,
    )
