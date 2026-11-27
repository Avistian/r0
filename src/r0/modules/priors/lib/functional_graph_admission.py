"""Shared array-free DAG validation and conservative graph workspace admission."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import function_geometry
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_function_admission import function_record_cost


def validate_functional_nodes(
    nodes: tuple[FunctionalCausalNode, ...], limits: CausalLimits
) -> None:
    if type(limits) is not CausalLimits:
        raise ValueError("functional graph requires CausalLimits")
    if type(nodes) is not tuple or not nodes:
        raise ValueError("functional graph requires a nonempty node tuple")
    if len(nodes) > limits.nodes:
        raise ValueError("functional graph node count limit exceeded")
    for index, node in enumerate(nodes):
        if type(node) is not FunctionalCausalNode or any(p >= index for p in node.parents):
            raise ValueError("functional nodes must be in acyclic parent-before-child order")
        if isinstance(node.function, FrozenParentFunction) and (
            not node.parents
            or tuple(c.input_width for c in node.function.children)
            != tuple(nodes[p].function.output_width for p in node.parents)
        ):
            raise ValueError("parent composition slices differ from ordered parent widths")
        if node.parents and node.function.input_width != sum(
            nodes[p].function.output_width for p in node.parents
        ):
            raise ValueError("function input width must match concatenated parent state widths")


def admit_functional_graph(
    nodes: tuple[FunctionalCausalNode, ...],
    limits: CausalLimits,
    *,
    rows: int,
    metadata: tuple[object, ...],
    features: int = 0,
    outputs: int = 0,
    target_calibration: bool = False,
) -> dict[str, int]:
    """Callers validate rows/DAG first and supply their complete retained metadata."""
    parameters, storage = function_record_cost(metadata, limits)
    if target_calibration:
        extra = 3 * outputs + 1
        parameters += extra
        storage += 64 * extra + 1024
        if parameters > limits.parameters:
            raise ValueError("target calibration parameter limit exceeded")
    geometries = [function_geometry(node.function) for node in nodes]
    widest = max(max(widths) for widths, _, _ in geometries)
    total = sum(node.function.output_width for node in nodes)
    if widest > limits.node_channels or total > limits.total_channels:
        raise ValueError("functional graph channel limit exceeded")
    fan_in = max(node.function.input_width for node in nodes)
    roots = max((node.root.width for node in nodes if node.root), default=0)
    rows = max(
        rows,
        max(floor for _, _, floor in geometries),
        256 if any(n.exogenous.inputs.bound is not None for n in nodes) else 0,
    )
    pipeline_work = max((n.pipeline.working_extra_channels for n in nodes if n.pipeline), default=0)
    source_bound = sum(
        (n.root.width if n.root else 0)
        + (
            n.function.output_width
            if any(n.noise_scale) and (n.noise_response.offset or n.noise_response.strength)
            else 0
        )
        for n in nodes
    )
    # State and readout certificates can each retain every preceding source.
    # Charge Python set entries separately from numeric evaluation buffers.
    analysis_bytes = 256 * total * source_bound + 256 * total
    working = (
        8
        * rows
        * (
            6 * total
            + 2 * fan_in
            + 2 * features
            + (8 if target_calibration else 4) * outputs
            + max(work for _, work, _ in geometries)
            + 40 * roots
            + pipeline_work
            + 4
        )
        + analysis_bytes
        + 128 * parameters
        + 2 * (storage - 64 * parameters)
    )
    if working > limits.working_bytes:
        raise ValueError("estimated functional graph working memory exceeds declared limit")
    return {
        "nodes": len(nodes),
        "largest_node_channels": widest,
        "total_channels": total,
        "parameters": parameters,
        "estimated_analysis_bytes": analysis_bytes,
        "estimated_working_bytes": working,
    }
