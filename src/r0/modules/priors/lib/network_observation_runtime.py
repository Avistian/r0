"""Explicit target identity aliases preserve pinned source feature/target overlap."""

from typing import TYPE_CHECKING

from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_innovation_compile import compile_network_innovations
from r0.modules.priors.lib.network_observation_request import observation_source

if TYPE_CHECKING:
    from r0.modules.priors.lib.network_observation_sampling import SampledNetworkObservations


def materialize_network_observations(
    receipt: "SampledNetworkObservations", task: LinearRecipe
) -> FunctionalCausalRecipe:
    receipt.validate_replay()
    if type(task) is not LinearRecipe:
        raise ValueError("observation runtime requires linear task controls")
    expected = task.classes if task.task_type == "multiclass" else 1
    if task.features != receipt.request.features or expected != receipt.request.outputs:
        raise ValueError("observation task feature/output geometry differs")
    source, plan = observation_source(receipt.request)
    nodes, _ = compile_network_innovations(source, plan)

    def refs(ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((receipt.catalog[i].node, receipt.catalog[i].channel) for i in ids)

    observed, target = refs(receipt.features), refs(receipt.targets)
    overlap = tuple(ref for ref in target if ref in observed)
    if overlap:
        limits = source.limits
        count = len(overlap)
        widths = sum(nodes[index].function.output_width for index, _ in overlap)
        total = sum(n.function.output_width for n in nodes)
        report = admit_functional_graph(
            nodes,
            limits,
            rows=task.rows,
            metadata=(task, nodes, observed, target, limits),
            features=task.features,
            outputs=len(target),
        )
        # Bound alias matrix entries/metadata, channels, source certificates and live rows
        # before constructing any one-hot matrix. Conservative additional custody reserve.
        extra = 4096 * (widths + count) + 8 * task.rows * 32 * (widths + count)
        extra += 512 * count * (2 * total + source.function.input_width + 1)
        if (
            len(nodes) + count > limits.nodes
            or total + count > limits.total_channels
            or report["parameters"] + widths + 128 * count > limits.parameters
            or report["estimated_working_bytes"] + extra > limits.working_bytes
        ):
            raise ValueError("observation target alias geometry or working limit exceeded")
        additions: list[FunctionalCausalNode] = []
        targets: list[tuple[int, int]] = []
        for ref in target:
            if ref not in observed:
                targets.append(ref)
                continue
            index, channel = ref
            targets.append((len(nodes) + len(additions), 0))
            row = tuple(float(i == channel) for i in range(nodes[index].function.output_width))
            additions.append(FunctionalCausalNode((index,), FrozenLinearFunction((row,))))
        nodes, target = (*nodes, *additions), tuple(targets)
    return FunctionalCausalRecipe(task, nodes, observed, target, source.limits)
