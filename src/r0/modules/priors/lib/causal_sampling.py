"""Sample fixed structural mechanisms on an explicitly selected graph prior."""

import math

import numpy as np

from r0.modules.priors.lib.causal import CausalNode, CausalRecipe
from r0.modules.priors.lib.causal_graph import CausalGraphPrior, sample_causal_graph
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint


def sample_causal_recipe(
    graph: CausalGraphPrior,
    *,
    task: LinearRecipe,
    seed: int,
    task_id: int,
    widths: tuple[int, ...] = (),
    limits: CausalLimits | None = None,
) -> CausalRecipe:
    """Return a frozen editable recipe; sampling never reads generated task rows."""
    limits = CausalLimits() if limits is None else limits
    topology = sample_causal_graph(graph, seed=seed, task_id=task_id, limits=limits)
    if type(widths) is not tuple or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError("node widths must be a tuple of positive integers")
    widths = widths or (1,) * graph.nodes
    if len(widths) != graph.nodes:
        raise ValueError("one width is required per causal node")
    outputs = task.classes if task.task_type == "multiclass" else 1
    if sum(widths) < task.features + outputs:
        raise ValueError("not enough distinct channels for observed features and targets")
    limits.admit_dimensions(
        widths,
        topology.parents,
        rows=task.rows,
        calibration_rows=0,
        features=task.features,
        outputs=outputs,
    )

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint(("gaussian-affine-v1", seed, task_id, keys)), 16)
        )

    nodes: list[CausalNode] = []
    for i, (width, parents) in enumerate(zip(widths, topology.parents, strict=True)):
        incoming = sum(widths[p] for p in parents)
        weights = rng("weights", i).normal(size=(incoming, width)) / math.sqrt(max(1, incoming))
        nodes.append(
            CausalNode(
                parents=parents,
                weights=tuple(tuple(float(v) for v in row) for row in weights),
                bias=tuple(float(v) for v in rng("bias", i).normal(0, 0.25, size=width)),
                noise_scale=(0.1 if parents else 1.0,) * width,
            )
        )
    channels = tuple((i, j) for i, width in enumerate(widths) for j in range(width))
    chosen = [channels[int(i)] for i in rng("readouts").permutation(len(channels))]
    return CausalRecipe(
        task=task,
        nodes=tuple(nodes),
        observed=tuple(chosen[: task.features]),
        target=tuple(chosen[task.features : task.features + outputs]),
        limits=limits,
        topology=topology,
        construction="gaussian-affine-v1",
    )
