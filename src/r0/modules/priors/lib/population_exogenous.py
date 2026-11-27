"""Support/query draw separation, scattered back to the declared original row order."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_root_base import draw_root_base


def population_exogenous(
    node: FunctionalCausalNode,
    index: int,
    rows: int,
    rng: Callable[..., np.random.Generator],
    partitions: tuple[tuple[int, ...], ...],
    *,
    root: bool,
) -> NDArray[np.float64]:
    """Each split owns a keyed stream; query size cannot move support draw offsets."""
    width = node.root.width if root and node.root else node.function.output_width
    result = np.empty((rows, width), dtype=np.float64)
    for label, indices in zip(("support", "query"), partitions, strict=True):

        def stream(*keys: object, population_label: str = label) -> np.random.Generator:
            return rng("fit-population-v1", population_label, *keys)

        if root:
            assert node.root is not None
            values = np.asarray(
                draw_root_base(node.root, len(indices), stream("root-base", index)).values
            )
        else:
            values = sample_exogenous(node.exogenous, len(indices), width, stream, index)
        result[list(indices)] = values
    return result
