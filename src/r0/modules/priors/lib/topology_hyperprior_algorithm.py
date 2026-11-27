"""Literal one-parent redirection, with balanced contiguous component allocation."""

from collections.abc import Callable

from r0.modules.priors.lib.causal_graph import CausalGraphPrior
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.topology_hyperprior_admission import topology_edge_specs


def hyperprior_components(nodes: int, count: int) -> tuple[tuple[int, int], ...]:
    return tuple((nodes * c // count, nodes * (c + 1) // count) for c in range(count))


def hyperprior_parents(
    topology: CausalGraphPrior, draw: Callable[[RawDrawSpec], float | int]
) -> tuple[tuple[int, ...], ...]:
    parents: list[tuple[int, ...]] = []
    for start, stop in hyperprior_components(topology.nodes, topology.components):
        parents.append(())
        for node in range(start + 1, stop):
            select, redirect = topology_edge_specs(node, start)
            offset, uniform = draw(select), draw(redirect)
            if (
                type(offset) is not int
                or not 0 <= offset < node - start
                or type(uniform) is not float
                or not 0 <= uniform < 1
            ):
                raise ValueError("redirection draw lies outside its exact support")
            selected = start + offset
            predecessor = parents[selected]
            parents.append(
                (
                    predecessor[0]
                    if uniform < topology.redirection_probability and predecessor
                    else selected,
                )
            )
    return tuple(parents)
