"""Pure graph operations driven by a retained or live raw-draw adapter."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.structural_graph_prior import (
    AttachmentGraphPrior,
    MotifGraphPrior,
    StructuralGraphPrior,
)

type Draw = Callable[
    [str, str, tuple[int, ...], tuple[tuple[str, Any], ...]], tuple[float | int, ...]
]
type Edges = set[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class GraphRewrite:
    step: int
    kind: str
    subject: tuple[int, ...]
    removed: tuple[tuple[int, int], ...]
    added: tuple[tuple[int, int], ...]
    reason: str

    def __post_init__(self) -> None:
        if (
            type(self.step) is not int
            or self.step < 0
            or self.kind not in ("redirect", "path", "bypass")
            or self.reason
            not in ("applied", "node-has-no-through-path", "no-edge", "no-admissible-replacement")
            or type(self.subject) is not tuple
            or any(type(n) is not int or n < 0 for n in self.subject)
        ):
            raise ValueError("graph rewrite requires canonical step, subject and disposition")
        for edges in (self.removed, self.added):
            if (
                type(edges) is not tuple
                or any(
                    type(edge) is not tuple
                    or len(edge) != 2
                    or any(type(n) is not int for n in edge)
                    or not 0 <= edge[0] < edge[1]
                    for edge in edges
                )
                or tuple(sorted(set(edges))) != edges
            ):
                raise ValueError("graph rewrite edge deltas must be canonical forward edges")


def _index(draw: Draw, role: str, count: int) -> int:
    return int(draw(role, "integers", (1,), (("high", count),))[0])


def _attachment(prior: AttachmentGraphPrior, draw: Draw) -> Edges:
    edges: Edges = set()
    degrees = [0] * prior.nodes
    for component in range(prior.components):
        start = prior.nodes * component // prior.components
        stop = prior.nodes * (component + 1) // prior.components
        for node in range(start, stop):
            candidates = list(range(start, node))
            for slot in range(min(prior.edges_per_node, len(candidates))):
                logits = [
                    prior.power * math.log(degrees[p] + prior.attractiveness) for p in candidates
                ]
                if any(not math.isfinite(logit) for logit in logits):
                    raise ValueError("nonfinite attachment weights; graph was not retried")
                maximum = max(logits)
                weights = [math.exp(logit - maximum) for logit in logits]
                total = math.fsum(weights)
                probabilities = tuple(weight / total for weight in weights)
                selected = int(
                    draw(
                        f"attachment:{node}:{slot}",
                        "choice",
                        (1,),
                        (("count", len(candidates)), ("probabilities", probabilities)),
                    )[0]
                )
                parent = candidates.pop(selected)
                edges.add((parent, node))
                degrees[parent] += 1
                degrees[node] += 1
    return edges


_MOTIFS: dict[str, tuple[tuple[int, int], ...]] = {
    "chain": ((0, 1), (1, 2)),
    "fork": ((0, 1), (0, 2)),
    "collider": ((0, 2), (1, 2)),
}


def _motifs(prior: MotifGraphPrior, draw: Draw) -> Edges:
    edges: Edges = set()
    count = prior.components
    for depth in range(prior.depth):
        expanded: Edges = set()
        roots: list[tuple[int, ...]] = []
        sinks: list[tuple[int, ...]] = []
        for node in range(count):
            choice = _index(draw, f"motif:{depth}:{node}", len(prior.motifs))
            motif = _MOTIFS[prior.motifs[choice]]
            start = 3 * node
            expanded.update((start + a, start + b) for a, b in motif)
            roots.append(tuple(start + i for i in range(3) if all(b != i for _, b in motif)))
            sinks.append(tuple(start + i for i in range(3) if all(a != i for a, _ in motif)))
        for a, b in sorted(edges):
            expanded.update((source, target) for source in sinks[a] for target in roots[b])
        edges = expanded
        count *= 3
    return edges


def _rewrite(prior: MotifGraphPrior, edges: Edges, step: int, draw: Draw) -> GraphRewrite:
    kind = prior.rewrite_kinds[_index(draw, f"rewrite:{step}:kind", len(prior.rewrite_kinds))]
    before = edges.copy()
    subject: tuple[int, ...] = ()
    reason = "applied"
    if kind == "bypass":
        node = _index(draw, f"rewrite:{step}:node", prior.nodes)
        subject = (node,)
        incoming = [a for a, b in sorted(edges) if b == node]
        outgoing = [b for a, b in sorted(edges) if a == node]
        if not incoming or not outgoing:
            reason = "node-has-no-through-path"
        else:
            edges.difference_update((a, node) for a in incoming)
            edges.difference_update((node, b) for b in outgoing)
            edges.update((a, b) for a in incoming for b in outgoing)
    elif not edges:
        reason = "no-edge"
    else:
        a, b = sorted(edges)[_index(draw, f"rewrite:{step}:edge", len(edges))]
        subject = (a, b)
        if kind == "redirect":
            component_start = (b // (3**prior.depth)) * (3**prior.depth)
            candidates = [p for p in range(component_start, b) if p != a and (p, b) not in edges]
        else:
            candidates = [p for p in range(a + 1, b) if (a, p) not in edges and (p, b) not in edges]
        if not candidates:
            reason = "no-admissible-replacement"
        else:
            selected = candidates[_index(draw, f"rewrite:{step}:replacement", len(candidates))]
            subject = (a, b, selected)
            edges.remove((a, b))
            edges.add((selected, b))
            if kind == "path":
                edges.add((a, selected))
    return GraphRewrite(
        step, kind, subject, tuple(sorted(before - edges)), tuple(sorted(edges - before)), reason
    )


def construct_structural_graph(
    prior: StructuralGraphPrior, draw: Draw
) -> tuple[tuple[tuple[int, ...], ...], tuple[GraphRewrite, ...]]:
    if isinstance(prior, AttachmentGraphPrior):
        edges = _attachment(prior, draw)
        rewrites: tuple[GraphRewrite, ...] = ()
    else:
        edges = _motifs(prior, draw)
        rewrites = tuple(_rewrite(prior, edges, step, draw) for step in range(prior.rewrite_steps))
    parents: list[list[int]] = [[] for _ in range(prior.nodes)]
    for parent, node in sorted(edges):
        parents[node].append(parent)
    return tuple(tuple(incoming) for incoming in parents), rewrites
