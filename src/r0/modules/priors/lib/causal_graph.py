"""Replayable topology draws, separate from structural equations and readouts."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class CausalGraphPrior:
    family: str = "forward"
    nodes: int = 16
    components: int = 1
    edge_probability: float = 0.25
    layer_widths: tuple[int, ...] = ()
    version: str = "causal-graph-v1"

    def __post_init__(self) -> None:
        if self.version != "causal-graph-v1" or self.family not in ("forward", "layered"):
            raise ValueError("unsupported causal graph version or family")
        if (
            type(self.nodes) is not int
            or type(self.components) is not int
            or not 1 <= self.components <= self.nodes < 2**63
        ):
            raise ValueError("graph sizes must satisfy 1 <= components <= nodes < 2**63")
        if type(self.edge_probability) not in (int, float) or not (
            math.isfinite(self.edge_probability) and 0 <= self.edge_probability <= 1
        ):
            raise ValueError("edge probability must be finite and in [0, 1]")
        if type(self.layer_widths) is not tuple or any(
            type(width) is not int or width < 1 for width in self.layer_widths
        ):
            raise ValueError("layer widths must be a tuple of positive integers")
        if self.family == "layered":
            if sum(self.layer_widths) * self.components != self.nodes:
                raise ValueError("layer widths repeated per component must match node count")
        elif self.layer_widths:
            raise ValueError("layer widths apply only to layered graphs")


@dataclass(frozen=True, slots=True)
class CausalGraph:
    prior: CausalGraphPrior
    parents: tuple[tuple[int, ...], ...]
    seed: int
    task_id: int

    def __post_init__(self) -> None:
        if type(self.prior) is not CausalGraphPrior or (
            type(self.parents) is not tuple or len(self.parents) != self.prior.nodes
        ):
            raise ValueError("graph topology must match its prior node count")
        if any(type(v) is not int or not 0 <= v < 2**63 for v in (self.seed, self.task_id)):
            raise ValueError("graph seed and task_id must be integers in [0, 2**63)")
        for node, incoming in enumerate(self.parents):
            if (
                type(incoming) is not tuple
                or any(type(p) is not int or not 0 <= p < node for p in incoming)
                or tuple(sorted(set(incoming))) != incoming
            ):
                raise ValueError("graph parents must be unique sorted earlier node indices")


def sample_causal_graph(
    prior: CausalGraphPrior,
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits | None = None,
) -> CausalGraph:
    """Sample once; preserve disconnected draws without conditioning on usefulness."""
    if type(prior) is not CausalGraphPrior:
        raise ValueError("graph sampling requires CausalGraphPrior")
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    limits = CausalLimits() if limits is None else limits
    if type(limits) is not CausalLimits:
        raise ValueError("graph sampling limits must use CausalLimits")
    if prior.nodes > min(limits.nodes, limits.total_channels):
        raise ValueError("causal graph node count limit exceeded")
    rng = np.random.default_rng(int(record_fingerprint((asdict(prior), seed, task_id)), 16))
    parents: list[tuple[int, ...]] = []
    edge_count = 0
    for component in range(prior.components):
        start = prior.nodes * component // prior.components
        stop = prior.nodes * (component + 1) // prior.components
        layer_starts = [start]
        for width in prior.layer_widths:
            layer_starts.append(layer_starts[-1] + width)
        for node in range(start, stop):
            candidates = range(start, node)
            if prior.family == "layered":
                layer = next(i for i, end in enumerate(layer_starts[1:]) if node < end)
                candidates = (
                    range(layer_starts[layer - 1], layer_starts[layer]) if layer > 0 else range(0)
                )
            incoming = tuple(
                candidates[int(i)]
                for i in np.flatnonzero(rng.random(len(candidates)) < prior.edge_probability)
            )
            edge_count += len(incoming)
            if 2 * prior.nodes + edge_count > limits.parameters:
                raise ValueError("causal graph minimum parameter limit exceeded")
            if 128 * prior.nodes + 64 * edge_count > limits.working_bytes:
                raise ValueError("estimated causal graph working memory exceeds declared limit")
            parents.append(incoming)
    return CausalGraph(prior, tuple(parents), seed, task_id)
