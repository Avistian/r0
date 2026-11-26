"""Replayable topology draws, separate from structural equations and readouts."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits


@dataclass(frozen=True, slots=True)
class CausalGraphPrior:
    family: str = "forward"
    nodes: int = 16
    components: int = 1
    edge_probability: float = 0.25
    version: str = "causal-graph-v1"


@dataclass(frozen=True, slots=True)
class CausalGraph:
    prior: CausalGraphPrior
    parents: tuple[tuple[int, ...], ...]
    seed: int
    task_id: int


def sample_causal_graph(
    prior: CausalGraphPrior,
    *,
    seed: int,
    task_id: int,
    limits: CausalLimits | None = None,
) -> CausalGraph:
    """Sample once; preserve disconnected draws without conditioning on usefulness."""
    raise NotImplementedError("causal graph sampling is not implemented")
