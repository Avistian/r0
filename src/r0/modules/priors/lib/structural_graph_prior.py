"""Explicit adaptation contracts for degree attachment and hierarchical motifs."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits


@dataclass(frozen=True, slots=True)
class AttachmentGraphPrior:
    nodes: int
    components: int = 1
    edges_per_node: int = 1
    attractiveness: float = 1.0
    power: float = 1.0
    version: str = "degree-attachment-dag-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "degree-attachment-dag-v1"
            or any(type(n) is not int for n in (self.nodes, self.components, self.edges_per_node))
            or not 1 <= self.components <= self.nodes < 2**53
            or not 1 <= self.edges_per_node < 2**53
            or not finite_number(self.attractiveness)
            or self.attractiveness <= 0
            or not finite_number(self.power)
            or self.power < 0
        ):
            raise ValueError(
                "attachment prior requires bounded geometry and finite positive attractiveness"
            )


@dataclass(frozen=True, slots=True)
class MotifGraphPrior:
    depth: int
    components: int = 1
    motifs: tuple[str, ...] = ("chain", "fork", "collider")
    rewrite_steps: int = 0
    rewrite_kinds: tuple[str, ...] = ("redirect", "path", "bypass")
    version: str = "recursive-motif-dag-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-motif-dag-v1"
            or type(self.depth) is not int
            or not 0 <= self.depth <= 39
            or type(self.components) is not int
            or not 1 <= self.components < 2**53
            or type(self.rewrite_steps) is not int
            or not 0 <= self.rewrite_steps < 2**53
        ):
            raise ValueError("motif prior requires bounded depth, components and rewrite count")
        for values, allowed in (
            (self.motifs, ("chain", "fork", "collider")),
            (self.rewrite_kinds, ("redirect", "path", "bypass")),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(v) is not str or v not in allowed for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("motif catalogs require distinct supported choices")

    @property
    def nodes(self) -> int:
        return self.components * 3**self.depth


type StructuralGraphPrior = AttachmentGraphPrior | MotifGraphPrior


def admit_structural_graph(prior: StructuralGraphPrior, limits: CausalLimits) -> None:
    if (
        type(prior) not in (AttachmentGraphPrior, MotifGraphPrior)
        or type(limits) is not CausalLimits
    ):
        raise ValueError("structural graph requires frozen prior and limits")
    n = prior.nodes
    if n > min(limits.nodes, limits.total_channels):
        raise ValueError("structural graph node limit exceeded")
    if isinstance(prior, AttachmentGraphPrior):
        # Dynamic categorical vectors are retained at every attachment, including
        # choices not selected. Bound their total support before any draws.
        entries = n * n * min(prior.edges_per_node, n)
    else:
        # Dense port connections, retained edge-delta receipts and replay copies.
        entries = n * n * (prior.rewrite_steps + prior.depth + 1)
    if 4 * entries + 16 * n + 512 > limits.parameters:
        raise ValueError("structural graph parameter limit exceeded")
    if 512 * entries + 2048 * n + 8 * 1024**2 > limits.working_bytes:
        raise ValueError("structural graph working memory limit exceeded")
