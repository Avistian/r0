"""A standalone network equation with explicit incoming state slices."""

from dataclasses import dataclass

from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.structural_network import NetworkBlock


@dataclass(frozen=True, slots=True)
class FrozenNetworkBlockFunction:
    incoming_widths: tuple[int, ...]
    block: NetworkBlock
    version: str = "frozen-network-block-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-network-block-v1"
            or type(self.incoming_widths) is not tuple
            or not self.incoming_widths
            or type(self.block) is not NetworkBlock
        ):
            raise ValueError("network block requires frozen incoming widths and equation")
        for width in self.incoming_widths:
            function_width(width)
        b = self.block
        used = set(b.parents) | {s.source for s in b.skips}
        if used != set(range(len(self.incoming_widths))):
            raise ValueError("network block incoming states must be exactly its used ancestors")
        if sum(self.incoming_widths[p] for p in b.parents) != b.experts[0].input_width:
            raise ValueError("network block parent widths disagree with affine equation")
        if any(s.affine.input_width != self.incoming_widths[s.source] for s in b.skips):
            raise ValueError("network block skip widths disagree with incoming states")

    @property
    def input_width(self) -> int:
        return sum(self.incoming_widths)

    @property
    def output_width(self) -> int:
        return self.block.output_width

    @property
    def parameter_count(self) -> int:
        b = self.block
        return (
            len(self.incoming_widths)
            + len(b.parents)
            + sum(e.parameter_count for e in b.experts)
            + (b.gate.parameter_count if b.gate else 0)
            + sum(1 + s.affine.parameter_count for s in b.skips)
        )
