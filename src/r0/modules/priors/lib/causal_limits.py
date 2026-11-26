"""Explicit causal geometry limits and conservative array-admission estimates."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r0.modules.priors.lib.causal import CausalNode


@dataclass(frozen=True, slots=True)
class CausalLimits:
    nodes: int = 1024
    node_channels: int = 1024
    total_channels: int = 8192
    parameters: int = 1_000_000
    working_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        values = (
            self.nodes,
            self.node_channels,
            self.total_channels,
            self.parameters,
            self.working_bytes,
        )
        if any(type(value) is not int or not 1 <= value < 2**63 for value in values):
            raise ValueError("causal limits must be positive integers below 2**63")

    def admit(
        self,
        nodes: tuple["CausalNode", ...],
        *,
        rows: int,
        calibration_rows: int,
        features: int,
        outputs: int,
    ) -> dict[str, int]:
        """Bound declared geometry before tensors; this is not measured process RSS."""
        widths = tuple(len(node.bias) for node in nodes)
        count, widest, total = len(nodes), max(widths), sum(widths)
        if count > self.nodes:
            raise ValueError("causal node count limit exceeded")
        if widest > self.node_channels:
            raise ValueError("causal node channel limit exceeded")
        if total > self.total_channels:
            raise ValueError("causal total channel limit exceeded")
        parameters = sum(
            2 * len(node.bias) + sum(len(row) for row in node.weights) for node in nodes
        )
        if parameters > self.parameters:
            raise ValueError("causal parameter limit exceeded")
        fan_in = max(sum(widths[parent] for parent in node.parents) for node in nodes)
        peak_rows = max(rows, calibration_rows)
        # Float64: simultaneous values/innovations, calibration, parent concatenation,
        # temporary evaluation buffers, observation/target arrays, and serialized
        # parameter copies. Interpreter/library baseline and allocator overhead are
        # deliberately not described as measured or hard OS-enforced memory.
        working = (
            8 * peak_rows * (6 * total + 2 * fan_in + 2 * features + 4 * outputs + 4)
            + 128 * parameters
        )
        if working > self.working_bytes:
            raise ValueError("estimated causal working memory exceeds declared limit")
        return {
            "nodes": count,
            "largest_node_channels": widest,
            "total_channels": total,
            "parameters": parameters,
            "estimated_working_bytes": working,
        }
