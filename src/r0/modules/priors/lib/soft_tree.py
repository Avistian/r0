"""Compile authored oblique soft trees into existing frozen network equations."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
)

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class SoftTreeBranch:
    weights: tuple[float, ...]
    threshold: float
    left: int
    right: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.weights) is not tuple
            or not self.weights
            or any(not finite_number(v) for v in self.weights)
            or not finite_number(self.threshold)
            or not finite_number(self.temperature)
            or self.temperature <= 0
            or type(self.left) is not int
            or type(self.right) is not int
            or self.left < 0
            or self.right < 0
            or self.left == self.right
        ):
            raise ValueError(
                "soft tree requires finite weights, positive temperature and distinct children"
            )


def compile_soft_tree(
    input_width: int,
    nodes: tuple[SoftTreeBranch | NetworkAffine, ...],
    *,
    limits: CausalLimits = _DEFAULT,
) -> FrozenNetworkFunction:
    """Node zero is root; children follow parents; affine records are leaf models.

    A branch returns (1-sigmoid(z))*left + sigmoid(z)*right, with
    z=(weights @ input-threshold)/temperature. At equality it averages both.
    """
    from r0.modules.priors.lib.frozen_function_evaluation import admit_frozen_function

    if (
        type(input_width) is not int
        or input_width < 1
        or type(nodes) is not tuple
        or not nodes
        or any(type(n) not in (SoftTreeBranch, NetworkAffine) for n in nodes)
        or type(limits) is not CausalLimits
    ):
        raise ValueError("soft tree requires frozen nodes and explicit geometry")
    if len(nodes) > limits.nodes or input_width > limits.node_channels:
        raise ValueError("soft tree construction geometry exceeds limits")
    incoming = [0] * len(nodes)
    leaves = [n for n in nodes if isinstance(n, NetworkAffine)]
    if not leaves or any(
        n.input_width != input_width or n.output_width != leaves[0].output_width for n in leaves
    ):
        raise ValueError("soft tree leaf input/output widths must agree")
    output = leaves[0].output_width
    for i, node in enumerate(nodes):
        if isinstance(node, SoftTreeBranch):
            if len(node.weights) != input_width or any(
                not i < child < len(nodes) for child in (node.left, node.right)
            ):
                raise ValueError(
                    "soft tree children must follow parent and weights must match input"
                )
            incoming[node.left] += 1
            incoming[node.right] += 1
    if incoming != [0] + [1] * (len(nodes) - 1):
        raise ValueError("soft tree must be reachable without shared children")
    width = input_width + 2 * output
    branches = len(nodes) - len(leaves)
    parameters = sum(n.parameter_count for n in leaves) + branches * (
        (2 * output + 2) * width + 2 * output + 16
    )
    if (
        width > limits.node_channels
        or input_width + len(nodes) * output > limits.total_channels
        or parameters > limits.parameters
        or 65536 + 512 * parameters > limits.working_bytes
    ):
        raise ValueError("soft tree compiled construction exceeds limits")
    blocks: list[NetworkBlock] = []
    states: dict[int, int] = {}
    for i in reversed(range(len(nodes))):
        node = nodes[i]
        if isinstance(node, NetworkAffine):
            block = NetworkBlock((0,), (node,))
        else:
            experts = tuple(
                NetworkAffine(
                    tuple(
                        tuple(float(j == input_width + side * output + k) for j in range(width))
                        for k in range(output)
                    ),
                    (0.0,) * output,
                )
                for side in (0, 1)
            )
            gate = NetworkAffine(
                (
                    (0.0,) * width,
                    tuple(w / node.temperature for w in node.weights) + (0.0,) * (2 * output),
                ),
                (0.0, -node.threshold / node.temperature),
            )
            block = NetworkBlock((0, states[node.left], states[node.right]), experts, gate)
        blocks.append(block)
        states[i] = len(blocks)
    function = FrozenNetworkFunction(input_width, tuple(blocks))
    admit_frozen_function(function, 1, limits)
    return function
