"""Frozen feedforward architectures, independent of the legacy fitted MLP law."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


@dataclass(frozen=True, slots=True)
class NetworkAffine:
    matrix: Matrix
    bias: tuple[float, ...]
    mask: tuple[tuple[bool, ...], ...] = ()
    multiplier: float = 1.0
    activation: CausalActivation = CausalActivation()
    placement: str = "after"

    def __post_init__(self) -> None:
        height, width = matrix_shape(self.matrix)
        if (
            type(self.bias) is not tuple
            or len(self.bias) != height
            or any(not finite_number(v) for v in self.bias)
            or not finite_number(self.multiplier)
            or type(self.activation) is not CausalActivation
            or self.placement not in ("before", "after")
            or type(self.mask) is not tuple
            or self.mask
            and (
                len(self.mask) != height
                or any(
                    type(row) is not tuple
                    or len(row) != width
                    or any(type(v) is not bool for v in row)
                    for row in self.mask
                )
            )
        ):
            raise ValueError(
                "network affine requires finite aligned bias/mask and explicit activation"
            )

    @property
    def input_width(self) -> int:
        return len(self.matrix[0])

    @property
    def output_width(self) -> int:
        return len(self.matrix)

    @property
    def parameter_count(self) -> int:
        return (
            self.output_width * self.input_width * (2 if self.mask else 1)
            + len(self.bias)
            + 1
            + self.activation.parameter_count
        )


@dataclass(frozen=True, slots=True)
class NetworkProjection:
    source: int
    affine: NetworkAffine

    def __post_init__(self) -> None:
        if (
            type(self.source) is not int
            or self.source < 0
            or type(self.affine) is not NetworkAffine
        ):
            raise ValueError("network projection requires an explicit earlier state and affine map")
        if self.affine.activation != CausalActivation():
            raise ValueError("skip projections are affine; nonlinear paths belong in experts")


@dataclass(frozen=True, slots=True)
class NetworkBlock:
    parents: tuple[int, ...]
    experts: tuple[NetworkAffine, ...]
    gate: NetworkAffine | None = None
    skips: tuple[NetworkProjection, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.parents) is not tuple
            or not self.parents
            or any(type(p) is not int or p < 0 for p in self.parents)
            or len(set(self.parents)) != len(self.parents)
            or type(self.experts) is not tuple
            or not self.experts
            or any(type(e) is not NetworkAffine for e in self.experts)
            or type(self.skips) is not tuple
            or any(type(s) is not NetworkProjection for s in self.skips)
        ):
            raise ValueError(
                "network block requires ordered unique parents and frozen experts/skips"
            )
        shape = (self.experts[0].input_width, self.experts[0].output_width)
        if any((e.input_width, e.output_width) != shape for e in self.experts):
            raise ValueError("network experts must share input/output widths")
        if self.gate is None:
            if len(self.experts) != 1:
                raise ValueError("multiple network experts require an explicit softmax gate")
        elif (
            type(self.gate) is not NetworkAffine
            or self.gate.input_width != shape[0]
            or self.gate.output_width != len(self.experts)
            or self.gate.activation != CausalActivation()
        ):
            raise ValueError("network gate must emit one affine logit per expert")
        if any(s.affine.output_width != shape[1] for s in self.skips):
            raise ValueError("skip projections must match block output width")

    @property
    def output_width(self) -> int:
        return self.experts[0].output_width


@dataclass(frozen=True, slots=True)
class FrozenNetworkFunction:
    input_width: int
    blocks: tuple[NetworkBlock, ...]
    version: str = "frozen-structural-network-v1"

    def __post_init__(self) -> None:
        function_width(self.input_width)
        if (
            self.version != "frozen-structural-network-v1"
            or type(self.blocks) is not tuple
            or not self.blocks
        ):
            raise ValueError("network requires its version and a nonempty frozen block sequence")
        widths = [self.input_width]
        for block in self.blocks:
            if type(block) is not NetworkBlock or any(p >= len(widths) for p in block.parents):
                raise ValueError("network parents must refer to input=0 or earlier block states")
            if sum(widths[p] for p in block.parents) != block.experts[0].input_width:
                raise ValueError("network ordered parent slices disagree with affine input width")
            for skip in block.skips:
                if skip.source >= len(widths) or skip.affine.input_width != widths[skip.source]:
                    raise ValueError("network skip must project an earlier state's exact width")
            widths.append(block.output_width)

    @property
    def output_width(self) -> int:
        return self.blocks[-1].output_width

    @property
    def parameter_count(self) -> int:
        return 1 + sum(
            len(b.parents)
            + sum(e.parameter_count for e in b.experts)
            + (b.gate.parameter_count if b.gate else 0)
            + sum(1 + s.affine.parameter_count for s in b.skips)
            for b in self.blocks
        )
