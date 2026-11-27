"""Separate parent transforms and explicitly frozen elementwise or neural reductions."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.fitted_tree_records import FrozenFittedTreeFunction
from r0.modules.priors.lib.frozen_function_composites import (
    FrozenDiscretizationFunction,
    FrozenEMFunction,
    FrozenMLPFunction,
    FrozenProductFunction,
)
from r0.modules.priors.lib.frozen_function_primitives import (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenRFFFunction,
    FrozenTreeFunction,
)

type ParentMember = (
    FrozenLinearFunction
    | FrozenQuadraticFunction
    | FrozenTreeFunction
    | FrozenDiscretizationFunction
    | FrozenRFFFunction
    | FrozenEMFunction
    | FrozenProductFunction
    | FrozenMLPFunction
    | FrozenFittedTreeFunction
)
_MEMBER_TYPES = (
    FrozenLinearFunction,
    FrozenQuadraticFunction,
    FrozenTreeFunction,
    FrozenDiscretizationFunction,
    FrozenRFFFunction,
    FrozenEMFunction,
    FrozenProductFunction,
    FrozenMLPFunction,
    FrozenFittedTreeFunction,
)


@dataclass(frozen=True, slots=True)
class FrozenParentFunction:
    """Children consume disjoint consecutive slices, in declared parent order.

    An optional joint function consumes their concatenation and contributes the
    FIRST member to the reduction (MITRA2 residual semantics). It is never added
    after reducing the projections. Neural reductions consume member-major
    concatenation. Weights are already frozen normalized nonnegative weights.
    No parent-composition child can recursively contain another composition.
    """

    children: tuple[ParentMember, ...]
    aggregation: str = "sum"
    joint: ParentMember | None = None
    weights: tuple[float, ...] = ()
    reducer: FrozenMLPFunction | None = None
    version: str = "frozen-parent-function-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "frozen-parent-function-v1"
            or type(self.children) is not tuple
            or not self.children
            or any(type(c) not in _MEMBER_TYPES for c in self.children)
            or self.aggregation
            not in ("sum", "product", "mean", "max", "logsumexp", "weighted", "mlp")
        ):
            raise ValueError(
                "parent composition requires nonrecursive frozen children and a reducer"
            )
        if any(c.output_width != self.output_width for c in self.children):
            raise ValueError("parent transforms must share their output width")
        if self.joint is not None and (
            type(self.joint) not in _MEMBER_TYPES
            or self.joint.input_width != self.input_width
            or self.joint.output_width != self.output_width
        ):
            raise ValueError("joint member must consume all ordered parent slices")
        members = len(self.children) + int(self.joint is not None)
        if (
            type(self.weights) is not tuple
            or (
                self.aggregation == "weighted"
                and (
                    len(self.weights) != members
                    or any(not finite_number(w) or w < 0 for w in self.weights)
                    or abs(sum(self.weights) - 1) > 1e-12
                )
            )
            or (self.aggregation != "weighted" and self.weights)
        ):
            raise ValueError("weighted reduction requires one normalized frozen weight per member")
        if self.aggregation == "mlp":
            if type(self.reducer) is not FrozenMLPFunction or (
                self.reducer.input_width,
                self.reducer.output_width,
            ) != (members * self.output_width, self.output_width):
                raise ValueError("neural reducer must consume member-major concatenation")
        elif self.reducer is not None:
            raise ValueError("only neural reduction can retain a reducer function")

    @property
    def input_width(self) -> int:
        return sum(c.input_width for c in self.children)

    @property
    def output_width(self) -> int:
        return self.children[0].output_width

    @property
    def parameter_count(self) -> int:
        return (
            sum(c.parameter_count for c in self.children)
            + (self.joint.parameter_count if self.joint else 0)
            + (self.reducer.parameter_count if self.reducer else 0)
            + len(self.weights)
        )
