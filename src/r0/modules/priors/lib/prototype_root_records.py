"""Frozen shared prototype banks and explicit count/lifecycle conventions."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.prototype_weight_law import PrototypeWeightLaw
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase
from r0.modules.priors.lib.source_root_base import SourceRootBase
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape


@dataclass(frozen=True, slots=True)
class FrozenPrototypeRoot:
    prototypes: Matrix
    weights: PrototypeWeightLaw = PrototypeWeightLaw()
    version: str = "frozen-prototype-root-v1"

    def __post_init__(self) -> None:
        count, width = matrix_shape(self.prototypes)
        if (
            self.version != "frozen-prototype-root-v1"
            or count > 4096
            or width > 1024
            or type(self.weights) is not PrototypeWeightLaw
        ):
            raise ValueError("invalid frozen prototype root")
        self.weights.draw_spec(count, 1, ("prototype-emission",))

    @property
    def width(self) -> int:
        return len(self.prototypes[0])

    @property
    def count(self) -> int:
        return len(self.prototypes)

    @property
    def moment_order(self) -> float:
        """Conditional on this finite frozen bank, the convex hull is bounded."""
        return math.inf

    @property
    def constant_channels(self) -> tuple[bool, ...]:
        return tuple(
            all(row[j] == self.prototypes[0][j] for row in self.prototypes)
            for j in range(self.width)
        )


@dataclass(frozen=True, slots=True)
class PrototypeRootPlan:
    base: JointRootBase | SourceRootBase
    reference_rows: int
    rho: float
    weights: PrototypeWeightLaw = PrototypeWeightLaw()
    count_rule: str = "ceil-r0"
    max_prototypes: int = 256
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "prototype-root-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "prototype-root-plan-v1"
            or type(self.base) not in (JointRootBase, SourceRootBase, SampledSourceRootBase)
            or type(self.reference_rows) is not int
            or not 1 <= self.reference_rows < 2**31
            or not finite_number(self.rho)
            or not 0 < self.rho <= 1
            or self.count_rule not in ("ceil-r0", "floor-r0", "nearest-even-r0")
            or type(self.max_prototypes) is not int
            or not 1 <= self.max_prototypes <= 4096
            or type(self.weights) is not PrototypeWeightLaw
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid prototype root plan")
        if not 1 <= self.count <= self.max_prototypes:
            raise ValueError("prototype count outside declared limit; no clipping")
        self.weights.draw_spec(self.count, 1, ("prototype-emission",))

    @property
    def count(self) -> int:
        value = self.rho * self.reference_rows
        return (
            math.ceil(value)
            if self.count_rule == "ceil-r0"
            else math.floor(value)
            if self.count_rule == "floor-r0"
            else round(value)
        )

    @property
    def construction_moment_order(self) -> float:
        """Bound before conditioning on a finite sampled bank, distinct from its support."""
        return self.base.moment_order if isinstance(self.base, SourceRootBase) else math.inf


def admit_prototype_work(
    count: int,
    width: int,
    rows: int,
    limits: CausalLimits,
    *,
    metadata_parameters: int = 0,
    metadata_bytes: int = 0,
) -> None:
    if (
        type(limits) is not CausalLimits
        or type(rows) is not int
        or not 1 <= rows < 2**31
        or type(count) is not int
        or not 1 <= count <= 4096
        or type(width) is not int
        or not 1 <= width <= 1024
    ):
        raise ValueError("invalid prototype emission dimensions or limits")
    report = limits.admit_dimensions(
        (width,),
        ((),),
        rows=max(rows, count),
        calibration_rows=0,
        features=width,
        outputs=width,
        extra_parameters=metadata_parameters + 4 * count * width + 3 * rows * (count + width),
        extra_working_channels=8 * count + 4 * width,
    )
    if report["estimated_working_bytes"] + metadata_bytes > limits.working_bytes:
        raise ValueError("prototype retained metadata working limit exceeded")
