"""Versioned heterogeneous variable layout and shared prior request for each joint node."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_plans import validate_graph_geometry
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_domain_prior import (
    AsymmetricNativeDomainPrior,
    NativeDomainPlan,
    NativeDomainPrior,
)
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class NativeVectorDomainPlan:
    variables: tuple[NativeDomainPlan, ...]
    version: str = "native-vector-domain-plan-v1"

    @property
    def score_width(self) -> int:
        return sum(v.score_width for v in self.variables)

    @property
    def state_width(self) -> int:
        return sum(v.state_width for v in self.variables)

    @property
    def asymmetric_draws(self) -> int:
        return sum(
            v.cardinality for v in self.variables if v.frequency_law == "asymmetric-dirichlet"
        )


@dataclass(frozen=True, slots=True)
class NativeVectorGraphRequest:
    parents: tuple[tuple[int, ...], ...]
    domains: tuple[tuple[NativeDomainPrior, ...], ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    rows: int = 64
    context_rows: int = 48
    split: int = 0
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    composition: ParentCompositionPrior | None = None
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "native-vector-graph-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-vector-graph-request-v1"
            or type(self.parents) is not tuple
            or type(self.domains) is not tuple
            or len(self.parents) != len(self.domains)
            or type(self.root_prior) is not JointRootPrior
            or type(self.function_prior) is not JointFunctionPrior
            or type(self.raw_limits) is not RawDrawLimits
            or type(self.limits) is not CausalLimits
            or self.composition is not None
            and type(self.composition) is not ParentCompositionPrior
        ):
            raise ValueError("native vector request requires frozen geometry, priors and limits")
        if any(
            type(ds) is not tuple
            or not ds
            or len(ds) > self.limits.node_channels
            or any(
                type(d) not in (NativeDomainPrior, AsymmetricNativeDomainPrior)
                or d.continuous_width != 1
                for d in ds
            )
            for ds in self.domains
        ):
            raise ValueError("native vector nodes require bounded ordered scalar variable priors")
        widths = tuple(
            sum(
                max(
                    1,
                    max(d.cardinalities) if "categorical" in d.kinds else 0,
                    2 if "binary" in d.kinds else 0,
                )
                for d in ds
            )
            for ds in self.domains
        )
        validate_graph_geometry(self.parents, widths, self.calibration_rows, self.limits)
        if type(self.observed) is not tuple:
            raise ValueError("native vector observations require ordered references")
        for ref in (*self.observed, self.target):
            if (
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(i) is not int for i in ref)
                or not 0 <= ref[0] < len(self.domains)
                or not 0 <= ref[1] < len(self.domains[ref[0]])
            ):
                raise ValueError("native vector readout requires logical-variable indices")
        if self.target in self.observed:
            raise ValueError("native vector target variable cannot be observed")
        target = self.domains[self.target[0]][self.target[1]]
        if "categorical" in target.kinds and max(target.cardinalities) > 100:
            raise ValueError("native vector target exceeds shared 100-class limit")
        if (
            self.mode not in ("local", "global", "meta")
            or self.categorical_policy not in ("shared", "refresh")
            or self.global_policy not in ("scalar-quantile-v1", "strict-contract-v1")
        ):
            raise ValueError("unsupported native vector preference policy")
        self.task()

    def task(self, *, kind: str = "continuous", cardinality: int = 0) -> LinearRecipe:
        return LinearRecipe(
            rows=self.rows,
            features=len(self.observed),
            signal_features=len(self.observed),
            active_features=len(self.observed),
            context_rows=self.context_rows,
            split=self.split,
            task_type="regression"
            if kind == "continuous"
            else "binary"
            if cardinality == 2
            else "multiclass",
            classes=cardinality,
        )
