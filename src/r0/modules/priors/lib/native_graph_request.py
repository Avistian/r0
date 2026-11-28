"""Explicit native DAG geometry with sampled variable-domain and mechanism priors."""

from dataclasses import dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_plans import validate_graph_geometry
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.native_domain_prior import AsymmetricNativeDomainPrior, NativeDomainPrior
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits


@dataclass(frozen=True, slots=True)
class NativeGraphRequest:
    parents: tuple[tuple[int, ...], ...]
    domains: tuple[NativeDomainPrior, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    rows: int = 64
    context_rows: int = 48
    split: int = 0
    calibration_rows: int = 256
    root_prior: JointRootPrior = JointRootPrior()
    function_prior: JointFunctionPrior = JointFunctionPrior()
    mode: str = "meta"
    categorical_policy: str = "shared"
    global_policy: str = "scalar-quantile-v1"
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "native-graph-request-v1"

    def __post_init__(self) -> None:
        from r0.modules.priors.lib.native_parent_request import NativeParentGraphRequest

        expected = {
            NativeGraphRequest: "native-graph-request-v1",
            NativeParentGraphRequest: "native-parent-graph-request-v1",
        }.get(type(self))
        if (
            expected is None
            or self.version != expected
            or type(self.parents) is not tuple
            or type(self.domains) is not tuple
            or (
                len(self.domains) != len(self.parents)
                or any(
                    type(d) not in (NativeDomainPrior, AsymmetricNativeDomainPrior)
                    for d in self.domains
                )
                or type(self.root_prior) is not JointRootPrior
                or type(self.function_prior) is not JointFunctionPrior
                or type(self.raw_limits) is not RawDrawLimits
            )
        ):
            raise ValueError("native construction requires frozen geometry/domain/mechanism priors")
        widths = tuple(
            max(
                d.continuous_width if "continuous" in d.kinds else 0,
                max(d.cardinalities) if "categorical" in d.kinds else 0,
                2 if "binary" in d.kinds else 0,
            )
            for d in self.domains
        )
        validate_graph_geometry(self.parents, widths, self.calibration_rows, self.limits)
        if type(self.observed) is not tuple:
            raise ValueError("native observations require an ordered tuple")
        for ref in (*self.observed, self.target):
            if (
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(v) is not int for v in ref)
                or (not 0 <= ref[0] < len(self.parents) or ref[1] < 0)
            ):
                raise ValueError("native readout references require valid node/channel pairs")
            domain = self.domains[ref[0]]
            guaranteed = domain.continuous_width if domain.kinds == ("continuous",) else 1
            if ref[1] >= guaranteed:
                raise ValueError("native readout channel is not guaranteed by every domain option")
        if self.target in self.observed:
            raise ValueError("native target cannot be an observed channel")
        target_prior = self.domains[self.target[0]]
        if "categorical" in target_prior.kinds and max(target_prior.cardinalities) > 100:
            raise ValueError("native target cardinality exceeds shared task class limit of 100")
        self.task()
        if (
            self.mode not in ("meta", "global", "local")
            or self.categorical_policy not in ("shared", "refresh")
            or self.global_policy not in ("scalar-quantile-v1", "strict-contract-v1")
        ):
            raise ValueError("unsupported native construction preference policy")

    def task(self, *, kind: str = "continuous", cardinality: int = 0) -> LinearRecipe:
        task_type = (
            "regression" if kind == "continuous" else "binary" if cardinality == 2 else "multiclass"
        )
        return LinearRecipe(
            rows=self.rows,
            features=len(self.observed),
            signal_features=len(self.observed),
            active_features=len(self.observed),
            context_rows=self.context_rows,
            split=self.split,
            task_type=task_type,
            classes=cardinality,
        )
