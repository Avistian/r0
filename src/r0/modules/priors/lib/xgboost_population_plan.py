"""Actual-population histogram controls; independent fit plans remain unchanged."""

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan


@dataclass(frozen=True, slots=True)
class PopulationXGBoostPlan(XGBoostFitPlan):
    version: str = "population-xgboost-hist-fit-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "population-xgboost-hist-fit-plan-v1"
            or type(self.rows) is not int
            or not 1 <= self.rows <= 1024
        ):
            raise ValueError("population XGBoost requires 1..1024 actual fit rows")
        self.independent_budget_plan()

    def independent_budget_plan(self) -> XGBoostFitPlan:
        values = asdict(self)
        values.update(rows=max(32, self.rows), version="independent-xgboost-hist-fit-plan-v1")
        return XGBoostFitPlan(**values)

    def raw_specs(self) -> tuple[RawDrawSpec, ...]:
        specs = self.independent_budget_plan().raw_specs()[1:]
        return tuple(
            RawDrawSpec(
                ("xgboost", "population-fit"),
                s.role,
                s.distribution,
                (self.rows, self.target_width) if i == 0 else s.shape,
                s.parameters,
            )
            for i, s in enumerate(specs)
        )

    def admit(self, limits: CausalLimits, raw_limits: RawDrawLimits) -> int:
        # Retain the existing conservative backend bound, including its spare input draw.
        return self.independent_budget_plan().admit(limits, raw_limits)

    @property
    def model_byte_bound(self) -> int:
        nodes = self.estimators * (2 ** (self.depth + 1) - 1)
        return 65536 + nodes * (1024 + 64 * self.output_width)

    @property
    def retained_number_bound(self) -> int:
        nodes = self.estimators * (2 ** (self.depth + 1) - 1)
        return (
            4096
            + 4 * nodes * (16 + self.output_width)
            + 8 * self.rows * (self.input_width + self.target_width)
        )
