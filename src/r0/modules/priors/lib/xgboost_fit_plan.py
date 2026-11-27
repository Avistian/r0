"""Explicit bounded synthetic CPU histogram fitting controls."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec


@dataclass(frozen=True, slots=True)
class XGBoostFitPlan:
    input_width: int = 3
    output_width: int = 2
    rows: int = 128
    estimators: int = 4
    depth: int = 3
    max_bin: int = 256
    objective: str = "reg:squarederror"
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    version: str = "independent-xgboost-hist-fit-plan-v1"

    def __post_init__(self) -> None:
        if self.version != "independent-xgboost-hist-fit-plan-v1" or self.objective not in (
            "reg:squarederror",
            "binary:logistic",
            "multi:softprob",
        ):
            raise ValueError("invalid XGBoost fit version or objective")
        for value, low, high in (
            (self.input_width, 1, 256),
            (self.output_width, 1, 100),
            (self.rows, 32, 1024),
            (self.estimators, 1, 32),
            (self.depth, 1, 8),
            (self.max_bin, 2, 256),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("XGBoost fit geometry exceeds its exploration bounds")
        if self.objective == "multi:softprob" and self.output_width < 2:
            raise ValueError("multiclass fitting requires at least two outputs")
        for value in (self.learning_rate, self.subsample, self.colsample_bytree):
            if not finite_number(value) or not 0 < value <= 1:
                raise ValueError("XGBoost rate or sampling proportion must lie in (0,1]")
        for value in (self.min_child_weight, self.reg_lambda, self.reg_alpha, self.gamma):
            if not finite_number(value) or not 0 <= value <= 1e6:
                raise ValueError("XGBoost regularization must be finite, nonnegative and bounded")

    @property
    def target_width(self) -> int:
        return 1 if self.objective == "multi:softprob" else self.output_width

    def raw_specs(self) -> tuple[RawDrawSpec, ...]:
        owner = ("xgboost", "independent-fit")
        classifier = self.objective != "reg:squarederror"
        high = self.output_width if self.objective == "multi:softprob" else 2
        return (
            RawDrawSpec(owner, "synthetic-inputs", "normal", (self.rows, self.input_width)),
            RawDrawSpec(
                owner,
                "synthetic-targets",
                "integers" if classifier else "normal",
                (self.rows, self.target_width),
                (("high", high),) if classifier else (),
            ),
            RawDrawSpec(owner, "backend-seed", "integers", (1,), (("high", 2**31),)),
        )

    def admit(self, limits: CausalLimits, raw_limits: RawDrawLimits) -> int:
        if type(limits) is not CausalLimits or type(raw_limits) is not RawDrawLimits:
            raise ValueError("XGBoost fit requires frozen caller limits")
        specs = self.raw_specs()
        raw_bytes = sum(
            s.retained_bytes(name)
            for s, name in zip(specs, ("input", "target", "backend"), strict=True)
        )
        if (
            len(specs) > raw_limits.draws
            or sum(s.numbers for s in specs) > raw_limits.numbers
            or raw_bytes + 6 * raw_limits.state_bytes > raw_limits.working_bytes
        ):
            raise ValueError("XGBoost aggregate raw fit limit exceeded before RNG copies")
        nodes = self.estimators * (2 ** (self.depth + 1) - 1)
        histogram = (2**self.depth) * self.max_bin * self.input_width * self.output_width * 2
        # Conservative histogram, fit arrays, Python records and model export buffers.
        report = limits.admit_dimensions(
            (self.input_width, self.output_width),
            ((), ()),
            rows=self.rows,
            calibration_rows=self.rows,
            features=self.input_width,
            outputs=self.output_width,
            extra_parameters=nodes * (8 + self.output_width)
            + 8 * self.rows * (self.input_width + self.target_width),
            extra_working_channels=16 * (self.input_width + self.output_width),
        )
        total = (
            report["estimated_working_bytes"]
            + raw_bytes
            + 8 * histogram
            + 8 * raw_limits.state_bytes
        )
        if total > limits.working_bytes:
            raise ValueError("XGBoost aggregate histogram/record workspace limit exceeded")
        return total

    def backend_parameters(self, seed: int) -> dict[str, Any]:
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("invalid XGBoost backend seed")
        result: dict[str, Any] = {
            "tree_method": "hist",
            "multi_strategy": "multi_output_tree",
            "device": "cpu",
            "objective": self.objective,
            "nthread": 1,
            "seed": seed,
            "max_depth": self.depth,
            "max_bin": self.max_bin,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,
            "alpha": self.reg_alpha,
            "gamma": self.gamma,
        }
        if self.objective == "multi:softprob":
            result["num_class"] = self.output_width
        return result
