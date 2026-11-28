"""Versioned independent synthetic fitting recipe, explicitly an r0 adaptation."""

from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits


@dataclass(frozen=True, slots=True)
class FittedTreePlan:
    family: str = "dt"
    task: str = "regression"
    input_width: int = 1
    output_width: int = 1
    rows: int = 128
    depth: int = 3
    estimators: int = 4
    classes: int = 2
    splitter: str = "random"
    classification_rule: str = "probability_mean"
    learning_rate: float = 0.1
    version: str = "independent-fitted-tree-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "independent-fitted-tree-plan-v1"
            or self.family not in ("dt", "extra_trees", "rf", "gbdt")
            or self.task not in ("regression", "classification")
            or self.splitter not in ("best", "random")
            or self.classification_rule not in ("probability_mean", "majority")
            or not finite_number(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError("invalid fitted-tree recipe; XGBoost remains pending, never aliased")
        for value, low, high in (
            (self.input_width, 1, 256),
            (self.output_width, 1, 16),
            (self.rows, 32, 1024),
            (self.depth, 1, 8),
            (self.estimators, 1, 32),
            (self.classes, 2, 100),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("fitted-tree recipe exceeds its bounded exploration envelope")
        if self.classification_rule == "majority" and (
            self.family == "gbdt" or self.task != "classification"
        ):
            raise ValueError("majority is a named classifier forest extension, not boosting")
        trees = self.tree_bound
        if trees > 4096 or trees * (2 ** (self.depth + 1) - 1) > 131072:
            raise ValueError("fitted-tree worst-case node/tree budget exceeded before fitting")
        if self.rows * (self.input_width + self.output_width) > 1048576:
            raise ValueError("fitted-tree synthetic cell budget exceeded")

    @property
    def tree_bound(self) -> int:
        stages = 1 if self.family == "dt" else self.estimators
        scores = (
            self.classes
            if self.family == "gbdt" and self.task == "classification" and self.classes > 2
            else 1
        )
        return stages * scores * self.output_width

    def admit(self, limits: CausalLimits) -> None:
        nodes = self.tree_bound * (2 ** (self.depth + 1) - 1)
        values = self.classes if self.task == "classification" and self.family != "gbdt" else 1
        limits.admit_dimensions(
            (self.input_width, self.output_width),
            ((), ()),
            rows=self.rows,
            calibration_rows=self.rows,
            features=self.input_width,
            outputs=self.output_width,
            extra_parameters=nodes * (4 + values)
            + 4 * self.rows * (self.input_width + self.output_width),
            extra_working_channels=8 * (self.input_width + self.output_width + self.classes),
        )
