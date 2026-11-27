"""Root base choices and the required width-preserving source function subtree."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_sampling import sample_joint_function
from r0.modules.priors.lib.joint_plan_bindings import categorical_binding, operation_events
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, sample_joint_weights

_BASES = ("uniform", "ball", "gaussian", "covariance")
_COVARIANCE_BASE = CategoricalParameter(
    "random_covariance_base_points", ("uniform-root-base-v1", "gaussian-root-base-v1")
)


@dataclass(frozen=True, slots=True)
class JointRootPrior:
    families: tuple[str, ...] = _BASES
    function: JointFunctionPrior = JointFunctionPrior()
    version: str = "joint-root-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-root-prior-v1"
            or type(self.function) is not JointFunctionPrior
            or type(self.families) is not tuple
            or not self.families
            or any(type(family) is not str or family not in _BASES for family in self.families)
            or len(set(self.families)) != len(self.families)
        ):
            raise ValueError(
                "root prior requires its version, function prior and distinct base catalog"
            )

    def request(self) -> CategoricalParameter:
        return CategoricalParameter(
            "random_base_points", tuple(f"{family}-root-base-v1" for family in self.families)
        )


_DEFAULT_PRIOR = JointRootPrior()


@dataclass(frozen=True, slots=True)
class JointRootPlan:
    operation: tuple[str, ...]
    width: int
    calibration_rows: int
    family: str
    function: JointFunctionPlan
    prior: JointRootPrior = _DEFAULT_PRIOR
    weights: JointWeightPlan | None = None
    covariance_base: str | None = None
    version: str = "joint-root-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        function_width(self.width)
        function_width(self.calibration_rows)
        if (
            self.version != "joint-root-plan-v1"
            or type(self.prior) is not JointRootPrior
            or type(self.family) is not str
            or self.family not in self.prior.families
        ):
            raise ValueError("root plan requires its version, prior and selected base")
        if self.family == "covariance":
            weight = self.weights
            matrix = self.prior.function.matrix
            if (
                type(weight) is not JointWeightPlan
                or weight.operation != (*self.operation, "weights")
                or (weight.width, weight.batch) != (self.width, 1)
                or weight.family != matrix.vector_family
                or weight.scalarization != matrix.scalarization
                or self.covariance_base not in ("uniform", "gaussian")
            ):
                raise ValueError("covariance root requires its owned raw-factor weights and base")
        elif self.weights is not None or self.covariance_base is not None:
            raise ValueError("only covariance roots can own factor weights and a base choice")
        if (
            type(self.function) is not JointFunctionPlan
            or self.function.operation != (*self.operation, "function")
            or self.function.cheap
            or self.function.prior != self.prior.function
            or (
                self.function.input_width,
                self.function.output_width,
                self.function.calibration_rows,
            )
            != (self.width, self.width, self.calibration_rows)
        ):
            raise ValueError("root requires its owned width-preserving full function plan")

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or events[0].operation != self.operation or events[0].role != "base":
            raise ValueError("root requires its initial base selector decision")
        draw = categorical_binding(events[0], decisions, self.prior.request())
        if draw.value != f"{self.family}-root-base-v1":
            raise ValueError("root base differs from its recorded realization")
        expected = (events[0],)
        if self.weights is not None:
            self.weights.validate(decisions)
            expected += operation_events(decisions, self.weights.operation)
            if len(expected) >= len(events):
                raise ValueError("covariance root is missing its base decision")
            event = events[len(expected)]
            if event.operation != self.operation or event.role != "covariance_base":
                raise ValueError("covariance base decision must follow its complete weight subtree")
            base = categorical_binding(event, decisions, _COVARIANCE_BASE)
            if base.value != f"{self.covariance_base}-root-base-v1":
                raise ValueError("covariance base differs from its recorded realization")
            expected += (event,)
        self.function.validate(decisions)
        expected += operation_events(decisions, self.function.operation)
        if events != expected:
            raise ValueError("root must own exactly its ordered descendant decisions")


def sample_joint_root(
    context: JointSamplingContext,
    name: str,
    *,
    width: int,
    calibration_rows: int,
    prior: JointRootPrior = _DEFAULT_PRIOR,
) -> JointRootPlan:
    if type(context) is not JointSamplingContext or type(prior) is not JointRootPrior:
        raise ValueError("root sampling requires a shared context and frozen prior")
    function_width(width)
    function_width(calibration_rows)
    with context.transaction(name) as operation:
        draw = operation.categorical("base", prior.request())
        family = prior.families[prior.request().choices.index(draw.value)]
        weights, covariance_base = None, None
        if family == "covariance":
            matrix = prior.function.matrix
            weights = sample_joint_weights(
                operation,
                "weights",
                width=width,
                batch=1,
                family=matrix.vector_family,
                scalarization=matrix.scalarization,
            )
            base = operation.categorical("covariance_base", _COVARIANCE_BASE)
            covariance_base = ("uniform", "gaussian")[_COVARIANCE_BASE.choices.index(base.value)]
        function = sample_joint_function(
            operation,
            "function",
            input_width=width,
            output_width=width,
            calibration_rows=calibration_rows,
            prior=prior.function,
        )
        plan = JointRootPlan(
            operation.operation,
            width,
            calibration_rows,
            family,
            function,
            prior,
            weights,
            covariance_base,
        )
        plan.validate(operation.freeze())
    return plan
