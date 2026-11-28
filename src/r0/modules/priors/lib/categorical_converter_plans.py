"""Source categorical mode/width decisions precede observation geometry allocation."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_converter import CONVERTER_MODES
from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_sampling import sample_joint_function
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_root_plans import JointRootPlan, JointRootPrior, sample_joint_root
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan, sample_joint_weights
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

_ALL_FEATURES = CategoricalParameter("cat_use_all_features", ("false", "true"))


@dataclass(frozen=True, slots=True)
class CategoricalConverterPrior:
    modes: tuple[str, ...] = CONVERTER_MODES
    function: JointFunctionPrior = JointFunctionPrior()
    embeddings: JointRootPrior = JointRootPrior()
    weight_family: str = "polynomial"
    scalarization: str = "scalarized-weight-batches-v1"
    version: str = "independent-category-converter-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "independent-category-converter-prior-v1"
            or type(self.modes) is not tuple
            or not self.modes
            or any(m not in CONVERTER_MODES for m in self.modes)
            or len(set(self.modes)) != len(self.modes)
            or type(self.function) is not JointFunctionPrior
            or type(self.embeddings) is not JointRootPrior
            or self.weight_family not in ("polynomial", "uniform")
            or self.scalarization
            not in ("scalarized-weight-batches-v1", "source-weight-batches-v1")
        ):
            raise ValueError(
                "category converter prior requires frozen catalogs and supported weight policy"
            )

    def request(self) -> CategoricalParameter:
        return CategoricalParameter("cat_mode", self.modes)


def converter_parameter(role: str, cardinality: int) -> NumericalParameter:
    if role == "width":
        return NumericalParameter(
            "cat_n_features",
            1,
            cardinality,
            mode="local",
            scale="log",
            integer="source-clip-floor-v1",
        )
    if role == "exponent":
        return NumericalParameter("cat_disc_p", 0.5, 4.0, scale="log")
    if role == "factor":
        return NumericalParameter("softmax_disc_factor", 0.1, 10.0, scale="log")
    raise ValueError("unknown category converter parameter")


@dataclass(frozen=True, slots=True)
class CategoricalConverterPlan:
    operation: tuple[str, ...]
    prior: CategoricalConverterPrior
    vocabulary: tuple[str, ...]
    mode: str
    width: int
    calibration_rows: int
    all_features: bool | None
    exponent: float = 2.0
    factor: float = 1.0
    embeddings: JointRootPlan | None = None
    weights: JointWeightPlan | None = None
    function: JointFunctionPlan | None = None
    version: str = "independent-category-converter-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            self.version != "independent-category-converter-plan-v1"
            or type(self.prior) is not CategoricalConverterPrior
            or self.mode not in self.prior.modes
            or type(self.width) is not int
            or not 1 <= self.width < 2**63
            or type(self.calibration_rows) is not int
            or not 1 <= self.calibration_rows < 2**63
            or type(self.vocabulary) is not tuple
            or len(self.vocabulary) < 2
            or any(type(v) is not str or not v for v in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)
        ):
            raise ValueError(
                "converter plan requires version, vocabulary, geometry and enabled mode"
            )
        if self.mode.startswith("softmax_"):
            if (
                self.width != len(self.vocabulary)
                or self.all_features is not None
                or self.exponent != 2.0
                or type(self.embeddings) is not JointRootPlan
                or type(self.weights) is not JointWeightPlan
                or self.embeddings.operation != (*self.operation, "embeddings")
                or self.embeddings.prior != self.prior.embeddings
                or (self.embeddings.width, self.embeddings.calibration_rows)
                != (self.width, len(self.vocabulary))
                or self.weights.operation != (*self.operation, "weights")
                or (self.weights.width, self.weights.batch) != (self.width, 1)
                or self.weights.family != self.prior.weight_family
                or self.weights.scalarization != self.prior.scalarization
            ):
                raise ValueError(
                    "softmax plan requires its owned embedding and log-bias constructions"
                )
        elif (
            type(self.all_features) is not bool
            or self.width > len(self.vocabulary)
            or self.all_features
            and self.width != len(self.vocabulary)
            or not self.all_features
            and self.width >= len(self.vocabulary)
            or self.embeddings is not None
            or self.weights is not None
            or self.factor != 1.0
            or self.calibration_rows < len(self.vocabulary)
        ):
            raise ValueError(
                "nearest plan requires valid width and sufficient independent center rows"
            )
        if self.mode == "neighbor_func":
            if (
                type(self.function) is not JointFunctionPlan
                or not self.function.cheap
                or self.function.operation != (*self.operation, "function")
                or self.function.prior != self.prior.function
                or (
                    self.function.input_width,
                    self.function.output_width,
                    self.function.calibration_rows,
                )
                != (self.width, self.width, self.calibration_rows)
            ):
                raise ValueError("transformed centers require their owned cheap function plan")
        elif self.function is not None:
            raise ValueError("only transformed-center mode can own a post function")

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        own = tuple(e for e in events if e.operation == self.operation)
        roles = (
            ("mode", "factor")
            if self.mode.startswith("softmax_")
            else (
                "mode",
                "all_features",
                *(("width",) if not self.all_features else ()),
                "exponent",
            )
        )
        if (
            tuple(e.role for e in own) != roles
            or categorical_binding(own[0], decisions, self.prior.request()).value != self.mode
        ):
            raise ValueError("converter mode/scalar ownership differs from retained decisions")
        if (
            self.all_features is not None
            and (categorical_binding(own[1], decisions, _ALL_FEATURES).value == "true")
            != self.all_features
        ):
            raise ValueError("converter channel-width choice differs")
        for event in own[1:]:
            if event.role != "all_features":
                value = numerical_binding(
                    event, decisions, converter_parameter(event.role, len(self.vocabulary))
                ).values[0]
                if getattr(self, event.role) != value:
                    raise ValueError("converter parameter differs from retained named draw")
        child_events = ()
        for child in (self.embeddings, self.weights, self.function):
            if child is not None:
                child.validate(decisions)
                child_events += operation_events(decisions, child.operation)
        if {e.operation for e in events} != {e.operation for e in (*own, *child_events)} or len(
            events
        ) != len(own) + len(child_events):
            raise ValueError("converter contains foreign decision ownership")


def sample_categorical_converter(
    context: JointSamplingContext,
    name: str,
    *,
    vocabulary: tuple[str, ...],
    calibration_rows: int,
    prior: CategoricalConverterPrior,
) -> CategoricalConverterPlan:
    with context.transaction(name) as operation:
        mode = operation.categorical("mode", prior.request()).value
        all_features = None
        width = len(vocabulary)
        exponent, factor = 2.0, 1.0
        embeddings, weights, function = None, None, None
        if mode.startswith("neighbor_"):
            all_features = operation.categorical("all_features", _ALL_FEATURES).value == "true"
            if not all_features:
                width = int(
                    operation.numerical(
                        "width", converter_parameter("width", len(vocabulary))
                    ).values[0]
                )
            exponent = operation.numerical(
                "exponent", converter_parameter("exponent", len(vocabulary))
            ).values[0]
            if mode == "neighbor_func":
                function = sample_joint_function(
                    operation,
                    "function",
                    input_width=width,
                    output_width=width,
                    calibration_rows=calibration_rows,
                    prior=prior.function,
                    cheap=True,
                )
        else:
            factor = operation.numerical(
                "factor", converter_parameter("factor", len(vocabulary))
            ).values[0]
            embeddings = sample_joint_root(
                operation,
                "embeddings",
                width=width,
                calibration_rows=len(vocabulary),
                prior=prior.embeddings,
            )
            weights = sample_joint_weights(
                operation,
                "weights",
                width=width,
                family=prior.weight_family,
                scalarization=prior.scalarization,
            )
        result = CategoricalConverterPlan(
            operation.operation,
            prior,
            vocabulary,
            mode,
            width,
            calibration_rows,
            all_features,
            exponent,
            factor,
            embeddings,
            weights,
            function,
        )
        result.validate(operation.freeze())
        return result
