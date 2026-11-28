"""Named per-parent choices: TabICLv2 catalog and explicitly distinct extensions."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_sampling import sample_joint_function
from r0.modules.priors.lib.joint_plan_bindings import categorical_binding
from r0.modules.priors.lib.joint_plan_ownership import JointPlanOwner, plan_owners
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path


@dataclass(frozen=True, slots=True)
class ParentCompositionPrior:
    """Finite catalogs; source claims never include the r0 extension catalog.

    tabiclv2: concat/aggregate; sum/product/max/logsumexp.
    mitra2: joint nonlinear member PLUS one linear projection per parent, reduced
    together for multi-parent nodes. Weight and neural parameter laws are r0
    adaptations, because the paper specifies the operators but not those laws.
    extension: separate parent functions with the broader reduction catalog.
    """

    source: str = "tabiclv2"
    mode: str = "sample"
    aggregation: str = "sample"
    version: str = "parent-composition-prior-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "parent-composition-prior-v1"
            or self.source not in ("tabiclv2", "mitra2", "extension")
            or self.mode not in ("sample", "concat", "aggregate")
            or self.aggregation not in ("sample", *self.choices)
        ):
            raise ValueError("parent composition requires its version and source-specific controls")
        if self.source == "mitra2" and self.mode != "sample":
            raise ValueError(
                "MITRA2 uses concat for one parent and joint-member reduction for multiple"
            )
        if self.mode == "concat" and self.aggregation != "sample":
            raise ValueError("concat cannot declare an unused aggregation override")

    @property
    def choices(self) -> tuple[str, ...]:
        if self.source == "tabiclv2":
            return ("sum", "product", "max", "logsumexp")
        if self.source == "mitra2":
            return ("mean", "weighted", "mlp", "product", "max")
        return ("sum", "product", "mean", "max", "logsumexp", "weighted", "mlp")

    @property
    def mode_request(self) -> CategoricalParameter:
        return CategoricalParameter("multi_function_func_type", ("concat", "aggregate"))

    @property
    def aggregation_request(self) -> CategoricalParameter:
        name = (
            "aggregation_op_type"
            if self.source == "tabiclv2"
            else f"r0_{self.source}_aggregation_v1"
        )
        return CategoricalParameter(name, self.choices)


@dataclass(frozen=True, slots=True)
class ParentFunctionPlan:
    operation: tuple[str, ...]
    prior: ParentCompositionPrior
    parent_widths: tuple[int, ...]
    mode: str
    aggregation: str
    children: tuple[JointFunctionPlan, ...]
    joint: JointFunctionPlan | None = None
    reducer: JointFunctionPlan | None = None
    version: str = "parent-function-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            self.version != "parent-function-plan-v1"
            or type(self.prior) is not ParentCompositionPrior
            or type(self.parent_widths) is not tuple
            or not self.parent_widths
            or any(type(w) is not int or w < 1 for w in self.parent_widths)
            or type(self.children) is not tuple
            or not self.children
            or any(type(c) is not JointFunctionPlan for c in self.children)
            or self.mode not in ("concat", "aggregate", "residual-members")
        ):
            raise ValueError("parent plan requires its frozen version, widths and children")
        expected_mode = "concat" if len(self.parent_widths) == 1 else "residual-members"
        if self.prior.source == "mitra2":
            if self.mode != expected_mode:
                raise ValueError("MITRA2 mode differs from parent count")
        elif self.mode == "residual-members" or (
            self.prior.mode != "sample" and self.mode != self.prior.mode
        ):
            raise ValueError("parent mode differs from declared source controls")
        if self.mode == "concat":
            if self.aggregation != "none" or self.joint is not None or self.reducer is not None:
                raise ValueError("concat cannot retain reduction members")
            widths = (self.input_width,)
        else:
            widths = self.parent_widths
            if self.aggregation not in self.prior.choices or (
                self.prior.aggregation != "sample" and self.aggregation != self.prior.aggregation
            ):
                raise ValueError("aggregation differs from its source catalog")
        if tuple(c.input_width for c in self.children) != widths:
            raise ValueError("parent child input slices differ from declared widths")
        for i, child in enumerate(self.children):
            if child.operation != (*self.operation, f"parent:{i}"):
                raise ValueError("parent child operation ownership differs")
        if (self.mode == "residual-members") != (self.joint is not None):
            raise ValueError("only residual-member mode owns a joint function")
        if self.joint and (
            type(self.joint) is not JointFunctionPlan
            or self.joint.operation != (*self.operation, "joint")
            or self.joint.input_width != self.input_width
            or any(c.prior.functions != "r0-linear-projection-v1" for c in self.children)
        ):
            raise ValueError("joint member or linear parent projection contract differs")
        if (self.aggregation == "mlp") != (self.reducer is not None):
            raise ValueError("only MLP aggregation owns a reducer function")
        if self.reducer and (
            type(self.reducer) is not JointFunctionPlan
            or self.reducer.operation != (*self.operation, "reducer")
            or self.reducer.prior.functions != "r0-neural-reducer-v1"
            or self.reducer.input_width
            != (len(self.children) + int(self.joint is not None)) * self.output_width
        ):
            raise ValueError("neural reducer geometry or ownership differs")
        if any(
            (c.output_width, c.calibration_rows) != (self.output_width, self.calibration_rows)
            for c in self.functions
        ):
            raise ValueError("parent function output width or fit rows differ")

    @property
    def input_width(self) -> int:
        return sum(self.parent_widths)

    @property
    def output_width(self) -> int:
        return self.children[0].output_width

    @property
    def calibration_rows(self) -> int:
        return self.children[0].calibration_rows

    @property
    def functions(self) -> tuple[JointFunctionPlan, ...]:
        return (
            *((self.joint,) if self.joint else ()),
            *self.children,
            *((self.reducer,) if self.reducer else ()),
        )

    def owners(self, reference: tuple[str | int, ...]) -> tuple[JointPlanOwner, ...]:
        result = [JointPlanOwner(self.operation, "parent-function", reference)]
        if self.joint:
            result.extend(plan_owners(self.joint, (*reference, "joint")))
        for i, child in enumerate(self.children):
            result.extend(plan_owners(child, (*reference, "children", i)))
        if self.reducer:
            result.extend(plan_owners(self.reducer, (*reference, "reducer")))
        return tuple(result)

    def validate(self, decisions: JointSamplingRecord) -> None:
        if (
            self.version != "parent-function-plan-v1"
            or type(self.prior) is not ParentCompositionPrior
        ):
            raise ValueError("unsupported parent plan")
        for child in self.functions:
            child.validate(decisions)
        events = tuple(e for e in decisions.events if e.operation == self.operation)
        expected: list[tuple[str, CategoricalParameter, str]] = []
        if self.prior.source != "mitra2" and self.prior.mode == "sample":
            expected.append(("mode", self.prior.mode_request, self.mode))
        if self.mode != "concat" and self.prior.aggregation == "sample":
            expected.append(("aggregation", self.prior.aggregation_request, self.aggregation))
        if len(events) != len(expected):
            raise ValueError("parent selector event count differs")
        for event, (role, request, value) in zip(events, expected, strict=True):
            if event.role != role or categorical_binding(event, decisions, request).value != value:
                raise ValueError("parent selector binding differs")


def sample_parent_function(
    context: JointSamplingContext,
    name: str,
    *,
    parent_widths: tuple[int, ...],
    output_width: int,
    calibration_rows: int,
    functions: JointFunctionPrior,
    prior: ParentCompositionPrior,
) -> ParentFunctionPlan:
    with context.transaction(name) as operation:
        if prior.source == "mitra2":
            mode = "concat" if len(parent_widths) == 1 else "residual-members"
        else:
            mode = (
                operation.categorical("mode", prior.mode_request).value
                if prior.mode == "sample"
                else prior.mode
            )
        aggregation = (
            "none"
            if mode == "concat"
            else (
                operation.categorical("aggregation", prior.aggregation_request).value
                if prior.aggregation == "sample"
                else prior.aggregation
            )
        )
        joint = (
            sample_joint_function(
                operation,
                "joint",
                input_width=sum(parent_widths),
                output_width=output_width,
                calibration_rows=calibration_rows,
                prior=functions,
            )
            if mode == "residual-members"
            else None
        )
        widths = (sum(parent_widths),) if mode == "concat" else parent_widths
        child_prior = (
            replace(functions, functions="r0-linear-projection-v1") if joint else functions
        )
        children = tuple(
            sample_joint_function(
                operation,
                f"parent:{i}",
                input_width=width,
                output_width=output_width,
                calibration_rows=calibration_rows,
                prior=child_prior,
            )
            for i, width in enumerate(widths)
        )
        reducer = (
            sample_joint_function(
                operation,
                "reducer",
                input_width=(len(children) + int(joint is not None)) * output_width,
                output_width=output_width,
                calibration_rows=calibration_rows,
                prior=replace(functions, functions="r0-neural-reducer-v1"),
            )
            if aggregation == "mlp"
            else None
        )
        result = ParentFunctionPlan(
            operation.operation, prior, parent_widths, mode, aggregation, children, joint, reducer
        )
        result.validate(operation.freeze())
        return result
