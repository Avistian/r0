"""Frozen source function choices bound to their shared decision subtrees."""

from __future__ import annotations

from dataclasses import dataclass

from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.joint_activation_plans import JointActivationPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_requests import (
    FunctionParameters,
    discretization_case,
    function_requests,
    validate_parameters,
)
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.joint_weight_plans import JointWeightPlan
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class JointFunctionPlan:
    operation: tuple[str, ...]
    input_width: int
    output_width: int
    calibration_rows: int
    slot: str
    prior: JointFunctionPrior = JointFunctionPrior()
    cheap: bool = False
    matrices: tuple[JointMatrixPlan, ...] = ()
    parameters: FunctionParameters = ()
    center_count: int | None = None
    center_rule: str | None = None
    weights: JointWeightPlan | None = None
    activations: tuple[JointActivationPlan, ...] = ()
    children: tuple[JointFunctionPlan, ...] = ()
    version: str = "joint-function-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        for value in (self.input_width, self.output_width, self.calibration_rows):
            function_width(value)
        if (
            self.version != "joint-function-plan-v1"
            or type(self.prior) is not JointFunctionPrior
            or type(self.cheap) is not bool
        ):
            raise ValueError("function plan requires its version, prior and selector scope")
        if self.family not in ("lin", "quad", "tree", "disc", "em", "gp", "mlp", "prod"):
            raise ValueError("unsupported function plan branch")
        validate_parameters(
            self.parameters,
            function_requests(
                self.family, self.calibration_rows, self.output_width, self.prior, self.parameters
            ),
        )
        if self.family == "disc":
            expected = discretization_case(self.calibration_rows, self.prior, self.parameters)
            if (
                type(self.center_count) is not int
                or (self.center_count, self.center_rule) != expected
            ):
                raise ValueError("discretization must retain its exact count and source case")
        elif self.center_count is not None or self.center_rule is not None:
            raise ValueError("only discretization can own a center-count case")
        radial = self.family == "gp" and dict(self.parameters)["product_kernel"] is False
        if radial:
            weight = self.weights
            if (
                type(weight) is not JointWeightPlan
                or weight.operation != (*self.operation, "weights")
                or (weight.width, weight.batch) != (self.input_width, 1)
                or weight.family != self.prior.matrix.vector_family
                or weight.scalarization != self.prior.matrix.scalarization
            ):
                raise ValueError("radial GP requires its owned raw-factor weight plan")
        elif self.weights is not None:
            raise ValueError("only radial GP can own a raw-factor weight plan")
        if self.family == "prod":
            if (
                self.matrices != ()
                or self.activations != ()
                or type(self.children) is not tuple
                or len(self.children) != 2
                or any(type(child) is not JointFunctionPlan for child in self.children)
            ):
                raise ValueError("product requires exactly its two cheap function children")
            for name, child in zip(("left", "right"), self.children, strict=True):
                if (
                    not child.cheap
                    or child.operation != (*self.operation, name)
                    or child.prior != self.prior
                    or (child.input_width, child.output_width, child.calibration_rows)
                    != (self.input_width, self.output_width, self.calibration_rows)
                ):
                    raise ValueError("product child selector, geometry, prior or ownership differs")
            return
        if self.children != ():
            raise ValueError("only product can own nested function children")
        if self.family == "mlp":
            self._validate_mlp()
            return
        if self.activations != ():
            raise ValueError("only MLP can own standard activation stages")
        if self.family in ("tree", "gp"):
            if self.matrices != ():
                raise ValueError("tree and GP functions cannot own normalized matrix subtrees")
            return
        if (
            type(self.matrices) is not tuple
            or len(self.matrices) != 1
            or type(self.matrices[0]) is not JointMatrixPlan
        ):
            raise ValueError("function requires its single frozen matrix plan")
        matrix = self.matrices[0]
        k = min(self.input_width, 20) + 1
        geometry = (
            (self.output_width, self.input_width, 1)
            if self.family != "quad"
            else (k, k, self.output_width)
        )
        if self.family == "em":
            geometry = (self.output_width, int(dict(self.parameters)["components"]), 1)
        if (
            matrix.operation != (*self.operation, "matrix")
            or matrix.prior != self.prior.matrix
            or matrix.simple
            or (matrix.rows, matrix.columns, matrix.batch) != geometry
        ):
            raise ValueError("function matrix ownership, prior or source geometry differs")

    def _validate_mlp(self) -> None:
        values = dict(self.parameters)
        layers, width = int(values["layers"]), int(values["width"])
        count = layers - 1 + int(values["start_activation"]) + int(values["end_activation"])
        if (
            type(self.matrices) is not tuple
            or len(self.matrices) != layers
            or any(type(matrix) is not JointMatrixPlan for matrix in self.matrices)
            or type(self.activations) is not tuple
            or len(self.activations) != count
            or any(type(activation) is not JointActivationPlan for activation in self.activations)
        ):
            raise ValueError("MLP must retain every ordered linear and activation child")
        widths = (self.input_width, *((width,) * (layers - 1)), self.output_width)
        for i, matrix in enumerate(self.matrices):
            if (
                matrix.operation != (*self.operation, f"linear:{i}")
                or matrix.prior != self.prior.matrix
                or matrix.simple
                or (matrix.rows, matrix.columns, matrix.batch) != (widths[i + 1], widths[i], 1)
            ):
                raise ValueError("MLP matrix stage geometry, prior or ownership differs")
        for i, activation in enumerate(self.activations):
            if (
                activation.operation != (*self.operation, f"activation:{i}")
                or activation.selector != "standard"
            ):
                raise ValueError("MLP requires its owned standard activation stages")

    @property
    def ordered_children(
        self,
    ) -> tuple[JointMatrixPlan | JointActivationPlan | JointFunctionPlan, ...]:
        if self.family == "prod":
            return self.children
        if self.family != "mlp":
            return self.matrices
        values = dict(self.parameters)
        children: list[JointMatrixPlan | JointActivationPlan] = []
        index = 0
        if values["start_activation"]:
            children.append(self.activations[index])
            index += 1
        for i, matrix in enumerate(self.matrices):
            children.append(matrix)
            if i < len(self.matrices) - 1 or values["end_activation"]:
                children.append(self.activations[index])
                index += 1
        return tuple(children)

    @property
    def family(self) -> str:
        return self.prior.family(self.slot, cheap=self.cheap)

    @property
    def parameter_count(self) -> int:
        """Geometry and realized scalars, excluding prior configuration and the ledger.

        JSON admission separately counts every raw numeric field, including Boolean
        selectors and configured prior counts. This is not a graph storage bound.
        """
        return (
            3
            + len(self.parameters)
            + int(self.center_count is not None)
            + sum(matrix.parameter_count for matrix in self.matrices)
            + (self.weights.parameter_count if self.weights is not None else 0)
            + sum(activation.parameter_count for activation in self.activations)
            + sum(child.parameter_count for child in self.children)
        )

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        if not events or events[0].operation != self.operation or events[0].role != "family":
            raise ValueError("function plan requires its initial selector decision")
        draw = categorical_binding(events[0], decisions, self.prior.request(cheap=self.cheap))
        if draw.value != self.slot:
            raise ValueError("function slot differs from its recorded realization")
        expected = (events[0],)
        requests = function_requests(
            self.family, self.calibration_rows, self.output_width, self.prior, self.parameters
        )
        if len(events) < 1 + len(requests):
            raise ValueError("function branch is missing its scalar decisions")
        for (role, request), (_, value) in zip(requests, self.parameters, strict=True):
            if self.family == "gp" and role == "scale":
                assert self.weights is not None
                self.weights.validate(decisions)
                expected += operation_events(decisions, self.weights.operation)
            if len(expected) >= len(events):
                raise ValueError("function is missing an ordered parameter decision")
            event = events[len(expected)]
            if event.operation != self.operation or event.role != role:
                raise ValueError("function parameter decision order or ownership differs")
            observed = (
                numerical_binding(event, decisions, request).values[0]
                if isinstance(request, NumericalParameter)
                else categorical_binding(event, decisions, request).value == "true"
            )
            if observed != value:
                raise ValueError("function parameter differs from its recorded realization")
            expected += (event,)
        for child in self.ordered_children:
            child.validate(decisions)
            expected += operation_events(decisions, child.operation)
        if events != expected:
            raise ValueError("function must own exactly its ordered descendant decisions")
