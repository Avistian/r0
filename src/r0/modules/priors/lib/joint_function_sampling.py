"""Transactional source function selectors and ordered named child choices."""

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.frozen_function_primitives import function_width
from r0.modules.priors.lib.joint_activation_plans import (
    JointActivationPlan,
    sample_joint_activation,
)
from r0.modules.priors.lib.joint_function_plans import JointFunctionPlan
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_function_requests import (
    FunctionParameters,
    discretization_case,
    function_requests,
    gp_scale_request,
)
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPlan, sample_joint_matrix
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_weight_plans import sample_joint_weights
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

_DEFAULT_PRIOR = JointFunctionPrior()


def _parameter(
    operation: JointSamplingContext, role: str, request: NumericalParameter | CategoricalParameter
) -> float | bool:
    if isinstance(request, NumericalParameter):
        return operation.numerical(role, request).values[0]
    return operation.categorical(role, request).value == "true"


def _mlp_children(
    operation: JointSamplingContext,
    input_width: int,
    output_width: int,
    prior: JointFunctionPrior,
    parameters: FunctionParameters,
) -> tuple[tuple[JointMatrixPlan, ...], tuple[JointActivationPlan, ...]]:
    values = dict(parameters)
    layers, width = int(values["layers"]), int(values["width"])
    widths = (input_width, *((width,) * (layers - 1)), output_width)
    matrices: list[JointMatrixPlan] = []
    activations: list[JointActivationPlan] = []
    if values["start_activation"]:
        activations.append(sample_joint_activation(operation, "activation:0", selector="standard"))
    for i in range(layers):
        matrices.append(
            sample_joint_matrix(
                operation, f"linear:{i}", rows=widths[i + 1], columns=widths[i], prior=prior.matrix
            )
        )
        if i < layers - 1 or values["end_activation"]:
            activations.append(
                sample_joint_activation(
                    operation, f"activation:{len(activations)}", selector="standard"
                )
            )
    return tuple(matrices), tuple(activations)


def sample_joint_function(
    context: JointSamplingContext,
    name: str,
    *,
    input_width: int,
    output_width: int,
    calibration_rows: int,
    prior: JointFunctionPrior = _DEFAULT_PRIOR,
    cheap: bool = False,
) -> JointFunctionPlan:
    if type(context) is not JointSamplingContext or type(prior) is not JointFunctionPrior:
        raise ValueError("function sampling requires a shared context and frozen prior")
    for value in (input_width, output_width, calibration_rows):
        function_width(value)
    request = prior.request(cheap=cheap)
    with context.transaction(name) as operation:
        slot = operation.categorical("family", request).value
        family = prior.family(slot, cheap=cheap)
        if family not in ("lin", "quad", "tree", "disc", "em", "gp", "mlp", "prod"):
            raise ValueError("unsupported function plan branch")
        parameters = tuple(
            (role, _parameter(operation, role, request))
            for role, request in function_requests(family, calibration_rows, output_width, prior)
        )
        weights = None
        if family == "gp" and dict(parameters)["product_kernel"] is False:
            weights = sample_joint_weights(
                operation,
                "weights",
                width=input_width,
                batch=1,
                family=prior.matrix.vector_family,
                scalarization=prior.matrix.scalarization,
            )
            parameters = (
                *parameters,
                ("scale", operation.numerical("scale", gp_scale_request()).values[0]),
            )
        center_count, center_rule = (
            discretization_case(calibration_rows, prior, parameters)
            if family == "disc"
            else (None, None)
        )
        matrices: tuple[JointMatrixPlan, ...] = ()
        activations: tuple[JointActivationPlan, ...] = ()
        children: tuple[JointFunctionPlan, ...] = ()
        if family == "prod":
            children = tuple(
                sample_joint_function(
                    operation,
                    child,
                    input_width=input_width,
                    output_width=output_width,
                    calibration_rows=calibration_rows,
                    prior=prior,
                    cheap=True,
                )
                for child in ("left", "right")
            )
        if family == "mlp":
            matrices, activations = _mlp_children(
                operation, input_width, output_width, prior, parameters
            )
        if family in ("lin", "quad", "disc", "em"):
            k = min(input_width, 20) + 1
            rows, columns, batch = (
                (output_width, input_width, 1) if family != "quad" else (k, k, output_width)
            )
            if family == "em":
                columns = int(dict(parameters)["components"])
            matrices = (
                sample_joint_matrix(
                    operation, "matrix", rows=rows, columns=columns, batch=batch, prior=prior.matrix
                ),
            )
        plan = JointFunctionPlan(
            operation.operation,
            input_width,
            output_width,
            calibration_rows,
            slot,
            prior,
            cheap,
            matrices,
            parameters,
            center_count,
            center_rule,
            weights,
            activations,
            children,
        )
        plan.validate(operation.freeze())
    return plan
