"""Named scalar laws and their frozen source-function parameter contracts."""

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.parameter_preferences import NumericalParameter

type FunctionParameters = tuple[tuple[str, float | bool], ...]
type FunctionRequests = tuple[tuple[str, NumericalParameter | CategoricalParameter], ...]


def gp_scale_request() -> NumericalParameter:
    return NumericalParameter("rand_gen_gp_scale", 0.5, 10.0, scale="log")


def tree_requests() -> FunctionRequests:
    return (
        ("trees", NumericalParameter("rt_n_trees", 1, 128, scale="log", integer="round")),
        ("depth", NumericalParameter("rt_depth", 1, 8, integer="source-clip-floor-v1")),
    )


def function_requests(
    family: str,
    rows: int,
    outputs: int,
    prior: JointFunctionPrior,
    parameters: FunctionParameters = (),
) -> FunctionRequests:
    if family == "tree":
        return tree_requests()
    if family == "mlp":
        return (
            (
                "width",
                NumericalParameter("nn_width", 1, 128, scale="log", integer="source-clip-floor-v1"),
            ),
            (
                "layers",
                NumericalParameter(
                    "nn_n_layers", 1, 4, scale="log", integer="source-clip-floor-v1"
                ),
            ),
            ("start_activation", CategoricalParameter("nn_start_act", ("false", "true"))),
            ("end_activation", CategoricalParameter("nn_end_act", ("false", "true"))),
        )
    if family == "gp":
        requests = (
            ("a", NumericalParameter("gp_a", 2.0, 20.0, scale="log")),
            (
                "product_kernel",
                CategoricalParameter("generalized_gp_use_product_kernel", ("false", "true")),
            ),
        )
        return (
            (*requests, ("scale", gp_scale_request()))
            if dict(parameters).get("product_kernel") is False
            else requests
        )
    if family == "disc":
        requests: FunctionRequests = ()
        if rows > 1 and prior.discretization_centers is None:
            requests = (
                (
                    "centers",
                    NumericalParameter(
                        "rand_discr_n_centers",
                        2,
                        min(rows, prior.max_discretization_cardinality),
                        scale="log",
                        integer="source-clip-floor-v1",
                    ),
                ),
            )
        return (*requests, ("distance_p", NumericalParameter("cat_disc_p", 0.5, 4.0, scale="log")))
    if family == "em":
        return (
            (
                "components",
                NumericalParameter(
                    "random_em_components",
                    2,
                    max(16, 2 * outputs) + 1,
                    scale="log",
                    integer="source-clip-floor-v1",
                ),
            ),
            ("spread_factor", NumericalParameter("gen_em_assignment_std_factor", 0.0, 1.0)),
            ("distance_p", NumericalParameter("gen_em_assignment_p_inv", 1.0, 4.0, scale="log")),
            ("distance_q", NumericalParameter("gen_em_assignment_q", 1.0, 2.0)),
        )
    return ()


def discretization_case(
    rows: int, prior: JointFunctionPrior, parameters: FunctionParameters
) -> tuple[int, str]:
    if rows == 1:
        return 1, "single-fit-row-v1"
    if prior.discretization_centers is not None:
        return min(rows, prior.discretization_centers), "explicit-capped-count-v1"
    rule = (
        "sampled-degenerate-count-v1"
        if min(rows, prior.max_discretization_cardinality) == 2
        else "sampled-source-count-v1"
    )
    return int(dict(parameters)["centers"]), rule


def validate_parameters(parameters: FunctionParameters, requests: FunctionRequests) -> None:
    if (
        type(parameters) is not tuple
        or len(parameters) != len(requests)
        or any(type(pair) is not tuple or len(pair) != 2 for pair in parameters)
    ):
        raise ValueError("function parameters must retain exactly the ordered branch fields")
    for (role, value), (expected, request) in zip(parameters, requests, strict=True):
        if isinstance(request, CategoricalParameter):
            if role != expected or type(value) is not bool:
                raise ValueError("function Boolean parameter must retain its exact role and type")
            continue
        if role != expected or not finite_number(value):
            raise ValueError("function parameter role or finite numeric value is invalid")
        if request.integer != "none" and type(value) is not int:
            raise ValueError("function integer parameter must retain an integer value")
        if request.integer == "source-clip-floor-v1":
            low, high = min(request.low, request.high - 1), request.high - 1
        else:
            low, high = request.low, request.high
        if not low <= value <= high:
            raise ValueError("function parameter lies outside its source bounds")
