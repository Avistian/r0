"""Complete binding catalog for every matrix field controlled by the shared prior."""

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.matrix_preference_prior import (
    CausalMatrixPrior,
    component_request,
    distance_request,
    vector_requests,
)
from r0.modules.priors.lib.parameter_preferences import NumericalParameter
from r0.modules.priors.lib.transformed_weights import (
    GaussianWeights,
    NormalizedWeights,
    TransformedWeights,
)
from r0.modules.priors.lib.weight_matrices import (
    ImportanceFactorizedWeights,
    ImportanceWeights,
    LaplaceKernelWeights,
)
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
)

type Path = tuple[str | int, ...]
type Contract = tuple[str, Path, str, NumericalParameter | CategoricalParameter]


def matrix_contract(
    matrix: NormalizedWeights, prior: CausalMatrixPrior, path: Path, *, simple: bool = False
) -> list[Contract]:
    raw = matrix.source
    result: list[Contract] = [
        ("categorical", path + ("version",), "value", prior.family_request(simple=simple))
    ]

    def vector(value: WeightVector, at: Path, *, polynomial: bool) -> None:
        if polynomial:
            if type(value) is not PolynomialWeightVector:
                raise ValueError("matrix preference vector family disagrees with its prior")
            for field, request in zip(
                ("decay", "log_noise_scale"), vector_requests(len(value.draws)), strict=True
            ):
                result.append(("numerical", at + (field,), "value", request))
        elif type(value) is not UniformWeightVector:
            raise ValueError("matrix preference vector family disagrees with its prior")

    polynomial = prior.vector_family == "polynomial"
    if isinstance(raw, GaussianWeights):
        pass
    elif isinstance(raw, ImportanceWeights):
        for index, value in enumerate(raw.importance):
            vector(value, path + ("importance", index), polynomial=polynomial)
    elif isinstance(raw, ImportanceFactorizedWeights):
        vector(raw.diagonal, path + ("diagonal",), polynomial=polynomial)
    elif isinstance(raw, LaplaceKernelWeights):
        cloud = raw.point_construction
        if cloud is None:
            raise ValueError("preferred kernel matrices require their point construction")
        point_path = path + ("point_construction",)
        result.append(("categorical", point_path + ("law",), "value", prior.point_request()))
        if cloud.law == "gaussian-mixture":
            result.append(("numerical", point_path + ("centers",), "length", component_request()))
        else:
            result.append(
                ("categorical", point_path + ("base_law",), "value", prior.base_request())
            )
        vector(
            cloud.component_weights,
            point_path + ("component_weights",),
            polynomial=polynomial and cloud.law == "gaussian-mixture",
        )
        for index, value in enumerate(cloud.factor_weights):
            vector(value, point_path + ("factor_weights", index), polynomial=polynomial)
        result.append(("numerical", path + ("distance_scale",), "value", distance_request()))
    elif isinstance(raw, TransformedWeights):
        if simple:
            raise ValueError("preferred matrix transformation cannot nest recursively")
        result.extend(matrix_contract(raw.child, prior, path + ("child", "source"), simple=True))
    else:
        raise ValueError("unsupported preferred matrix construction")
    return result
