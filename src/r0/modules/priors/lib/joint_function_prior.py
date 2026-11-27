"""Source RandomFunction catalogs with stable identities for duplicated slots."""

from dataclasses import dataclass

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPrior

_DEFAULT = ("mlp", "tree", "disc", "lin", "quad", "gp", "em", "prod")
_PRESETS = {"default": _DEFAULT, "tabpfnv2": ("mlp", "tree", "disc")}
_FORCED = {"r0-linear-projection-v1": "lin", "r0-neural-reducer-v1": "mlp"}
_CHEAP = ("gp", "tree", "disc", "lin", "quad")


def _expand(functions: str) -> tuple[str, ...]:
    if type(functions) is not str or not functions or len(functions) > 4096:
        raise ValueError("function catalog must be a nonempty bounded source token string")
    if functions in _FORCED:
        return (_FORCED[functions],)
    families: list[str] = []
    for token in functions.split(","):
        if token in _PRESETS:
            families.extend(_PRESETS[token])
        elif token in _DEFAULT:
            families.append(token)
        else:
            raise ValueError("unsupported source function token; whitespace is not stripped")
        if len(families) > 256:
            raise ValueError("function catalog exceeds the declared 256-slot categorical limit")
    return tuple(families)


@dataclass(frozen=True, slots=True)
class JointFunctionPrior:
    functions: str = "default"
    matrix: JointMatrixPrior = JointMatrixPrior()
    frequencies: int = 256
    max_discretization_cardinality: int = 256
    discretization_centers: int | None = None
    version: str = "joint-function-prior-v1"

    def __post_init__(self) -> None:
        _expand(self.functions)
        if self.version != "joint-function-prior-v1" or type(self.matrix) is not JointMatrixPrior:
            raise ValueError("function prior requires its version and a frozen matrix prior")
        if (
            type(self.frequencies) is not int
            or not 1 <= self.frequencies < 2**63
            or type(self.max_discretization_cardinality) is not int
            or not 1 <= self.max_discretization_cardinality < 2**63
            or (
                self.discretization_centers is not None
                and (
                    type(self.discretization_centers) is not int
                    or not 1 <= self.discretization_centers < 2**63
                )
            )
        ):
            raise ValueError("function counts must obey their explicit positive integer bounds")

    def request(self, *, cheap: bool = False) -> CategoricalParameter:
        if type(cheap) is not bool:
            raise ValueError("function selector scope must be Boolean")
        families = _CHEAP if cheap else _expand(self.functions)
        return CategoricalParameter(
            "cheap_random_function_fct_type"
            if cheap
            else self.functions
            if self.functions in _FORCED
            else "random_function_fct_type",
            tuple(f"function-slot-{i}-{family}-v1" for i, family in enumerate(families)),
        )

    def family(self, slot: str, *, cheap: bool = False) -> str:
        request = self.request(cheap=cheap)
        if type(slot) is not str or slot not in request.choices:
            raise ValueError("function slot is absent from its declared selector")
        families = _CHEAP if cheap else _expand(self.functions)
        return families[request.choices.index(slot)]
