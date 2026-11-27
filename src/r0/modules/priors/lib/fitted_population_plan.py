"""Explicit fitting populations; source contracts and r0 choices stay distinct."""

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan


@dataclass(frozen=True, slots=True)
class PopulationTreePlan(FittedTreePlan):
    """The backend options of a fitted tree, admitting real support sets of size one."""

    version: str = "population-fitted-tree-plan-v1"

    def __post_init__(self) -> None:
        if self.version != "population-fitted-tree-plan-v1" or type(self.rows) is not int:
            raise ValueError("population tree plan requires its version and integer rows")
        if not 1 <= self.rows <= 1024:
            raise ValueError("population tree fitting requires 1..1024 actual rows")
        fields = asdict(self)
        fields.update(rows=max(32, self.rows), version="independent-fitted-tree-plan-v1")
        FittedTreePlan(**fields)


@dataclass(frozen=True, slots=True)
class MixedFitPopulation:
    """Mitra-inspired disclosed laws with explicit, otherwise unspecified settings.

    Geometric draws are rejected above the resource cap, never clipped or redrawn.
    Category probabilities may be uniform, or sampled from a symmetric Dirichlet.
    The latter is an explicitly named r0 extension.
    """

    categorical_fraction: tuple[float, float] = (0.0, 1.0)
    gaussian_mean: float = 0.0
    gaussian_scale: float = 1.0
    target_gaussian_mean: float = 0.0
    target_gaussian_scale: float = 1.0
    geometric_probability: float = 0.5
    maximum_cardinality: int = 100
    probabilities: str = "uniform"
    concentration: float = 1.0
    target_cardinality: str = "declared"
    version: str = "mitra-inspired-explicit-mixed-fit-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "mitra-inspired-explicit-mixed-fit-v1"
            or type(self.categorical_fraction) is not tuple
            or len(self.categorical_fraction) != 2
            or any(not finite_number(v) for v in self.categorical_fraction)
            or not 0 <= self.categorical_fraction[0] <= self.categorical_fraction[1] <= 1
            or any(
                not finite_number(v)
                for v in (
                    self.gaussian_mean,
                    self.gaussian_scale,
                    self.target_gaussian_mean,
                    self.target_gaussian_scale,
                    self.geometric_probability,
                    self.concentration,
                )
            )
            or self.gaussian_scale <= 0
            or self.target_gaussian_scale <= 0
            or not 0 < self.geometric_probability <= 1
            or self.concentration <= 0
            or type(self.maximum_cardinality) is not int
            or not 1 <= self.maximum_cardinality <= 100
            or self.probabilities not in ("uniform", "dirichlet-r0")
            or self.target_cardinality not in ("declared", "geometric")
        ):
            raise ValueError("invalid explicit mixed fitting population")
