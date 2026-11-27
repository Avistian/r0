"""Explicit adapted native-domain priors within a shared named preference scope."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class NativeDomainPrior:
    kinds: tuple[str, ...] = ("continuous", "binary", "categorical")
    cardinalities: tuple[int, ...] = (2, 3, 4, 8)
    emissions: tuple[str, ...] = ("softmax", "argmax", "threshold")
    frequency_laws: tuple[str, ...] = ("uniform", "dirichlet", "logistic-normal")
    threshold_laws: tuple[str, ...] = ("normal-quantile", "uniform-quantile", "sorted-normal")
    continuous_width: int = 1
    concentration: tuple[float, float] = (0.1, 10.0)
    temperature: tuple[float, float] = (0.1, 3.0)
    threshold_scale: tuple[float, float] = (0.1, 3.0)
    version: str = "native-domain-prior-v1"

    def __post_init__(self) -> None:
        if self.version != (
            "native-asymmetric-domain-prior-v1"
            if type(self) is AsymmetricNativeDomainPrior
            else "native-domain-prior-v1"
        ):
            raise ValueError("unsupported native domain prior")
        for values, allowed in (
            (self.kinds, ("continuous", "binary", "categorical")),
            (self.emissions, ("softmax", "argmax", "threshold")),
            (
                self.frequency_laws,
                ("uniform", "dirichlet", "logistic-normal")
                + (("asymmetric-dirichlet",) if type(self) is AsymmetricNativeDomainPrior else ()),
            ),
            (self.threshold_laws, ("normal-quantile", "uniform-quantile", "sorted-normal")),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(v) is not str or v not in allowed for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("native prior requires distinct named supported options")
        if (
            type(self.cardinalities) is not tuple
            or not self.cardinalities
            or len(self.cardinalities) > 256
            or any(type(k) is not int or not 2 <= k < 2**31 for k in self.cardinalities)
            or len(set(self.cardinalities)) != len(self.cardinalities)
        ):
            raise ValueError(
                "native cardinality catalog requires 1..256 distinct bounded category counts"
            )
        if type(self.continuous_width) is not int or not 1 <= self.continuous_width < 2**31:
            raise ValueError("native continuous width must be a positive bounded integer")
        for role, bounds in (
            ("concentration", self.concentration),
            ("temperature", self.temperature),
            ("threshold_scale", self.threshold_scale),
        ):
            if type(bounds) is not tuple or len(bounds) != 2:
                raise ValueError("native numerical prior requires a pair of bounds")
            if (
                any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError("native numerical bounds must be finite positive and ordered")
            if bounds[0] != bounds[1]:
                NumericalParameter(f"native_{role}", *bounds, scale="log")


@dataclass(frozen=True, slots=True)
class AsymmetricNativeDomainPrior(NativeDomainPrior):
    """r0 extension: explicit per-cardinality Gamma concentration vectors."""

    frequency_laws: tuple[str, ...] = ("asymmetric-dirichlet",)
    concentration_vectors: tuple[tuple[float, ...], ...] = ()
    version: str = "native-asymmetric-domain-prior-v1"

    def __post_init__(self) -> None:
        NativeDomainPrior.__post_init__(self)
        required: set[int] = set(self.cardinalities) if "categorical" in self.kinds else set()
        if "binary" in self.kinds:
            required.add(2)
        vectors = self.concentration_vectors
        if (
            type(vectors) is not tuple
            or len(vectors) != len(required)
            or any(type(row) is not tuple or len(row) not in required for row in vectors)
            or {len(row) for row in vectors} != required
            or any(
                type(a) not in (int, float) or not math.isfinite(a) or a <= 0
                for row in vectors
                for a in row
            )
        ):
            raise ValueError(
                "asymmetric concentrations require one positive vector per reachable cardinality"
            )

    def concentrations(self, cardinality: int) -> tuple[float, ...]:
        return next(row for row in self.concentration_vectors if len(row) == cardinality)


@dataclass(frozen=True, slots=True)
class NativeDomainPlan:
    operation: tuple[str, ...]
    prior: NativeDomainPrior
    kind: str
    cardinality: int
    emission: str
    frequency_law: str = "uniform"
    concentration: float = 1.0
    temperature: float = 1.0
    threshold_law: str = "none"
    threshold_scale: float = 1.0
    version: str = "native-domain-plan-v1"

    @property
    def asymmetric_draws(self) -> int:
        return self.cardinality if self.frequency_law == "asymmetric-dirichlet" else 0

    @property
    def score_width(self) -> int:
        if self.kind == "continuous":
            return self.prior.continuous_width
        return 1 if self.emission == "threshold" else self.cardinality

    @property
    def state_width(self) -> int:
        return self.score_width if self.kind == "continuous" else self.cardinality


@dataclass(frozen=True, slots=True)
class AsymmetricNativeDomainPlan(NativeDomainPlan):
    concentrations: tuple[float, ...] = ()
    version: str = "native-asymmetric-domain-plan-v1"


def sample_native_domain(
    context: JointSamplingContext, name: str, prior: NativeDomainPrior
) -> NativeDomainPlan:
    """Fixed choices are explicit forced decisions; stochastic choices share cached names."""
    if type(context) is not JointSamplingContext or type(prior) not in (
        NativeDomainPrior,
        AsymmetricNativeDomainPrior,
    ):
        raise ValueError("native domain sampling requires its shared context and frozen prior")
    with context.transaction(name) as operation:

        def choose(role: str, choices: tuple[str, ...]) -> str:
            if len(choices) == 1:
                operation.forced_identity(role, reason="configured-singleton")
                return choices[0]
            return operation.categorical(
                role, CategoricalParameter(f"native_{role}", choices)
            ).value

        def number(role: str, bounds: tuple[float, float]) -> float:
            if bounds[0] == bounds[1]:
                operation.forced_identity(role, reason="configured-constant")
                return float(bounds[0])
            return float(
                operation.numerical(
                    role, NumericalParameter(f"native_{role}", *bounds, scale="log")
                ).values[0]
            )

        kind = choose("kind", prior.kinds)
        if kind == "continuous":
            return NativeDomainPlan(operation.operation, prior, kind, 0, "identity")
        cardinality = (
            2
            if kind == "binary"
            else int(choose("cardinality", tuple(str(k) for k in prior.cardinalities)))
        )
        if kind == "binary":
            operation.forced_identity("cardinality", reason="binary-domain-cardinality")
        emission = choose("emission", prior.emissions)
        frequency = choose("frequency_law", prior.frequency_laws)
        concentration = (
            number("concentration", prior.concentration)
            if frequency not in ("uniform", "asymmetric-dirichlet")
            else 1.0
        )
        temperature = number("temperature", prior.temperature) if emission == "softmax" else 1.0
        threshold = (
            choose("threshold_law", prior.threshold_laws) if emission == "threshold" else "none"
        )
        scale = number("threshold_scale", prior.threshold_scale) if emission == "threshold" else 1.0
        args = (
            operation.operation,
            prior,
            kind,
            cardinality,
            emission,
            frequency,
            concentration,
            temperature,
            threshold,
            scale,
        )
        if frequency == "asymmetric-dirichlet":
            assert isinstance(prior, AsymmetricNativeDomainPrior)
            operation.forced_identity("concentrations", reason="configured-cardinality-vector")
            return AsymmetricNativeDomainPlan(
                *args, concentrations=prior.concentrations(cardinality)
            )
        return NativeDomainPlan(*args)
