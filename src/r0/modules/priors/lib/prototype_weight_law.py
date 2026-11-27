"""Declared reconstructions of underspecified prototype-weight temperature laws."""

import math
from dataclasses import dataclass

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec


@dataclass(frozen=True, slots=True)
class PrototypeWeightLaw:
    mode: str = "multinomial-trial-temperature-r0"
    trials: int | None = 16
    beta: float = 1.0
    probabilities: tuple[float, ...] = ()
    max_trials: int = 1_000_000
    version: str = "prototype-weight-law-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "prototype-weight-law-v1"
            or self.mode
            not in (
                "multinomial-trial-temperature-r0",
                "multinomial-probability-temperature-r0",
                "dirichlet-concentration-r0",
            )
            or not finite_number(self.beta)
            or self.beta <= 0
            or type(self.max_trials) is not int
            or not 1 <= self.max_trials < 2**31
            or type(self.probabilities) is not tuple
            or len(self.probabilities) > 4096
            or any(not finite_number(v) or not 0 <= v <= 1 for v in self.probabilities)
            or self.probabilities
            and not math.isclose(math.fsum(self.probabilities), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("invalid prototype weight law")
        if self.mode == "dirichlet-concentration-r0":
            if self.trials is not None or self.max_trials != 1_000_000:
                raise ValueError("unused Dirichlet trial controls")
            if any(v == 0 for v in self.probabilities):
                raise ValueError("Dirichlet prototype probabilities must be positive")
        else:
            if type(self.trials) is not int or not 1 <= self.trials <= self.max_trials:
                raise ValueError("prototype trials must be positive bounded integers")
            self.effective_trials()

    def effective_trials(self) -> int:
        if self.trials is None:
            raise ValueError("Dirichlet weights have no trial count")
        product = (
            self.trials * self.beta
            if self.mode == "multinomial-trial-temperature-r0"
            else self.trials
        )
        if not math.isfinite(product) or product <= 0 or product > self.max_trials:
            raise ValueError("prototype temperature trial limit exceeded; no clipping")
        return math.ceil(product)

    def probability_vector(self, count: int) -> tuple[float, ...]:
        if type(count) is not int or not 1 <= count <= 4096:
            raise ValueError("prototype count must be in 1..4096")
        if self.probabilities and len(self.probabilities) != count:
            raise ValueError("prototype probabilities differ from bank count")
        values = self.probabilities or (1 / count,) * count
        if self.mode == "multinomial-probability-temperature-r0":
            # Subtract before division so very small beta does not overflow positive logits.
            logs = tuple(math.log(v) if v else -math.inf for v in values)
            largest = max(logs)
            values = tuple(
                math.exp((v - largest) / self.beta) if v != -math.inf else 0.0 for v in logs
            )
            total = math.fsum(values)
            values = tuple(v / total for v in values)
        return values

    def draw_spec(self, count: int, rows: int, owner: tuple[str, ...]) -> RawDrawSpec:
        probabilities = self.probability_vector(count)
        if self.mode == "dirichlet-concentration-r0":
            alpha = tuple(self.beta * count * p for p in probabilities)
            return RawDrawSpec(
                owner,
                "prototype-gamma",
                "gamma_vector",
                (rows, count),
                (("concentrations", alpha),),
            )
        return RawDrawSpec(
            owner,
            "prototype-counts",
            "multinomial",
            (rows, count),
            (("trials", self.effective_trials()), ("probabilities", probabilities)),
        )
