"""Recorded nonnegative weight-vector laws, separate from realized matrix entries."""

import math
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number


@dataclass(frozen=True, slots=True)
class UniformWeightVector:
    draws: tuple[float, ...] = (0.5,)
    version: str = "uniform-weight-vector-v1"

    def __post_init__(self) -> None:
        if self.version != "uniform-weight-vector-v1" or (
            type(self.draws) is not tuple
            or not self.draws
            or any(not finite_number(v) or not 0 <= v < 1 for v in self.draws)
        ):
            raise ValueError("uniform weight vectors require a nonempty finite tuple in [0,1)")

    @property
    def parameter_count(self) -> int:
        return len(self.draws)

    def values(self) -> tuple[float, ...]:
        return self.draws

    def log_values(self) -> tuple[float, ...]:
        return tuple(math.log(v) if v else -math.inf for v in self.draws)


def weight_vector_from_dict(data: dict[str, Any]) -> UniformWeightVector:
    fields = dict(data)
    if "draws" in fields:
        fields["draws"] = tuple(fields["draws"])
    return UniformWeightVector(**fields)
