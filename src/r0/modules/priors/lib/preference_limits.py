"""Admission limits for retained sampling decisions, independent of matrix storage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreferenceLimits:
    max_draws: int = 10_000
    max_values: int = 100_000
    max_parameters: int = 1_000_000

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or not 1 <= n < 2**63
            for n in (self.max_draws, self.max_values, self.max_parameters)
        ):
            raise ValueError("preference limits must be positive integers below 2**63")

    def admit(self, *, draws: int, values: int, parameters: int) -> None:
        if any(type(n) is not int or n < 0 for n in (draws, values, parameters)):
            raise ValueError("preference admission counts must be nonnegative integers")
        if draws > self.max_draws:
            raise ValueError("parameter preference draw limit exceeded")
        if values > self.max_values:
            raise ValueError("parameter preference value limit exceeded")
        if parameters > self.max_parameters:
            raise ValueError("parameter preference parameter limit exceeded")
