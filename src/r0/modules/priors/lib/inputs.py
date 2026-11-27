"""Independent continuous inputs, including explicitly truncated heavy tails."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw


@dataclass(frozen=True, slots=True)
class InputDistribution:
    distribution: str = "gaussian"
    location: float = 0.0
    scale: float = 1.0
    df: float = 5.0
    mixture_gap: float = 2.0
    bound: float | None = None
    version: str = "inputs-v1"

    def __post_init__(self) -> None:
        if self.version != "inputs-v1" or self.distribution not in (
            "gaussian",
            "uniform",
            "student_t",
            "mixture",
        ):
            raise ValueError("unsupported input distribution")
        for value in (self.location, self.scale, self.df, self.mixture_gap):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("input parameters must be finite numbers")
        if self.scale <= 0 or self.df <= 0 or self.mixture_gap < 0:
            raise ValueError("input scale/df must be positive and mixture gap nonnegative")
        if self.bound is not None and (
            type(self.bound) not in (float, int)
            or not math.isfinite(self.bound)
            or self.bound < 0.1
        ):
            raise ValueError("truncation bound must be finite and at least 0.1")


def sample_inputs(
    spec: InputDistribution,
    rows: int,
    columns: int,
    rng: Callable[..., np.random.Generator],
    channel: str,
    *,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = (),
) -> NDArray[np.float64]:
    result = np.empty((rows, columns), dtype=np.float64)
    for j in range(columns):
        values_rng, signs_rng = rng(channel, j, "values"), rng(channel, j, "signs")

        def draw(
            size: int,
            values_rng: np.random.Generator = values_rng,
            signs_rng: np.random.Generator = signs_rng,
            draw_owner: tuple[str, ...] = (*owner, f"column:{j}"),
        ) -> NDArray[np.float64]:
            if spec.distribution == "uniform":
                return recorded_draw(
                    values_rng,
                    capture,
                    owner=draw_owner,
                    role="values",
                    distribution="uniform",
                    shape=(size,),
                    parameters=(("low", -math.sqrt(3)), ("high", math.sqrt(3))),
                )
            if spec.distribution == "student_t":
                return recorded_draw(
                    values_rng,
                    capture,
                    owner=draw_owner,
                    role="values",
                    distribution="student_t",
                    shape=(size,),
                    parameters=(("df", spec.df),),
                )
            values = recorded_draw(
                values_rng,
                capture,
                owner=draw_owner,
                role="values",
                distribution="normal",
                shape=(size,),
            )
            if spec.distribution == "mixture":
                signs = recorded_draw(
                    signs_rng,
                    capture,
                    owner=draw_owner,
                    role="mixture-components",
                    distribution="integers",
                    shape=(size,),
                    parameters=(("high", 2),),
                )
                values += spec.mixture_gap * (2 * signs - 1)
            return values

        if spec.bound is None:
            values = draw(rows)
        else:
            accepted: list[NDArray[np.float64]] = []
            count = 0
            for _ in range(128):
                batch = draw(max(rows, 256))
                batch = batch[np.abs(batch) <= spec.bound]
                accepted.append(batch)
                count += len(batch)
                if count >= rows:
                    break
            if count < rows:
                raise ValueError("input truncation exceeded its draw budget; task was not retried")
            values = np.concatenate(accepted)[:rows]
        result[:, j] = spec.location + spec.scale * values
    if not np.isfinite(result).all():
        raise ValueError("nonfinite sampled input; task was not retried")
    return result
