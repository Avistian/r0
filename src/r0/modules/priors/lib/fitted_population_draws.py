"""Recorded mixed synthetic populations and direct categorical fake targets."""

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.fitted_population_plan import MixedFitPopulation, PopulationTreePlan
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw


def population_draws(
    plan: PopulationTreePlan,
    mixed: MixedFitPopulation | None,
    inputs: NDArray[np.float64] | None,
    streams: dict[str, np.random.Generator],
    capture: RawDrawLedger,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[int, ...], tuple[int, ...]]:
    """Return fit tables, backend seeds and categorical feature cardinalities."""

    def draw(
        stream: str,
        role: str,
        law: str,
        shape: tuple[int, ...],
        parameters: tuple[tuple[str, Any], ...] = (),
    ) -> NDArray[Any]:
        return recorded_draw(
            streams[stream],
            capture,
            owner=("population-fit",),
            role=role,
            distribution=law,
            shape=shape,
            parameters=parameters,
        )

    def cardinality(stream: str, role: str) -> int:
        assert mixed is not None
        u = float(draw(stream, role, "random", ()))
        if mixed.geometric_probability == 1:
            return 1
        ratio = math.log1p(-u) / math.log1p(-mixed.geometric_probability)
        if ratio >= mixed.maximum_cardinality:
            raise ValueError("geometric category draw exceeds admitted cap; no retry or clipping")
        return 1 + math.floor(ratio)

    def categories(stream: str, role: str, count: int) -> NDArray[np.float64]:
        assert mixed is not None
        weights = np.ones(count, dtype=np.float64)
        if mixed.probabilities == "dirichlet-r0":
            weights = draw(
                stream, role + "-mass", "gamma", (count,), (("concentration", mixed.concentration),)
            )
            if not np.isfinite(weights.sum()) or weights.sum() <= 0:
                raise ValueError("undefined categorical mass; no retry")
        probabilities = tuple(map(float, weights / weights.sum()))
        return draw(
            stream,
            role,
            "choice",
            (plan.rows,),
            (("count", count), ("probabilities", probabilities)),
        ).astype(np.float64)

    cardinalities = [0] * plan.input_width
    x = inputs
    if mixed is not None:
        if inputs is not None:
            raise ValueError("mixed fit population cannot consume episode or ancestor inputs")
        low, high = mixed.categorical_fraction
        fraction = (
            low
            if low == high
            else float(
                draw("input", "categorical-fraction", "uniform", (), (("low", low), ("high", high)))
            )
        )
        count = math.floor(fraction * (plan.input_width + 1))
        if count > plan.input_width:
            raise ValueError("categorical feature count exceeds width; no silent clipping")
        order = draw(
            "input",
            "categorical-indices",
            "permutation",
            (plan.input_width,),
            (("count", plan.input_width),),
        )
        chosen = set(map(int, order[:count]))
        x = np.empty((plan.rows, plan.input_width), dtype=np.float64)
        for column in range(plan.input_width):
            if column in chosen:
                cardinalities[column] = cardinality("input", f"cardinality-{column}")
                x[:, column] = categories("input", f"feature-{column}", cardinalities[column])
            else:
                x[:, column] = mixed.gaussian_mean + mixed.gaussian_scale * draw(
                    "input", f"feature-{column}", "normal", (plan.rows,)
                )
    if x is None:
        raise ValueError("fitting requires an explicit population")
    if plan.task == "regression":
        y = draw("target", "targets", "normal", (plan.rows, plan.output_width)).astype(np.float64)
        if mixed is not None:
            y = mixed.target_gaussian_mean + mixed.target_gaussian_scale * y
    elif mixed is None:
        y = draw(
            "target",
            "targets",
            "integers",
            (plan.rows, plan.output_width),
            (("high", plan.classes),),
        ).astype(np.float64)
    else:
        y = np.empty((plan.rows, plan.output_width), dtype=np.float64)
        for column in range(plan.output_width):
            count = (
                cardinality("target", f"target-cardinality-{column}")
                if mixed.target_cardinality == "geometric"
                else plan.classes
            )
            if count > plan.classes:
                raise ValueError("target geometric cardinality exceeds declared class cap")
            y[:, column] = categories("target", f"target-{column}", count)
    seeds = tuple(
        map(
            int,
            draw("backend", "backend-seeds", "integers", (plan.output_width,), (("high", 2**31),)),
        )
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite mixed fitting population; no retry")
    return x, y, seeds, tuple(cardinalities)
