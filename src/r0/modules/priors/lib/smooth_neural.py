"""Independent bounded GP and random-network diagnostic representatives."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def gp_draw(
    rows: int, *, rng: np.random.Generator, observations: np.random.Generator
) -> tuple[Array, Array, dict[str, object], dict[str, object]]:
    knots = np.sort(rng.uniform(-2.0, 2.0, 8))
    distance = (knots[:, None] - knots[None, :]) / 0.6
    covariance = np.exp(-0.5 * distance**2) + 1e-4 * np.eye(8)
    factor = np.linalg.cholesky(covariance)
    z = rng.normal(size=8)
    values = factor @ z
    indices = observations.integers(0, 8, rows)
    parameters: dict[str, object] = {
        "domain": "finite sampled support; observations resample support points",
        "knots": knots.tolist(),
        "values": values.tolist(),
        "standard_normal": z.tolist(),
        "cholesky": factor.tolist(),
        "kernel": "rbf",
        "length_scale": 0.6,
        "signal_variance": 1.0,
        "diagonal_component_variance": 1e-4,
        "relevant": [0],
        "input": "uniform over eight sampled support coordinates",
    }
    return knots[indices, None], values[indices], parameters, {"support_indices": indices.tolist()}


def mlp_draw(x: Array, *, rng: np.random.Generator) -> tuple[Array, dict[str, object]]:
    depth = int(rng.integers(1, 3))
    widths = [int(rng.integers(4, 9)) for _ in range(depth)]
    sizes = [x.shape[1], *widths, 1]
    activation = str(rng.choice(["tanh", "relu"]))
    weights: list[list[list[float]]] = []
    hidden = x.copy()
    for index, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        matrix = rng.normal(0.0, 1 / np.sqrt(inputs), (inputs, outputs))
        weights.append(matrix.tolist())
        hidden = hidden @ matrix
        if index != len(sizes) - 2:
            hidden = np.tanh(hidden) if activation == "tanh" else np.maximum(hidden, 0.0)
    return hidden[:, 0], {
        "weights": weights,
        "activation": activation,
        "widths": widths,
        "bias": "none",
        "relevant": list(range(x.shape[1])),
        "input": "normal(0,1)",
    }
