"""Explicit local polynomial/log-Gaussian weight-vector draws shared by profiles."""

import math

import numpy as np

from r0.modules.priors.lib.weight_vectors import PolynomialWeightVector


def sample_polynomial_weights(width: int, rng: np.random.Generator) -> PolynomialWeightVector:
    low = 0.1 / math.log(1 + width)
    u = min(1.0, max(0.0, 1.2 * float(rng.random()) - 0.1))
    decay = low if u == 0 else 6.0 if u == 1 else math.exp(math.log(low) + u * math.log(6 / low))
    scale = math.exp(math.log(1e-4) + float(rng.random()) * math.log(10 / 1e-4))
    return PolynomialWeightVector(
        draws=tuple(float(v) for v in rng.normal(size=width)),
        decay=decay,
        log_noise_scale=scale,
        permutation=tuple(int(i) for i in rng.permutation(width)),
    )
