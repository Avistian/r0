"""Independent finite NumPy equations for fixed numerical function records."""

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=2e-11, atol=2e-12):
        raise RuntimeError(f"{label}: numerical mismatch")


def evaluate(f, x):
    kind = f.version
    if kind == "frozen-linear-function-v1":
        return x @ np.asarray(f.matrix).T
    if kind == "frozen-quadratic-function-v1":
        z = np.column_stack((x[:, f.indices], np.ones(len(x))))
        return np.asarray(
            [
                [
                    sum(zr[i] * q[i][j] * zr[j] for i in range(len(zr)) for j in range(len(zr)))
                    for q in f.matrices
                ]
                for zr in z
            ]
        )
    if kind == "frozen-rff-function-v1":
        return np.cos(x @ np.asarray(f.frequencies) + f.phases) @ np.asarray(f.coefficients)
    if kind == "frozen-oblivious-tree-function-v1":
        return np.asarray(
            [
                np.mean(
                    [
                        table[
                            sum(
                                (row[col] > threshold) * 2**level
                                for level, (col, threshold) in enumerate(
                                    zip(cols, cuts, strict=True)
                                )
                            )
                        ]
                        for cols, cuts, table in zip(
                            f.features, f.thresholds, f.leaves, strict=True
                        )
                    ],
                    axis=0,
                )
                for row in x
            ]
        )
    if kind == "frozen-discretization-function-v1":
        centers = np.asarray(f.centers)
        distance = np.sum(np.abs(x[:, None] - centers) ** f.p, axis=2) ** (1 / f.p)
        return evaluate(f.linear, centers)[distance.argmin(axis=1)]
    if kind == "frozen-em-function-v1":
        distance = np.sum(np.abs(x[:, None] - np.asarray(f.centers)) ** f.p, axis=2) ** (1 / f.p)
        scales = np.asarray(f.scales)
        logits = -np.log(scales) - (distance / scales) ** f.q
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return evaluate(f.linear, weights / weights.sum(axis=1, keepdims=True))
    if kind == "frozen-product-function-v1":
        return evaluate(f.left, x) * evaluate(f.right, x)
    raise NotImplementedError(kind)


def base_oracle(base, rng, rows):
    uniform = base.family == "uniform" or base.covariance_base == "uniform"
    raw = rng.random((rows, base.width)) if uniform else rng.normal(size=(rows, base.width))
    radii = rng.random(rows) if base.family == "ball" else np.asarray([])
    values = 2 * raw - 1 if uniform else raw.copy()
    if base.family == "ball":
        values = raw / np.linalg.norm(raw, axis=1)[:, None] * radii[:, None] ** (1 / base.width)
    if base.family == "covariance":
        factor = np.asarray(base.gaussian_factor) * np.asarray(base.weights.values())[None, :]
        values = values @ factor.T
    return raw, radii, values
