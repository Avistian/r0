"""Source-law oracle: no production construction/evaluation for expected answers."""

import numpy as np


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=2e-12)


def gaussian_matrix(plan, raw, actual):
    shape = (plan.batch, plan.rows, plan.columns)
    draws = raw.normal(size=shape)
    jitter = raw.normal(size=shape)
    for j, member in enumerate(actual.matrices):
        np.testing.assert_array_equal(member.source.draws, draws[j])
        np.testing.assert_array_equal(member.jitter, jitter[j])
    noisy = draws + 1e-6 * jitter
    result = noisy / (np.linalg.norm(noisy, axis=-1, keepdims=True) + 1e-6)
    close(actual.materialize(), result)
    return result


def oracle(plan, result, sample, raw, fit):
    """Return independent query evaluator; verify retained raw order and fit inputs."""
    p = dict(plan.parameters)
    family = plan.family
    if family == "prod":
        left = oracle(plan.children[0], result.left, sample, raw, fit)
        right = oracle(plan.children[1], result.right, sample, raw, fit)
        return lambda query: left(query) * right(query)
    if family == "lin":
        matrix = gaussian_matrix(plan.matrices[0], raw, result.matrix)[0]
        return lambda query: query @ matrix.T
    if family == "quad":
        selected = np.arange(plan.input_width)
        if plan.input_width > 20:
            permutation = raw.permutation(plan.input_width)
            np.testing.assert_array_equal(result.permutation, permutation)
            selected = permutation[:20]
        else:
            assert result.permutation == ()
        matrices = gaussian_matrix(plan.matrices[0], raw, result.matrix)

        def quadratic(query):
            z = np.column_stack((query[:, selected], np.ones(len(query))))
            return np.stack([np.sum((z @ q) * z, axis=1) for q in matrices], axis=1)

        return quadratic
    if family == "gp":
        i, k = plan.input_width, plan.prior.frequencies
        uniform = raw.random((i if p["product_kernel"] else 1, k))
        np.testing.assert_array_equal(result.uniform, uniform)
        frequency = np.clip(uniform, 1e-6, 1 - 1e-6) ** (1 / (1 - p["a"])) - 1
        if not p["product_kernel"]:
            directions = raw.normal(size=(i, k))
            factor = raw.normal(size=(i, i))
            weights = raw.random((1, i))[0]
            np.testing.assert_array_equal(result.directions, directions)
            np.testing.assert_array_equal(result.factor, factor)
            np.testing.assert_array_equal(result.weights.draws, weights)
            direction = directions / np.linalg.norm(directions, axis=0)
            frequency = (p["scale"] * weights[:, None] * factor) @ (direction * frequency)
        phase = raw.random(k)
        coeff = raw.normal(size=(k, plan.output_width))
        np.testing.assert_array_equal(result.phase_uniform, phase)
        np.testing.assert_array_equal(result.coefficient_gaussian, coeff)
        close(result.function.frequencies, frequency)
        return lambda query: np.cos(query @ frequency + 2 * np.pi * phase) @ (coeff / np.sqrt(k))
    if family == "tree":
        mass = np.maximum(sample.std(axis=0), 1e-8)
        close(result.feature_weights, mass)
        prob = mass / mass.max()
        prob /= prob.sum()
        dims = fit.choice(plan.input_width, size=(p["trees"], p["depth"]), replace=True, p=prob)
        rows = fit.integers(len(sample), size=dims.shape)
        leaves = raw.normal(size=(p["trees"], 2 ** p["depth"], plan.output_width))
        np.testing.assert_array_equal(result.features, dims)
        np.testing.assert_array_equal(result.threshold_rows, rows)
        np.testing.assert_array_equal(result.leaf_gaussian, leaves)
        thresholds = sample[rows, dims]
        close(result.function.thresholds, thresholds)

        def tree(query):
            indices = ((query[:, dims] > thresholds) * (2 ** np.arange(p["depth"]))).sum(axis=2)
            return leaves[np.arange(p["trees"]), indices].mean(axis=1)

        return tree
    if family == "disc":
        permutation = fit.permutation(len(sample)) if len(sample) > 1 else ()
        np.testing.assert_array_equal(result.permutation, permutation)
        selected = permutation[: plan.center_count] if len(sample) > 1 else [0]
        centers = sample[selected]
        matrix = gaussian_matrix(plan.matrices[0], raw, result.matrix)[0]
        targets = centers @ matrix.T
        close(result.targets, targets)

        def discretize(query):
            distances = (np.abs(query[:, None, :] - centers) ** p["distance_p"]).sum(axis=2)
            return targets[distances.argmin(axis=1)]

        return discretize
    if family == "em":
        count = p["components"]
        anchors = fit.integers(len(sample), size=count)
        jitter = raw.normal(size=(count, plan.input_width))
        spread = raw.normal(size=count)
        centers = sample[anchors] + jitter
        scales = np.exp(p["spread_factor"] * spread)
        np.testing.assert_array_equal(result.anchor_rows, anchors)
        np.testing.assert_array_equal(result.center_gaussian, jitter)
        np.testing.assert_array_equal(result.spread_gaussian, spread)

        def assignment(query):
            distances = (np.abs(query[:, None, :] - centers) ** p["distance_p"]).sum(axis=2)
            distances = distances ** (1 / p["distance_p"])
            logits = (
                -0.5 * np.log(2 * np.pi) - np.log(scales) - (distances / scales) ** p["distance_q"]
            )
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            return weights / weights.sum(axis=1, keepdims=True)

        close(result.assignments, assignment(sample))
        matrix = gaussian_matrix(plan.matrices[0], raw, result.matrix)[0]
        return lambda query: assignment(query) @ matrix.T
    raise AssertionError(f"Unsupported oracle family: {family}")
