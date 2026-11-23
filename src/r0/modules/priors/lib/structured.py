"""Small explicit causal motifs and piecewise rules for diagnostic inspection."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def scm_draw(
    x: Array, *, rng: np.random.Generator, observations: np.random.Generator
) -> tuple[Array, Array, dict[str, object], dict[str, object]]:
    motif = str(rng.choice(["chain", "fork", "collider"]))
    parents = {"chain": [[], [0], [1]], "fork": [[], [0], [0]], "collider": [[], [], [0, 1], [2]]}[
        motif
    ]
    weights = [rng.uniform(-1.5, 1.5, len(p)).tolist() for p in parents]
    noise_scales = [1.0 if not p else 0.1 for p in parents]
    noise = observations.normal(size=(len(x), len(parents))) * noise_scales
    nodes = np.empty_like(noise)
    for i, incoming in enumerate(parents):
        nodes[:, i] = noise[:, i]
        if incoming:
            nodes[:, i] += np.tanh(nodes[:, incoming] @ np.asarray(weights[i]))
    observed = ([1] if motif in ("chain", "fork") else [0, 1])[: x.shape[1]]
    features = x.copy()
    features[:, : len(observed)] = nodes[:, observed]
    return (
        features,
        nodes[:, -1],
        {
            "motif": motif,
            "parents": parents,
            "weights": weights,
            "node_noise_std": noise_scales,
            "observed_nodes": observed,
            "hidden_nodes": [i for i in range(len(parents) - 1) if i not in observed],
            "target_node": len(parents) - 1,
            "activation": "tanh",
            "relevant": list(range(len(observed))),
            "input": "SCM observed nodes plus independent normal(0,1) nuisance features",
        },
        {"nodes": nodes.tolist(), "node_noise": noise.tolist()},
    )


def tree_draw(
    x: Array, *, rng: np.random.Generator
) -> tuple[Array, dict[str, object], dict[str, object]]:
    depth = int(rng.integers(1, 4))
    internal = 2**depth - 1
    features = rng.integers(0, x.shape[1], internal)
    thresholds = rng.uniform(-1.0, 1.0, internal)
    leaves = rng.normal(size=2**depth)
    node = np.zeros(len(x), dtype=np.int64)
    for _ in range(depth):
        right = x[np.arange(len(x)), features[node]] > thresholds[node]
        node = 2 * node + 1 + right
    indices = node - internal
    return (
        leaves[indices],
        {
            "depth": depth,
            "split_features": features.tolist(),
            "thresholds": thresholds.tolist(),
            "leaves": leaves.tolist(),
            "relevant": sorted(set(int(v) for v in features)),
            "input": "normal(0,1)",
            "branch_rule": "left when x <= threshold",
        },
        {"leaf_indices": indices.tolist()},
    )
