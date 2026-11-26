"""Reconstruct structural values and innovations for selected original rows."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal import CausalNode, causal_values
from r0.modules.priors.lib.representatives import record_fingerprint


def causal_steps(
    card: dict[str, Any],
    rows: int,
    indices: Sequence[int],
    add: Callable[..., None],
) -> torch.Tensor:
    nodes = tuple(
        CausalNode(
            parents=tuple(node["parents"]),
            weights=tuple(tuple(row) for row in node["weights"]),
            bias=tuple(node["bias"]),
            noise_scale=tuple(node["noise_scale"]),
        )
        for node in card["nodes"]
    )

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint((card["version"], card["seed"], card["task_id"], keys)), 16)
        )

    values, innovations = causal_values(nodes, rows, rng, "cpu", indices)
    for i, (node, value, noise) in enumerate(zip(nodes, values, innovations, strict=True)):
        add(
            f"innovation-{i}",
            f"Node {i} structural innovation",
            [],
            "noise_scale * recorded independently keyed standard Gaussian draw",
            noise,
            {"noise_scale": node.noise_scale, "row_indices": list(indices)},
        )
        add(
            f"causal-{i}",
            f"Causal node {i}",
            [*[f"causal-{p}" for p in node.parents], f"innovation-{i}"],
            "concat(parent channels) @ weights + bias + innovation"
            if node.parents
            else "bias + innovation",
            value,
            {"weights": node.weights, "bias": node.bias, "node": i},
        )
    raw = torch.stack([values[i][:, j] for i, j in card["target"]], dim=1)
    calibration = card["calibration"]
    scores = (raw - torch.tensor(calibration["center"], dtype=torch.float64)) / torch.tensor(
        calibration["scale"], dtype=torch.float64
    )
    add(
        "signal",
        "Causal target readout",
        list(dict.fromkeys(f"causal-{i}" for i, _ in card["target"])),
        "(declared target channels - fixed calibration center) / fixed scale",
        scores,
        {
            "target": card["target"],
            "normalization": card["normalization"],
            "calibration": calibration,
        },
    )
    return scores
