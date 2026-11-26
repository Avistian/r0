"""Reconstruct structural values and innovations for selected original rows."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal import (
    ActivatedCausalNode,
    StochasticCausalNode,
    causal_node_from_dict,
    causal_rng,
    causal_values,
)


def causal_steps(
    card: dict[str, Any],
    rows: int,
    indices: Sequence[int],
    add: Callable[..., None],
) -> torch.Tensor:
    nodes = tuple(causal_node_from_dict(node) for node in card["nodes"])

    def rng(*keys: object) -> np.random.Generator:
        return causal_rng(card["seed"], card["task_id"], *keys)

    exogenous: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    values, innovations = causal_values(
        nodes, rows, rng, "cpu", indices, exogenous_values=exogenous, noise_scales=scales
    )
    for i, (node, value, noise) in enumerate(zip(nodes, values, innovations, strict=True)):
        stochastic = isinstance(node, StochasticCausalNode)
        if stochastic:
            add(
                f"exogenous-{i}",
                f"Node {i} exogenous draw",
                [],
                "recorded exogenous distribution before structural-noise scaling",
                exogenous[i],
                {"distribution": asdict(node.exogenous), "row_indices": list(indices)},
            )
            add(
                f"noise-scale-{i}",
                f"Node {i} structural-noise scale",
                [f"causal-{p}" for p in node.parents],
                "noise_scale * (offset + strength * abs(pre-noise affine response)**power)",
                scales[i],
                {
                    "noise_scale": node.noise_scale,
                    "response": asdict(node.noise_response),
                    "weights": node.weights,
                    "bias": node.bias,
                    "input_activation": asdict(node.input_activation),
                },
            )
        add(
            f"innovation-{i}",
            f"Node {i} structural innovation",
            [f"exogenous-{i}", f"noise-scale-{i}"] if stochastic else [],
            "recorded row-wise scale * recorded exogenous draw"
            if stochastic
            else "noise_scale * recorded independently keyed standard Gaussian draw",
            noise,
            {"noise_scale": node.noise_scale, "row_indices": list(indices)},
        )
        equation = (
            "concat(parent channels) @ weights + bias + innovation"
            if node.parents
            else "bias + innovation"
        )
        parameters: dict[str, Any] = {"weights": node.weights, "bias": node.bias, "node": i}
        if isinstance(node, ActivatedCausalNode):
            affine = (
                "input_activation(concat(parent channels)) @ weights + bias"
                if node.parents
                else "bias"
            )
            equation = (
                f"output_activation({affine} + innovation)"
                if node.noise_position == "before_output"
                else f"output_activation({affine}) + innovation"
            )
            parameters.update(
                input_activation=asdict(node.input_activation),
                output_activation=asdict(node.output_activation),
                noise_position=node.noise_position,
            )
        add(
            f"causal-{i}",
            f"Causal node {i}",
            [*[f"causal-{p}" for p in node.parents], f"innovation-{i}"],
            equation,
            value,
            parameters,
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
