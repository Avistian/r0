"""Reconstruct structural values and innovations for selected original rows."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal import (
    ActivatedCausalNode,
    MatrixCausalNode,
    RootInputCausalNode,
    StochasticCausalNode,
    causal_node_from_dict,
    causal_rng,
    causal_values,
)
from r0.modules.priors.lib.node_pipeline import pipeline_from_dict


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
    root_values: list[torch.Tensor | None] = []
    readouts: list[torch.Tensor] = []
    pipeline_steps: list[list[tuple[str, torch.Tensor]]] = []
    pipelines = (
        tuple(pipeline_from_dict(p) for p in card["pipelines"]) if "pipelines" in card else None
    )
    values, innovations = causal_values(
        nodes,
        rows,
        rng,
        "cpu",
        indices,
        exogenous_values=exogenous,
        noise_scales=scales,
        root_values=root_values,
        pipelines=pipelines,
        readout_values=readouts,
        pipeline_steps=pipeline_steps if pipelines is not None else None,
    )
    for i, (node, value, noise) in enumerate(zip(nodes, values, innovations, strict=True)):
        stochastic = isinstance(node, StochasticCausalNode)
        root_parents: list[str] = []
        if isinstance(node, RootInputCausalNode):
            raw_root = root_values[i]
            assert raw_root is not None
            add(
                f"root-input-{i}",
                f"Node {i} root input",
                [],
                "independent root law before output activation and pipeline",
                raw_root,
                {
                    "distribution": asdict(node.root_exogenous),
                    "root_scale": node.root_scale,
                    "stream": node.root_version,
                    "row_indices": list(indices),
                },
            )
            root_parents = [f"root-input-{i}"]
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
                [f"causal-{p}" for p in node.parents] + root_parents,
                "noise_scale * (offset + strength * abs(pre-noise affine response)**power)",
                scales[i],
                {
                    "noise_scale": node.noise_scale,
                    "response": asdict(node.noise_response),
                    "weights": node.weights,
                    "bias": node.bias,
                    "input_activation": asdict(node.input_activation),
                    **(
                        {"root_scale": node.root_scale}
                        if isinstance(node, RootInputCausalNode)
                        else {}
                    ),
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
        if isinstance(node, MatrixCausalNode):
            parameters["weight_construction"] = asdict(node.weight_construction)
        if isinstance(node, ActivatedCausalNode):
            affine = (
                "input_activation(concat(parent channels)) @ weights + bias"
                if node.parents
                else "bias"
            )
            if isinstance(node, RootInputCausalNode):
                affine = "bias + root_scale * independent_root_input"
                parameters.update(
                    root_exogenous=asdict(node.root_exogenous), root_scale=node.root_scale
                )
            equation = (
                f"output_activation({affine} + innovation)"
                if node.noise_position == "before_output"
                else f"output_activation({affine}) + innovation"
            )
            if pipelines is not None:
                parameters["pipeline"] = asdict(pipelines[i])
                equation = "node_importance * (pipeline(output_activation(affine)) + innovation)"
            parameters.update(
                input_activation=asdict(node.input_activation),
                output_activation=asdict(node.output_activation),
                noise_position=node.noise_position,
            )
        node_parents = [*[f"causal-{p}" for p in node.parents], *root_parents, f"innovation-{i}"]
        if pipelines is not None:
            previous = [f"causal-{p}" for p in node.parents] + root_parents
            for name, stage_value in pipeline_steps[i]:
                key = f"pipeline-{i}-{name}"
                formula = {
                    "input": "output_activation(input_activation(parents) @ weights + bias)",
                    "channel-importance": "previous * channel_importance",
                    "l2": "previous * frozen L2 factor",
                    "converter-state": "converter state before innovation",
                    "converted-readout": "converter readouts before innovation",
                }.get(name, "apply frozen ordered stage")
                if name == "input" and isinstance(node, RootInputCausalNode):
                    formula = "output_activation(bias + root_scale * independent_root_input)"
                add(
                    key,
                    f"Node {i} {name}",
                    previous,
                    formula,
                    stage_value,
                    {"pipeline": asdict(pipelines[i])},
                )
                if name not in ("converter-state", "converted-readout"):
                    previous = [key]
            node_parents = [f"pipeline-{i}-converter-state", f"innovation-{i}"]
        add(
            f"causal-{i}",
            f"Causal node {i}",
            node_parents,
            equation,
            value,
            parameters,
        )
        if pipelines is not None:
            add(
                f"readout-{i}",
                f"Node {i} converted readout",
                ([f"pipeline-{i}-converted-readout"] if pipelines[i].converters else [])
                + ([f"causal-{i}"] if len(pipelines[i].converters) < pipelines[i].channels else []),
                "explicit converter output before innovation and node importance; "
                "other channels use final state",
                readouts[i],
                {"pipeline": asdict(pipelines[i])},
            )
    raw = torch.stack([readouts[i][:, j] for i, j in card["target"]], dim=1)
    calibration = card["calibration"]
    scores = (raw - torch.tensor(calibration["center"], dtype=torch.float64)) / torch.tensor(
        calibration["scale"], dtype=torch.float64
    )
    add(
        "signal",
        "Causal target readout",
        list(
            dict.fromkeys(
                f"readout-{i}" if pipelines is not None else f"causal-{i}"
                for i, _ in card["target"]
            )
        ),
        "(declared target channels - fixed calibration center) / fixed scale",
        scores,
        {
            "target": card["target"],
            "normalization": card["normalization"],
            "calibration": calibration,
        },
    )
    return scores
