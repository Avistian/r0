"""Evaluate structural state and explicit converter readouts with shared innovations."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_activations import activate, activation_diagnostic
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode, CausalNode, StochasticCausalNode
from r0.modules.priors.lib.causal_stochastic import stochastic_value
from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline, PipelineEvaluation


def causal_values(
    nodes: tuple[CausalNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    indices: Sequence[int] | None = None,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
    exogenous_values: list[torch.Tensor] | None = None,
    noise_scales: list[torch.Tensor] | None = None,
    pipelines: tuple[CausalNodePipeline, ...] | None = None,
    readout_values: list[torch.Tensor] | None = None,
    pipeline_steps: list[list[tuple[str, torch.Tensor]]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay selected original rows without retaining all intermediate task values."""
    values: list[torch.Tensor] = []
    innovations: list[torch.Tensor] = []
    for i, node in enumerate(nodes):
        pipeline = (
            PipelineEvaluation(
                pipelines[i], i, diagnostics, record_steps=pipeline_steps is not None
            )
            if pipelines is not None
            else None
        )
        noise = (
            sample_exogenous(node.exogenous, rows, len(node.bias), rng, i)
            if isinstance(node, StochasticCausalNode)
            else np.column_stack(
                [rng("innovation", i, j).normal(size=rows) for j in range(len(node.bias))]
            )
        )
        if indices is not None:
            noise = noise[list(indices)]
        if exogenous_values is not None:
            exogenous_values.append(torch.tensor(noise, dtype=torch.float64, device=device))
        if isinstance(node, StochasticCausalNode):
            parent_values = (
                torch.cat([values[p] for p in node.parents], dim=1) if node.parents else None
            )
            value, innovation, scale = stochastic_value(
                node,
                torch.tensor(noise, dtype=torch.float64, device=device),
                parent_values,
                i,
                diagnostics,
                before_noise=pipeline.before_noise if pipeline is not None else None,
            )
            if noise_scales is not None:
                noise_scales.append(scale)
            value, readout = pipeline.finish(value) if pipeline is not None else (value, value)
            if readout_values is not None:
                readout_values.append(readout)
            if pipeline_steps is not None:
                pipeline_steps.append(pipeline.steps if pipeline is not None else [])
            values.append(value)
            innovations.append(innovation)
            continue
        if noise_scales is not None:
            noise_scales.append(
                torch.tensor(node.noise_scale, dtype=torch.float64, device=device).expand(
                    len(noise), -1
                )
            )
        innovation = torch.tensor(noise, dtype=torch.float64, device=device) * torch.tensor(
            node.noise_scale, dtype=torch.float64, device=device
        )
        value = innovation + torch.tensor(node.bias, dtype=torch.float64, device=device)
        if isinstance(node, ActivatedCausalNode) and node.noise_position == "after_output":
            value = torch.zeros_like(innovation) + torch.tensor(
                node.bias, dtype=torch.float64, device=device
            )
        if node.parents:
            parents = torch.cat([values[p] for p in node.parents], dim=1)
            if isinstance(node, ActivatedCausalNode):
                if diagnostics is not None:
                    diagnostic = activation_diagnostic(parents, node.input_activation)
                    if diagnostic is not None:
                        diagnostics.append({"node": i, "stage": "input", **diagnostic})
                parents = activate(parents, node.input_activation)
            value = value + parents @ torch.tensor(node.weights, dtype=torch.float64, device=device)
        if isinstance(node, ActivatedCausalNode):
            if diagnostics is not None:
                diagnostic = activation_diagnostic(value, node.output_activation)
                if diagnostic is not None:
                    diagnostics.append({"node": i, "stage": "output", **diagnostic})
            value = activate(value, node.output_activation)
            if pipeline is not None:
                value = pipeline.before_noise(value)
            if node.noise_position == "after_output":
                value = value + innovation
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite causal node {i}; task was not retried")
        value, readout = pipeline.finish(value) if pipeline is not None else (value, value)
        if readout_values is not None:
            readout_values.append(readout)
        if pipeline_steps is not None:
            pipeline_steps.append(pipeline.steps if pipeline is not None else [])
        values.append(value)
        innovations.append(innovation)
    return values, innovations
