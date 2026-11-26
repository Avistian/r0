"""Evaluate recorded stochastic equations, preserving separate exogenous and scale traces."""

from typing import Any

import torch

from r0.modules.priors.lib.causal_activations import activate, activation_diagnostic
from r0.modules.priors.lib.causal_nodes import StochasticCausalNode


def stochastic_value(
    node: StochasticCausalNode,
    raw: torch.Tensor,
    parents: torch.Tensor | None,
    node_index: int,
    diagnostics: list[dict[str, Any]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    affine = torch.zeros_like(raw) + torch.tensor(node.bias, dtype=raw.dtype, device=raw.device)
    if parents is not None:
        if diagnostics is not None:
            entry = activation_diagnostic(parents, node.input_activation)
            if entry is not None:
                diagnostics.append({"node": node_index, "stage": "input", **entry})
        affine = affine + activate(parents, node.input_activation) @ torch.tensor(
            node.weights, dtype=raw.dtype, device=raw.device
        )
    if not torch.isfinite(affine).all():
        raise ValueError("nonfinite pre-noise affine response; task was not retried")
    response = node.noise_response
    multiplier = torch.full_like(raw, response.offset)
    if response.strength:
        multiplier = multiplier + response.strength * affine.abs().pow(response.power)
    scale = multiplier * torch.tensor(node.noise_scale, dtype=raw.dtype, device=raw.device)
    if not torch.isfinite(scale).all():
        raise ValueError("nonfinite signal-dependent noise scale; task was not retried")
    innovation = scale * raw
    argument = affine + innovation if node.noise_position == "before_output" else affine
    if diagnostics is not None:
        entry = activation_diagnostic(argument, node.output_activation)
        if entry is not None:
            diagnostics.append({"node": node_index, "stage": "output", **entry})
    value = activate(argument, node.output_activation)
    if node.noise_position == "after_output":
        value = value + innovation
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite stochastic causal value; task was not retried")
    return value, innovation, scale
