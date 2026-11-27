"""Evaluate recorded stochastic equations, preserving separate exogenous and scale traces."""

from collections.abc import Callable
from typing import Any

import torch

from r0.modules.priors.lib.causal_activations import activate, activation_diagnostic
from r0.modules.priors.lib.causal_nodes import RootInputCausalNode, StochasticCausalNode
from r0.modules.priors.lib.signal_noise import signal_noise_scale


def stochastic_value(
    node: StochasticCausalNode,
    raw: torch.Tensor,
    parents: torch.Tensor | None,
    node_index: int,
    diagnostics: list[dict[str, Any]] | None,
    *,
    before_noise: Callable[[torch.Tensor], torch.Tensor] | None = None,
    root_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    affine = torch.zeros_like(raw) + torch.tensor(node.bias, dtype=raw.dtype, device=raw.device)
    if isinstance(node, RootInputCausalNode):
        assert root_input is not None
        affine = affine + root_input * root_input.new_tensor(node.root_scale)
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
    scale = signal_noise_scale(affine, node.noise_scale, node.noise_response)
    innovation = scale * raw
    argument = affine + innovation if node.noise_position == "before_output" else affine
    if diagnostics is not None:
        entry = activation_diagnostic(argument, node.output_activation)
        if entry is not None:
            diagnostics.append({"node": node_index, "stage": "output", **entry})
    value = activate(argument, node.output_activation)
    if before_noise is not None:
        value = before_noise(value)
    if node.noise_position == "after_output":
        value = value + innovation
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite stochastic causal value; task was not retried")
    return value, innovation, scale
