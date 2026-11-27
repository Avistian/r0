"""Evaluate recorded stochastic equations, preserving separate exogenous and scale traces."""

import math
from collections.abc import Callable
from typing import Any

import torch

from r0.modules.priors.lib.causal_activations import activate, activation_diagnostic
from r0.modules.priors.lib.causal_nodes import RootInputCausalNode, StochasticCausalNode


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
    response = node.noise_response
    base = torch.tensor(node.noise_scale, dtype=raw.dtype, device=raw.device)
    scale = torch.full_like(raw, response.offset)
    rescue = torch.zeros_like(raw, dtype=torch.bool)
    if response.strength:
        # A zero structural scale removes this path before evaluating its power.
        magnitude = torch.where(base == 0, 0.0, affine.abs())
        term = magnitude.pow(response.power)
        tiny = torch.finfo(raw.dtype).tiny
        rescue = (term < tiny) & (magnitude > 0) & (base > 0)
        term *= response.strength
        rescue |= (term < tiny) & (magnitude > 0) & (base > 0)
        scale += term
        del magnitude, term
    scale *= base
    if response.strength:
        rescue |= ~torch.isfinite(scale) & (base > 0)
    if torch.any(rescue):
        # Intermediate powers/products can overflow or underflow even when the
        # final positive term is representable. Keep ordinary arithmetic elsewhere.
        selected_base = base.expand_as(raw)[rescue]
        log_term = (
            selected_base.log()
            + math.log(response.strength)
            + response.power * affine[rescue].abs().log()
        )
        scale[rescue] = selected_base * response.offset + log_term.exp()
    if not torch.isfinite(scale).all():
        raise ValueError("nonfinite signal-dependent noise scale; task was not retried")
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
