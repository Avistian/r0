"""Stable signal-dependent noise scaling shared by affine and functional nodes."""

import math

import torch

from r0.modules.priors.lib.exogenous import NoiseResponse


def signal_noise_scale(
    signal: torch.Tensor, noise_scale: tuple[float, ...], response: NoiseResponse
) -> torch.Tensor:
    base = torch.tensor(noise_scale, dtype=signal.dtype, device=signal.device)
    scale = torch.full_like(signal, response.offset)
    rescue = torch.zeros_like(signal, dtype=torch.bool)
    if response.strength:
        # A zero structural scale removes this path before evaluating its power.
        magnitude = torch.where(base == 0, 0.0, signal.abs())
        term = magnitude.pow(response.power)
        tiny = torch.finfo(signal.dtype).tiny
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
        selected_base = base.expand_as(signal)[rescue]
        log_term = (
            selected_base.log()
            + math.log(response.strength)
            + response.power * signal[rescue].abs().log()
        )
        scale[rescue] = selected_base * response.offset + log_term.exp()
    if not torch.isfinite(scale).all():
        raise ValueError("nonfinite signal-dependent noise scale; task was not retried")
    return scale
