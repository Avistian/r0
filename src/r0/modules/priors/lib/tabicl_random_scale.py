"""Frozen shared or per-channel exp(2a)*(x+b), with original Gaussian draws."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits

if TYPE_CHECKING:
    from r0.modules.priors.lib.causal_activations import CausalActivation

RANDOM_SCALE_KIND = "tabicl-random-scale-v1"
_DEFAULT = CausalLimits()


def _scales(values: dict[str, float]) -> tuple[float, ...]:
    count = int(values["channels"]) if values["individual"] else 1
    try:
        result = tuple(math.exp(2 * values[f"normal_scale_{i}"]) for i in range(count))
    except OverflowError as exc:
        raise ValueError("nonfinite TabICL random scale; not clipped") from exc
    if any(not math.isfinite(scale) for scale in result):
        raise ValueError("nonfinite TabICL random scale; not clipped")
    return result


def validate_random_scale(values: dict[str, float]) -> None:
    width, individual = values.get("channels"), values.get("individual")
    if (
        type(width) is not int
        or not 1 <= width < 2**63
        or type(individual) is not int
        or individual not in (0, 1)
    ):
        raise ValueError("random scale requires a positive channel width and 0/1 individual flag")
    count = width if individual else 1
    if len(values) != 2 + 2 * count:
        raise ValueError("random scale requires every original Gaussian draw")
    if set(values) != {"channels", "individual"} | {
        f"{name}_{i}" for name in ("normal_scale", "normal_shift") for i in range(count)
    }:
        raise ValueError("random scale requires every original Gaussian draw")
    _scales(values)


def evaluate_random_scale(value: torch.Tensor, parameters: dict[str, float]) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != parameters["channels"]:
        raise ValueError("random scale input width differs from declared channels")
    scales = _scales(parameters)
    shifts = tuple(parameters[f"normal_shift_{i}"] for i in range(len(scales)))
    scale, shift = value.new_tensor(scales), value.new_tensor(shifts)
    shifted = value + shift
    result = scale * shifted
    overflow = ~torch.isfinite(shifted)
    if torch.any(overflow):
        result = torch.where(overflow, (value / 2 + shift / 2) * (scale * 2), result)
    return result


def sample_tabicl_random_scale(
    channels: int,
    *,
    individual: bool = False,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT,
) -> CausalActivation:
    from r0.modules.priors.lib.causal_activations import CausalActivation
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions

    if type(channels) is not int or not 1 <= channels < 2**63 or type(individual) is not bool:
        raise ValueError("random scale requires positive channels and a Boolean individual flag")
    count = channels if individual else 1
    if (
        type(limits) is not CausalLimits
        or channels > limits.node_channels
        or 2 * count + 2 > limits.parameters
        or 512 * count + 65536 > limits.working_bytes
    ):
        raise ValueError("random scale construction limit exceeded")
    with raw_transactions({"parameters": rng}, None) as (streams, _):
        a = streams["parameters"].normal(size=count)
        b = streams["parameters"].normal(size=count)
        values: list[tuple[str, float]] = [("channels", channels), ("individual", int(individual))]
        values += [(f"normal_scale_{i}", float(a[i])) for i in range(count)]
        values += [(f"normal_shift_{i}", float(b[i])) for i in range(count)]
        return CausalActivation(RANDOM_SCALE_KIND, tuple(values))
