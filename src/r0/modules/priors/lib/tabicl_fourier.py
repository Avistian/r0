"""Channelwise finite sine activations with explicit paper/code decay draws."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits

if TYPE_CHECKING:
    from r0.modules.priors.lib.causal_activations import CausalActivation

FOURIER_KINDS = {"tabicl-paper-fourier-v1", "tabicl-code-fourier-v1"}
_COMMON = {"input_scale", "input_shift", "output_scale", "output_shift"}
_DEFAULT = CausalLimits()


def validate_fourier_parameters(kind: str, values: dict[str, float]) -> None:
    count = values.get("terms")
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("Fourier activation requires 1..4096 explicit terms")
    required = {"terms", "decay_draw"} | {
        f"{name}_{i}" for name in ("frequency", "phase", "normal") for i in range(count)
    }
    if not required <= values.keys() or values.keys() - required - _COMMON:
        raise ValueError("Fourier activation parameters must retain every source draw")
    low, high = (0.7, 3.0) if kind == "tabicl-paper-fourier-v1" else (math.log(0.7), math.log(3.0))
    if not low <= values["decay_draw"] <= high:
        raise ValueError("Fourier decay draw lies outside the declared source law")
    if any(not 0 < values[f"frequency_{i}"] < count for i in range(count)):
        raise ValueError("Fourier frequency must be positive and below the declared term count")
    if any(not 0 <= values[f"phase_{i}"] < 2 * math.pi for i in range(count)):
        raise ValueError("Fourier phase lies outside [0,2pi)")


def evaluate_fourier_activation(value: torch.Tensor, parameters: dict[str, float]) -> torch.Tensor:
    count = int(parameters["terms"])
    exponent = -math.exp(parameters["decay_draw"])
    logs = [exponent * math.log(parameters[f"frequency_{i}"]) for i in range(count)]
    largest = max(logs)
    factors = [math.exp(log - largest) for log in logs]
    norm = math.sqrt(math.fsum(f * f for f in factors))
    result = torch.zeros_like(value)
    # Avoid a rows-by-channels-by-frequencies temporary; graph buffers remain bounded.
    for i, factor in enumerate(factors):
        argument = value * parameters[f"frequency_{i}"] + parameters[f"phase_{i}"]
        coefficient = parameters[f"normal_{i}"] * (factor / norm)
        result = result + coefficient * torch.sin(argument)
    return result


def sample_tabicl_fourier_activation(
    *,
    rng: np.random.Generator,
    profile: str = "pinned-code",
    terms: int = 256,
    limits: CausalLimits = _DEFAULT,
) -> CausalActivation:
    """Sample once; standardization is a separate independently fitted graph stage.

    Recorded numeric parameters contain the original frequency, phase, normal and
    exponent draws. NumPy stream ownership is an r0 adaptation, not upstream bit parity.
    """
    from r0.modules.priors.lib.causal_activations import CausalActivation
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions

    if profile not in ("paper", "pinned-code") or type(terms) is not int or not 1 <= terms <= 4096:
        raise ValueError("invalid Fourier source profile or term count")
    if (
        type(limits) is not CausalLimits
        or 3 * terms + 2 > limits.parameters
        or 512 * terms + 65536 > limits.working_bytes
    ):
        raise ValueError("Fourier parameter sampling limit exceeded")
    kind = "tabicl-paper-fourier-v1" if profile == "paper" else "tabicl-code-fourier-v1"
    low, high = (0.7, 3.0) if profile == "paper" else (math.log(0.7), math.log(3.0))
    with raw_transactions({"parameters": rng}, None) as (streams, _):
        active = streams["parameters"]
        frequencies = active.uniform(0, terms, terms)
        phases = active.uniform(0, 2 * math.pi, terms)
        decay = float(active.uniform(low, high))
        normals = active.normal(size=terms)
        values: list[tuple[str, float]] = [("terms", terms), ("decay_draw", decay)]
        for i in range(terms):
            values.extend(
                (
                    (f"frequency_{i}", float(frequencies[i])),
                    (f"phase_{i}", float(phases[i])),
                    (f"normal_{i}", float(normals[i])),
                )
            )
        return CausalActivation(kind, tuple(values))
