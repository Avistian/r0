"""Source-weighted activation types with independently frozen per-layer parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits

if TYPE_CHECKING:
    from r0.modules.priors.lib.node_pipeline import CausalNodePipeline

_DEFAULT = CausalLimits()
_SIMPLE = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "selu",
    "silu",
    "relu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign_positive",
    "gaussian_bump",
    "exp",
    "sqrt_abs",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
)


def sample_tabicl_activation_pipelines(
    calibrations: tuple[NDArray[np.float64], ...],
    *,
    calibration_ids: tuple[str, ...],
    rng: np.random.Generator,
    random_functions: bool = True,
    scale: bool = True,
    diverse: bool = True,
    individual: bool = False,
    profile: str = "pinned-code",
    limits: CausalLimits = _DEFAULT,
) -> tuple[CausalNodePipeline, ...]:
    """Build one pipeline per independent layer sample; return frozen realizations.

    With diversity enabled, half the probability selects a shared activation type
    and half selects each layer independently. Random function parameters are
    always fresh per layer, including when their type is shared.
    """
    from r0.modules.priors.lib.calibrated_activations import calibrate_standardization
    from r0.modules.priors.lib.causal_activations import CausalActivation
    from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
    from r0.modules.priors.lib.raw_draw_capture import raw_transactions
    from r0.modules.priors.lib.tabicl_fourier import sample_tabicl_fourier_activation
    from r0.modules.priors.lib.tabicl_random_scale import sample_tabicl_random_scale

    if (
        any(type(v) is not bool for v in (random_functions, scale, diverse, individual))
        or profile not in ("pinned-code", "paper")
        or type(limits) is not CausalLimits
        or type(calibrations) is not tuple
        or not calibrations
        or type(calibration_ids) is not tuple
        or len(calibration_ids) != len(calibrations)
    ):
        raise ValueError("invalid TabICL selector options or calibration samples")
    if any(type(a) is not np.ndarray or a.ndim != 2 or a.dtype != np.float64 for a in calibrations):
        raise ValueError("selector calibrations must be float64 matrices")
    widths = tuple(a.shape[1] for a in calibrations)
    # Bound every possible realization before fitting statistics or sampling types.
    parameters = sum(
        (4 * w + 2 if scale and random_functions else 2 * w + 1)
        + (2 * (w if individual else 1) + 2 if scale else 0)
        + (770 if random_functions else 0)
        for w in widths
    )
    working = 65536 + 512 * parameters + 128 * max(a.size for a in calibrations)
    if (
        len(widths) > limits.nodes
        or sum(widths) > limits.total_channels
        or any(not 1 <= w <= limits.node_channels for w in widths)
        or parameters > limits.parameters
        or working > limits.working_bytes
    ):
        raise ValueError("TabICL selector aggregate construction limit exceeded")
    standards = tuple(
        calibrate_standardization(a, calibration_id=identifier, profile="tabicl", limits=limits)
        for a, identifier in zip(calibrations, calibration_ids, strict=True)
    )
    pool = _SIMPLE + (("fourier",) * 10 if random_functions else ())
    with raw_transactions({"parameters": rng}, None) as (streams, _):
        active = streams["parameters"]
        selected = int(active.integers(len(pool) * (2 if diverse else 1)))
        types = (
            tuple(pool[int(active.integers(len(pool)))] for _ in widths)
            if selected >= len(pool)
            else (pool[selected],) * len(widths)
        )
        pipelines: list[CausalNodePipeline] = []
        for sample, identifier, standard, kind in zip(
            calibrations, calibration_ids, standards, types, strict=True
        ):
            stages: list[CausalActivation] = []
            value = torch.tensor(sample, dtype=torch.float64)
            if scale:
                scaling = sample_tabicl_random_scale(
                    sample.shape[1], individual=individual, rng=active, limits=limits
                )
                stages.extend((standard, scaling))
                value = scaling.apply(standard.apply(value))
            if kind == "fourier":
                # The epsilon makes this second standardization significant even
                # after the outer standardization and random scale.
                inner = (
                    calibrate_standardization(
                        value.numpy(), calibration_id=identifier, profile="tabicl", limits=limits
                    )
                    if scale
                    else standard
                )
                stages.extend(
                    (
                        inner,
                        sample_tabicl_fourier_activation(
                            rng=active, profile=profile, limits=limits
                        ),
                    )
                )
            else:
                stages.append(CausalActivation(kind))
            pipelines.append(CausalNodePipeline(sample.shape[1], tuple(stages)))
        return tuple(pipelines)
