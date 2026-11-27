"""Raw kernel construction is global function custody, distinct from emitted task rows."""

from collections.abc import Callable
from dataclasses import asdict

import torch

from r0.modules.priors.lib.feature_conv_sampling_records import SampledFeatureConv1D


def convolution_construction_steps(
    sampled: SampledFeatureConv1D, *, prefix: str, add: Callable[..., None]
) -> dict[int, str]:
    draws: dict[int, list[str]] = {}
    for index, record in enumerate(sampled.raw):
        identifier = f"{prefix}-construction-draw-{index}"
        layer = int(record.owner[1])
        draws.setdefault(layer, []).append(identifier)
        metadata = asdict(record)
        metadata.pop("values")
        add(
            identifier,
            f"Frozen kernel construction: layer {layer} {record.role}",
            [],
            "retained raw draw from its declared owned construction stream",
            torch.tensor(
                [record.values], dtype=torch.int64 if record.dtype == "<i8" else torch.float64
            ),
            {**metadata, "scope": "frozen function construction; not an emitted dataset row"},
        )
    identifiers: dict[int, str] = {}
    for index, layer in enumerate(sampled.function.layers):
        identifier = f"{prefix}-construction-parameters-{index}"
        coefficients = [v for output in layer.kernel for taps in output for v in taps]
        coefficients.extend(layer.bias)
        add(
            identifier,
            f"Realized shared kernel and bias {index}",
            draws[index],
            "chosen law transform; kernel mean + scale * raw / declared fan-in factor; "
            "independent bias",
            torch.tensor([coefficients], dtype=torch.float64),
            {
                "scope": "frozen function construction; not an emitted dataset row",
                "choices": sampled.choices[index],
                "request": asdict(sampled.request.layers[index]),
                "kernel_shape": [
                    len(layer.kernel),
                    len(layer.kernel[0]),
                    layer.geometry.kernel_size,
                ],
                "bias_count": len(layer.bias),
                "layout": "flattened kernel then bias",
                "rademacher_transform": "2 * retained integer - 1",
            },
        )
        identifiers[index] = identifier
    return identifiers
