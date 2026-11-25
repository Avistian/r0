"""Evaluate recorded analytic functions without sampling or adapting targets."""

import json
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.training_task import TrainingTask


def evaluate_signal(task: TrainingTask, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluate the saved noiseless signal on a bounded float64 input matrix.

    Returns normalized latent scores on the input device. Targets and observed labels
    are never read. Direct GP draws need separately identified conditional inspection.
    """
    if (
        inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 20000
        or inputs.shape[1] != task.inputs.shape[1]
        or inputs.dtype != torch.float64
        or inputs.device.type not in ("cpu", "cuda")
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError(
            "inputs must be finite float64 CPU/CUDA rows, 1..20000, matching task columns"
        )
    return evaluate_mechanism(json.loads(task.mechanism_json), inputs)


def evaluate_mechanism(card: dict[str, Any], inputs: torch.Tensor) -> torch.Tensor:
    def tensor(key: str) -> torch.Tensor:
        return torch.tensor(card[key], dtype=torch.float64, device=inputs.device)

    version = card["version"]
    if version == "composite-v1":
        scores = composite_scores(card, inputs)
    elif version in ("linear-v1", "grouped-linear-v1"):
        weights = tensor("weights")
        scores = inputs[:, : weights.shape[0]] @ weights
    elif version == "polynomial-v1":
        scores = polynomial_basis(card, inputs) @ tensor("weights")
    elif version == "quadratic-v1":
        directions, weights = tensor("directions"), tensor("weights")
        signal = inputs[:, : directions.shape[1]]
        # Preserve the recorded generator's CPU diagonal calculation across devices.
        diagonal = (np.asarray(card["directions"]) ** 2).T @ np.asarray(
            card["weights"], dtype=np.float64
        )
        strength = card["interaction_strength"]
        scores = (
            strength * ((signal @ directions.T) ** 2 @ weights)
            + (1 - strength) * (signal**2 @ torch.tensor(diagonal, device=inputs.device))
            + card["linear_strength"] * (signal @ tensor("linear_weights"))
        )
    elif version == "fourier-v1":
        frequencies, phases, coefficients = (
            tensor("frequencies"),
            tensor("phases"),
            tensor("coefficients"),
        )
        scores = torch.cat(
            [
                fourier_basis(
                    inputs[start : start + 256, : frequencies.shape[1]] @ frequencies.T,
                    phases,
                    centered=card.get("centered_basis", False),
                )
                @ coefficients
                for start in range(0, len(inputs), 256)
            ]
        )
    elif version == "smooth-v1":
        directions = np.asarray(card["directions"], dtype=np.float64)
        scores = smooth_scores(
            inputs[:, : directions.shape[1]],
            directions,
            np.asarray(card["centers"], dtype=np.float64),
            np.asarray(card["widths"], dtype=np.float64),
            np.asarray(card["weights"], dtype=np.float64),
            "sinusoidal" if card["family"].endswith("sinusoidal") else "radial",
        )
    elif version == "exact-gp-v1":
        raise ValueError(
            "direct GP inspection requires explicit conditional mean or additional sampling"
        )
    else:
        raise ValueError(f"unsupported signal version: {version!r}")
    if version != "linear-v1":
        scores = (scores - tensor("center")) / tensor("scale")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite evaluated signal; function was not changed or retried")
    return scores


def polynomial_basis(card: dict[str, Any], inputs: torch.Tensor) -> torch.Tensor:
    """Keep the original matrix shape, but never exponentiate inactive columns."""
    powers = torch.tensor(card["powers"], dtype=torch.long, device=inputs.device)
    active = torch.zeros(len(powers), dtype=torch.bool, device=inputs.device)
    active[card["active"]] = True
    signal = torch.where(active, inputs[:, : len(powers)], 0.0)
    return signal**powers


def fourier_basis(
    projection: torch.Tensor, phases: torch.Tensor, *, centered: bool = False
) -> torch.Tensor:
    if centered:
        # cos(phase + projection) - cos(phase), without subtracting nearby cosines.
        return -2 * torch.sin(projection / 2) * torch.sin(phases + projection / 2)
    return torch.cos(projection + phases)


def smooth_scores(
    x: torch.Tensor,
    directions: NDArray[np.float64],
    centers: NDArray[np.float64],
    widths: NDArray[np.float64],
    weights: NDArray[np.float64],
    family: str,
) -> torch.Tensor:
    direction_tensor = torch.tensor(directions, device=x.device)
    center_tensor = torch.tensor(centers, device=x.device)
    width_tensor = torch.tensor(widths, device=x.device)
    weight_tensor = torch.tensor(weights, device=x.device)
    parts: list[torch.Tensor] = []
    for start in range(0, x.shape[0], 256):
        part = x[start : start + 256]
        z = (part @ direction_tensor.T - center_tensor) / width_tensor
        if not torch.isfinite(z).all():
            raise ValueError("nonfinite smooth arguments; task was not retried")
        values = torch.sin(z) if family == "sinusoidal" else torch.exp(-0.5 * z.square())
        parts.append(values @ weight_tensor)
    return torch.cat(parts)


def composite_scores(card: dict[str, Any], inputs: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, len(inputs), 256):
        part = inputs[start : start + 256]
        leaves: list[torch.Tensor] = []
        for component in card["components"]:
            child = component_inputs(component, part)
            leaves.append(
                multiply_signals(
                    evaluate_mechanism(component["mechanism"], child), component["strength"]
                )
            )

        result = _evaluate_expression(card["expression"], leaves)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite composition; task was not retried")
        parts.append(result)
    return torch.cat(parts)


def _evaluate_expression(expression: Any, leaves: list[torch.Tensor]) -> torch.Tensor:
    if type(expression) is int:
        return leaves[expression]
    values = [_evaluate_expression(child, leaves) for child in expression["children"]]
    return combine_signals(values, expression["operation"], expression["strength"])


def component_inputs(component: dict[str, Any], inputs: torch.Tensor) -> torch.Tensor:
    """Map parent columns onto a child's active columns, zeroing unused inputs."""
    child = torch.zeros(
        (len(inputs), component["feature_count"]), dtype=inputs.dtype, device=inputs.device
    )
    child[:, component["mechanism"]["active"]] = inputs[:, component["input_columns"]]
    return child


def combine_signals(values: list[torch.Tensor], operation: str, strength: float) -> torch.Tensor:
    """Reduce validated child values in recorded order with the shared product guard."""
    result = values[0]
    for value in values[1:]:
        result = result + value if operation == "sum" else multiply_signals(result, value)
    return multiply_signals(result, strength)


def multiply_signals(left: torch.Tensor, right: torch.Tensor | float) -> torch.Tensor:
    result = left * right
    if torch.any((result == 0) & (left != 0) & (right != 0)):
        raise ValueError("composition product underflow; task was not retried")
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite composition product; task was not retried")
    return result
