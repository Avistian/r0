"""Context-only fitting and replay receipts for numeric observation maps."""

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.quantile_map import Kind, fit_quantile_map
from r0.modules.priors.lib.training_features import TrainingFeature


@dataclass(frozen=True, slots=True)
class QuantileColumnSpec:
    column: int
    kind: Kind = "quantile-normal"
    tail_probability: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.column) is not int
            or not 0 <= self.column < 256
            or self.kind not in ("rank", "quantile-uniform", "quantile-normal")
        ):
            raise ValueError("invalid quantile column specification")
        tail = self.tail_probability
        if tail is not None and (
            type(tail) not in (float, int)
            or not math.isfinite(tail)
            or (self.kind == "quantile-normal" and not np.finfo(np.float64).eps <= tail < 0.5)
            or (self.kind != "quantile-normal" and tail != 0)
        ):
            raise ValueError("invalid quantile tail probability")


def quantile_specs_from_dict(values: Any) -> tuple[QuantileColumnSpec, ...]:
    if type(values) not in (list, tuple) or not 1 <= len(values) <= 256:
        raise ValueError("quantile specifications require a bounded nonempty array")
    specs: list[QuantileColumnSpec] = []
    for value in cast(list[Any], values):
        if type(value) is not dict:
            raise ValueError("quantile specification must be an object")
        data = cast(dict[str, Any], value)
        if set(data) != {"column", "kind", "tail_probability"}:
            raise ValueError("quantile specification fields differ")
        specs.append(QuantileColumnSpec(**data))
    return tuple(specs)


def quantile_observation(
    specs: tuple[QuantileColumnSpec, ...],
    inputs: torch.Tensor,
    context_indices: torch.Tensor,
    schema: tuple[TrainingFeature, ...],
    limits: CausalLimits,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if (
        type(specs) is not tuple
        or not 1 <= len(specs) <= 256
        or any(type(s) is not QuantileColumnSpec for s in specs)
        or len({s.column for s in specs}) != len(specs)
        or inputs.device.type != "cpu"
        or inputs.dtype != torch.float64
        or inputs.ndim != 2
        or not 1 <= inputs.shape[0] <= 20000
        or not 1 <= inputs.shape[1] <= 256
        or not bool(torch.isfinite(inputs).all())
    ):
        raise ValueError("quantile observations require bounded finite inputs and unique columns")
    if (
        context_indices.device.type != "cpu"
        or context_indices.dtype != torch.int64
        or context_indices.ndim != 1
        or not 1 <= len(context_indices) <= len(inputs)
        or len(set(int(i) for i in context_indices)) != len(context_indices)
        or bool((context_indices < 0).any())
        or bool((context_indices >= len(inputs)).any())
    ):
        raise ValueError("quantile fitting requires unique valid context row indices")
    if schema and len(schema) != inputs.shape[1]:
        raise ValueError("quantile source schema width differs")
    if any(
        s.column >= inputs.shape[1] or (schema and schema[s.column].kind != "continuous")
        for s in specs
    ):
        raise ValueError("quantile maps require continuous source columns")
    fitted_values = len(specs) * len(context_indices)
    if (
        2 * fitted_values > limits.parameters
        or inputs.numel() * 32 + fitted_values * 24 > limits.working_bytes
    ):
        raise ValueError("quantile calibration exceeds declared resource limits")
    values = np.asarray(inputs.detach().numpy(), dtype=np.float64).copy()
    indices = context_indices.numpy()
    maps: list[dict[str, Any]] = []
    for spec in specs:
        fitted = fit_quantile_map(
            values[indices, spec.column], kind=spec.kind, tail_probability=spec.tail_probability
        )
        values[:, spec.column] = fitted.apply(values[:, spec.column])
        maps.append(asdict(fitted))
    return torch.tensor(values), {
        "scope": "context-only",
        "calibration_rows": [int(i) for i in context_indices],
        "specs": [asdict(s) for s in specs],
        "maps": maps,
    }
