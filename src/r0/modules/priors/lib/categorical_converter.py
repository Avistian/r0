"""Frozen category captures with explicit width-preserving descendant representations."""

import math
from dataclasses import dataclass

import torch

from r0.modules.priors.lib.frozen_function_evaluation import (
    FrozenFunction,
    evaluate_admitted_function,
    function_geometry,
)
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

CONVERTER_MODES = (
    "neighbor_id",
    "neighbor_disc",
    "neighbor_func",
    "neighbor_int",
    "softmax_id",
    "softmax_disc",
    "softmax_int",
)


@dataclass(frozen=True, slots=True)
class CategoricalEmission:
    categories: torch.Tensor
    state: torch.Tensor
    probabilities: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class CategoricalConverter:
    name: str
    start: int
    stop: int
    vocabulary: tuple[str, ...]
    mode: str
    representatives: Matrix
    exponent: float = 2.0
    mean: tuple[float, ...] = ()
    scale: tuple[float, ...] = ()
    bias: tuple[float, ...] = ()
    factor: float = 1.0
    post_function: FrozenFunction | None = None
    version: str = "categorical-converter-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "categorical-converter-v1"
            or type(self.name) is not str
            or not self.name
            or type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop < 2**63
            or self.mode not in CONVERTER_MODES
            or type(self.vocabulary) is not tuple
            or len(self.vocabulary) < 2
            or any(type(v) is not str or not v for v in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)
        ):
            raise ValueError(
                "categorical converter requires its version, name, span, mode and vocabulary"
            )
        if matrix_shape(self.representatives) != (len(self.vocabulary), self.width):
            raise ValueError("converter representatives must match category count and span width")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.exponent, self.factor)
        ):
            raise ValueError(
                "converter distance exponent and softmax factor must be finite positive"
            )
        for values in (self.mean, self.scale, self.bias):
            if type(values) is not tuple or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in values
            ):
                raise ValueError("converter standardization and bias require finite frozen tuples")
        if self.stochastic:
            if (
                self.width != len(self.vocabulary)
                or any(len(v) != self.width for v in (self.mean, self.scale, self.bias))
                or any(v <= 0 for v in self.scale)
                or self.exponent != 2.0
            ):
                raise ValueError("softmax converter needs one standardized logit per category")
        elif self.mean or self.scale or self.bias or self.factor != 1.0:
            raise ValueError("nearest converter cannot retain ignored softmax parameters")
        if self.mode == "neighbor_func":
            if self.post_function is None:
                raise ValueError("transformed-center converter requires a frozen post function")
            function_geometry(self.post_function)
            if (
                self.post_function.input_width != self.width
                or self.post_function.output_width != self.width
            ):
                raise ValueError("converter post function must preserve its input span width")
        elif self.post_function is not None:
            raise ValueError("post function applies only to transformed-center representation")

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def stochastic(self) -> bool:
        return self.mode.startswith("softmax_")

    def apply(
        self, values: torch.Tensor, *, uniforms: torch.Tensor | None = None
    ) -> CategoricalEmission:
        if values.ndim != 2 or values.shape[1] != self.width or not torch.isfinite(values).all():
            raise ValueError("converter input must be a finite matrix matching its span")
        representatives = values.new_tensor(self.representatives)
        if self.stochastic:
            if (
                uniforms is None
                or uniforms.shape != (len(values),)
                or uniforms.device != values.device
                or not torch.isfinite(uniforms).all()
                or not torch.all((uniforms >= 0) & (uniforms < 1))
            ):
                raise ValueError("softmax converter requires one injected uniform in [0,1) per row")
            scores = (
                self.factor
                * (values - values.new_tensor(self.mean))
                / values.new_tensor(self.scale)
            )
            scores = scores + values.new_tensor(self.bias)
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite converter logits; no replacement or retry")
            probabilities = torch.softmax(scores, dim=1)
            # The last bucket absorbs floating-point CDF roundoff only.
            categories = (
                (uniforms[:, None] >= probabilities.cumsum(dim=1))
                .sum(dim=1)
                .clamp_max(len(self.vocabulary) - 1)
            )
        else:
            if uniforms is not None:
                raise ValueError("nearest converter does not consume emission uniforms")
            distances = (
                ((values[:, None, :] - representatives[None, :, :]).abs() ** self.exponent)
                .sum(dim=2)
                .pow(1.0 / self.exponent)
            )
            if not torch.isfinite(distances).all():
                raise ValueError("nonfinite converter distances; no replacement or retry")
            categories = distances.argmin(dim=1)
            scores = -distances
            probabilities = torch.nn.functional.one_hot(categories, len(self.vocabulary)).to(
                values.dtype
            )
        state = values.clone() if self.mode.endswith("_id") else representatives[categories]
        if self.mode.endswith("_int"):
            state = categories[:, None].expand(-1, self.width).to(values.dtype).clone()
        if self.post_function is not None:
            state = evaluate_admitted_function(self.post_function, state)
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite converter representation; no replacement or retry")
        return CategoricalEmission(categories, state, probabilities, scores)
