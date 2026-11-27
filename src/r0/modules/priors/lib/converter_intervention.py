"""Explicit categorical capture cuts and frozen converter mechanism replacements."""

from dataclasses import dataclass

import torch

from r0.modules.priors.lib.categorical_converter import CategoricalConverter, CategoricalEmission
from r0.modules.priors.lib.converted_causal_records import ConvertedCausalNode
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function


@dataclass(frozen=True, slots=True)
class ConverterHardIntervention:
    node: int
    capture: str
    category: str
    location: str = "capture"
    version: str = "converter-hard-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converter-hard-intervention-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.capture) is not str
            or not self.capture
            or type(self.category) is not str
            or not self.category
            or self.location not in ("capture", "representation_and_capture")
        ):
            raise ValueError("converter clamp requires node, capture, category and exact location")


@dataclass(frozen=True, slots=True)
class ConverterSoftIntervention:
    node: int
    capture: str
    converter: CategoricalConverter
    version: str = "converter-soft-intervention-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converter-soft-intervention-v1"
            or type(self.node) is not int
            or self.node < 0
            or type(self.capture) is not str
            or not self.capture
            or type(self.converter) is not CategoricalConverter
        ):
            raise ValueError("converter replacement requires a frozen converter and capture")


def validate_converter_change(
    change: ConverterHardIntervention | ConverterSoftIntervention, node: ConvertedCausalNode
) -> None:
    converter = next((c for c in node.converters if c.name == change.capture), None)
    if converter is None:
        raise ValueError("converter intervention must name an existing capture")
    if isinstance(change, ConverterHardIntervention):
        if change.category not in converter.vocabulary:
            raise ValueError("converter category must belong to its vocabulary")
        if change.location == "representation_and_capture" and converter.mode.endswith("_id"):
            raise ValueError(
                "original-input converter has no unique category representation to force"
            )
    elif (
        change.converter.name,
        change.converter.start,
        change.converter.stop,
        change.converter.vocabulary,
    ) != (converter.name, converter.start, converter.stop, converter.vocabulary):
        raise ValueError("converter replacement must preserve name, span and vocabulary")


def forced_emission(
    converter: CategoricalConverter,
    change: ConverterHardIntervention,
    values: torch.Tensor,
    original: CategoricalEmission | None = None,
) -> CategoricalEmission:
    category = converter.vocabulary.index(change.category)
    categories = torch.full((len(values),), category, dtype=torch.int64, device=values.device)
    indicator = torch.nn.functional.one_hot(categories, len(converter.vocabulary)).to(values.dtype)
    if change.location == "capture":
        state = original.state if original is not None else values.clone()
    elif converter.mode.endswith("_int"):
        state = categories[:, None].expand(-1, converter.width).to(values.dtype).clone()
    else:
        state = values.new_tensor(converter.representatives)[categories]
        if converter.post_function is not None:
            state = evaluate_admitted_function(converter.post_function, state)
    if not torch.isfinite(state).all():
        raise ValueError("nonfinite forced converter representation")
    return CategoricalEmission(categories, state, indicator, indicator.clone())
