"""Frozen numerical converter semantics, separate from descendant state storage."""

from dataclasses import dataclass
from typing import Any

import torch

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.calibrated_activations import (
    CalibratedActivation,
    CalibratedStandardization,
    activation_from_dict,
)
from r0.modules.priors.lib.calibrated_transforms import CalibratedTransform
from r0.modules.priors.lib.causal_activations import CausalActivation, activation_diagnostic


@dataclass(frozen=True, slots=True)
class NumericalConverter:
    channel: int
    transform: CausalActivation = CausalActivation()
    variant: str = "readout-only"

    def __post_init__(self) -> None:
        if type(self.channel) is not int or self.channel < 0:
            raise ValueError("numerical converter channel must be nonnegative")
        if self.variant not in ("historical-tuple", "historical-graph", "readout-only"):
            raise ValueError("unsupported numerical converter variant")
        if (
            not (type(self.transform) is CausalActivation and self.transform == CausalActivation())
            and not (
                type(self.transform) is CalibratedTransform
                and self.transform.kind == "kumaraswamy"
                and self.transform.channels == 1
            )
            and not (
                type(self.transform) is CalibratedActivation and len(self.transform.input_mean) == 1
            )
        ):
            raise ValueError(
                "numerical converter requires identity or a one-channel "
                "frozen warp/standard activation"
            )

    @property
    def changes_state(self) -> bool:
        return self.variant != "readout-only"

    @property
    def captures_raw(self) -> bool:
        return self.variant == "historical-tuple"

    @property
    def parameter_count(self) -> int:
        return 1 + self.transform.parameter_count


@dataclass(frozen=True, slots=True)
class CausalNodePipeline:
    channels: int
    stages: tuple[CausalActivation, ...] = ()
    channel_importance: tuple[float, ...] = ()
    l2: CalibratedTransform | None = None
    converters: tuple[NumericalConverter, ...] = ()
    node_importance: float = 1.0
    version: str = "causal-node-pipeline-v1"

    def __post_init__(self) -> None:
        if self.version != "causal-node-pipeline-v1":
            raise ValueError("unsupported causal node pipeline version")
        if type(self.channels) is not int or not 1 <= self.channels < 2**63:
            raise ValueError("pipeline channels must be positive")
        if type(self.stages) is not tuple or any(
            type(a)
            not in (
                CausalActivation,
                CalibratedActivation,
                CalibratedTransform,
                CalibratedStandardization,
            )
            for a in self.stages
        ):
            raise ValueError("pipeline stages require frozen activation records")
        for stage in self.stages:
            stage.validate_width(self.channels)
        if type(self.channel_importance) is not tuple or (
            self.channel_importance
            and (
                len(self.channel_importance) != self.channels
                or any(not finite_number(v) for v in self.channel_importance)
            )
        ):
            raise ValueError("channel importance requires one finite factor per channel")
        if self.l2 is not None:
            if type(self.l2) is not CalibratedTransform or self.l2.kind != "l2_normalize":
                raise ValueError("pipeline L2 requires a frozen normalization transform")
            self.l2.validate_width(self.channels)
        if (
            type(self.converters) is not tuple
            or any(
                type(c) is not NumericalConverter or c.channel >= self.channels
                for c in self.converters
            )
            or len({c.channel for c in self.converters}) != len(self.converters)
        ):
            raise ValueError("pipeline converters must use unique valid channels")
        if not finite_number(self.node_importance) or self.node_importance <= 0:
            raise ValueError("node importance must be finite and positive")

    @property
    def parameter_count(self) -> int:
        return (
            2
            + sum(c.parameter_count for c in self.converters)
            + sum(a.parameter_count for a in self.stages)
            + len(self.channel_importance)
            + (self.l2.parameter_count if self.l2 is not None else 0)
        )

    @property
    def working_extra_channels(self) -> int:
        return (8 + len(self.stages)) * self.channels + max(
            [c.transform.working_extra_channels for c in self.converters]
            + [a.working_extra_channels for a in self.stages]
            + [self.l2.working_extra_channels if self.l2 is not None else 0]
        )


def pipeline_from_dict(data: dict[str, Any]) -> CausalNodePipeline:
    fields = dict(data)
    fields["stages"] = tuple(activation_from_dict(a) for a in fields.get("stages", ()))
    fields["channel_importance"] = tuple(fields.get("channel_importance", ()))
    if fields.get("l2") is not None:
        fields["l2"] = activation_from_dict(fields["l2"])
    converters: list[NumericalConverter] = []
    for raw in fields.get("converters", ()):
        item: dict[str, Any] = dict(raw)
        item["transform"] = activation_from_dict(item.get("transform", {}))
        converters.append(NumericalConverter(**item))
    fields["converters"] = tuple(converters)
    return CausalNodePipeline(**fields)


class PipelineEvaluation:
    """One node/row batch; captured readouts never alias propagated state."""

    def __init__(
        self,
        pipeline: CausalNodePipeline,
        node: int,
        diagnostics: list[dict[str, Any]] | None,
        *,
        record_steps: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.node = node
        self.diagnostics = diagnostics
        self.captured: dict[int, torch.Tensor] = {}
        self.record_steps = record_steps
        self.steps: list[tuple[str, torch.Tensor]] = []

    def retain(self, name: str, value: torch.Tensor) -> None:
        if self.record_steps:
            self.steps.append((name, value.clone()))

    def before_noise(self, value: torch.Tensor) -> torch.Tensor:
        self.retain("input", value)
        for index, stage in enumerate(self.pipeline.stages):
            if self.diagnostics is not None:
                entry = activation_diagnostic(value, stage)
                if entry is not None:
                    self.diagnostics.append(
                        {"node": self.node, "stage": f"pipeline-{index}", **entry}
                    )
            value = stage.apply(value)
            self.retain(f"stage-{index}", value)
        if self.pipeline.channel_importance:
            value = value * value.new_tensor(self.pipeline.channel_importance)
            self.retain("channel-importance", value)
        if self.pipeline.l2 is not None:
            value = self.pipeline.l2.apply(value)
            self.retain("l2", value)
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite node pipeline stage; task was not retried")
        state = value.clone()
        for converter in self.pipeline.converters:
            channel = converter.channel
            raw = value[:, channel : channel + 1]
            if self.diagnostics is not None:
                entry = activation_diagnostic(raw, converter.transform)
                if entry is not None:
                    self.diagnostics.append(
                        {"node": self.node, "stage": "converter", "channel": channel, **entry}
                    )
            warped = converter.transform.apply(raw)
            if converter.changes_state:
                state[:, channel : channel + 1] = warped
            self.captured[channel] = (raw if converter.captures_raw else warped).clone()
        self.retain("converter-state", state)
        if self.record_steps:
            readout = value.clone()
            for channel, captured in self.captured.items():
                readout[:, channel : channel + 1] = captured
            self.retain("converted-readout", readout)
        return state

    def finish(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = value * self.pipeline.node_importance
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite node importance output; task was not retried")
        readout = state.clone()
        for channel, captured in self.captured.items():
            readout[:, channel : channel + 1] = captured
        return state, readout
