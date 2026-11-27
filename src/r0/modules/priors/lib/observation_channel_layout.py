"""Allocate converter input spans before appending each node's hidden channels."""

from dataclasses import dataclass

from r0.modules.priors.lib.joint_observation_realization import JointObservationRealization
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord


def _layout(
    observations: JointObservationRealization,
    converter_widths: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[tuple[int, int, int], ...], ...], tuple[int, ...]]:
    plan = observations.plan
    limits = plan.limits
    if (
        type(converter_widths) is not tuple
        or len(converter_widths) != len(plan.groups)
        or any(
            type(row) is not tuple
            or len(row) != group.count
            or any(type(width) is not int or width < 1 for width in row)
            for row, group in zip(converter_widths, plan.groups, strict=True)
        )
    ):
        raise ValueError("converter widths must match every ordered feature group")
    total = sum(sum(row) for row in converter_widths)
    if total + plan.nodes > limits.total_channels:
        raise ValueError("converter reservations exceed total channel limit")
    if any(width >= limits.node_channels for row in converter_widths for width in row):
        raise ValueError("converter reservations exceed node channel limit")
    count = sum(group.count for group in plan.groups)
    if (
        8 * count + plan.nodes > limits.parameters
        or 1024 * count + 64 * plan.nodes > limits.working_bytes
    ):
        raise ValueError("converter channel layout metadata limit exceeded")
    cursors = [0] * plan.nodes
    result: list[tuple[tuple[int, int, int], ...]] = []
    for assignments, widths in zip(observations.assignments, converter_widths, strict=True):
        spans: list[tuple[int, int, int]] = []
        for node, width in zip(assignments, widths, strict=True):
            start, stop = cursors[node], cursors[node] + width
            # Every node still needs at least one latent channel. The subsequent
            # width sampler accounts for its full realized latent width.
            if stop >= limits.node_channels:
                raise ValueError("converter reservations exceed node channel limit")
            spans.append((node, start, stop))
            cursors[node] = stop
        result.append(tuple(spans))
    return tuple(result), tuple(cursors)


@dataclass(frozen=True, slots=True)
class ObservationChannelLayout:
    observations: JointObservationRealization
    converter_widths: tuple[tuple[int, ...], ...]
    spans: tuple[tuple[tuple[int, int, int], ...], ...]
    converter_channels: tuple[int, ...]
    version: str = "observation-channel-layout-v1"

    def validate(self, decisions: JointSamplingRecord) -> None:
        if (
            self.version != "observation-channel-layout-v1"
            or type(self.observations) is not JointObservationRealization
        ):
            raise ValueError("channel layout requires frozen observations and its version")
        self.observations.validate(decisions)
        if (
            type(self.spans) is not tuple
            or any(
                type(row) is not tuple
                or any(
                    type(span) is not tuple
                    or len(span) != 3
                    or any(type(n) is not int for n in span)
                    for span in row
                )
                for row in self.spans
            )
            or type(self.converter_channels) is not tuple
            or any(type(n) is not int for n in self.converter_channels)
            or (self.spans, self.converter_channels)
            != _layout(self.observations, self.converter_widths)
        ):
            raise ValueError("channel layout differs from ordered converter reservations")


def allocate_observation_channels(
    observations: JointObservationRealization,
    decisions: JointSamplingRecord,
    *,
    converter_widths: tuple[tuple[int, ...], ...],
) -> ObservationChannelLayout:
    """Allocate half-open input spans, not categorical outputs or latent widths."""
    if type(observations) is not JointObservationRealization:
        raise ValueError("channel layout requires frozen observations")
    observations.validate(decisions)
    spans, channels = _layout(observations, converter_widths)
    result = ObservationChannelLayout(observations, converter_widths, spans, channels)
    result.validate(decisions)
    return result
