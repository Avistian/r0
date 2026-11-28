"""Resolve typed source requests against already validated joint decisions."""

from r0.modules.priors.lib.categorical_preferences import CategoricalDraw, CategoricalParameter
from r0.modules.priors.lib.joint_sampling_records import JointDecision, JointSamplingRecord
from r0.modules.priors.lib.parameter_preferences import NumericalDraw, NumericalParameter


def categorical_binding(
    event: JointDecision, record: JointSamplingRecord, request: CategoricalParameter
) -> CategoricalDraw:
    if event.kind != "categorical" or event.draw_index is None:
        raise ValueError("plan binding requires a categorical decision")
    draw = record.categorical.draws[event.draw_index]
    if draw.request != request:
        raise ValueError("plan categorical request differs from its source contract")
    return draw


def numerical_binding(
    event: JointDecision, record: JointSamplingRecord, request: NumericalParameter
) -> NumericalDraw:
    if event.kind != "numerical" or event.draw_index is None:
        raise ValueError("plan binding requires a numerical decision")
    draw = record.numerical.draws[event.draw_index]
    if draw.request != request:
        raise ValueError("plan numerical request differs from its source contract")
    return draw


def operation_events(
    decisions: JointSamplingRecord, operation: tuple[str, ...]
) -> tuple[JointDecision, ...]:
    if type(decisions) is not JointSamplingRecord:
        raise ValueError("plan binding requires frozen joint decisions")
    return tuple(e for e in decisions.events if e.operation[: len(operation)] == operation)
