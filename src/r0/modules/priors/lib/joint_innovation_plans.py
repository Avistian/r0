"""Explicit r0 innovation hyperpriors; source Gaussian pipeline semantics stay separate."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_preferences import CategoricalParameter
from r0.modules.priors.lib.exogenous import NoiseResponse
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.joint_plan_bindings import (
    categorical_binding,
    numerical_binding,
    operation_events,
)
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.parameter_preferences import NumericalParameter


@dataclass(frozen=True, slots=True)
class JointInnovationPrior:
    families: tuple[str, ...] = ("gaussian", "uniform", "student_t", "mixture", "ball")
    factors: tuple[str, ...] = ("identity", "gaussian")
    truncation: tuple[float | None, ...] = (None, 2.0, 4.0, 8.0)
    degrees: tuple[float, ...] = (0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
    strengths: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)
    powers: tuple[float, ...] = (0.5, 1.0, 2.0)
    location: tuple[float, float] = (-0.5, 0.5)
    input_scale: tuple[float, float] = (0.25, 2.0)
    mixture_gap: tuple[float, float] = (0.25, 4.0)
    structural_scale: tuple[float, float] = (1e-5, 1.0)
    offset: tuple[float, float] = (1.0, 1.0)
    scale_scope: str = "node"
    version: str = "r0-joint-innovation-prior-v1"

    def __post_init__(self) -> None:
        if self.version != "r0-joint-innovation-prior-v1" or self.scale_scope not in (
            "node",
            "channel",
        ):
            raise ValueError("innovation prior requires its explicit version and scale scope")
        for values, allowed in (
            (self.families, ("gaussian", "uniform", "student_t", "mixture", "ball")),
            (self.factors, ("identity", "gaussian")),
        ):
            if (
                type(values) is not tuple
                or not 1 <= len(values) <= 256
                or any(v not in allowed for v in values)
            ):
                raise ValueError("innovation family/factor catalog is unsupported")
        for name, values, minimum in (
            ("truncation", self.truncation, 0.1),
            ("degrees", self.degrees, 0.0),
            ("strengths", self.strengths, 0.0),
            ("powers", self.powers, 0.0),
        ):
            if (
                type(values) is not tuple
                or not 1 <= len(values) <= 256
                or any(
                    not (name == "truncation" and v is None)
                    and (
                        v is None
                        or not finite_number(v)
                        or v < minimum
                        or name in ("degrees", "powers")
                        and v == 0
                    )
                    for v in values
                )
            ):
                raise ValueError("innovation parameter catalog is invalid")
        for name, bounds in (
            ("location", self.location),
            ("input_scale", self.input_scale),
            ("mixture_gap", self.mixture_gap),
            ("structural_scale", self.structural_scale),
            ("offset", self.offset),
        ):
            if type(bounds) is not tuple or len(bounds) != 2:
                raise ValueError("innovation numeric laws require a frozen pair of bounds")
            innovation_number(name, bounds)
            if name == "offset" and bounds[0] < 0:
                raise ValueError("innovation response offset must be nonnegative")


def innovation_number(name: str, bounds: tuple[float, float], width: int = 0) -> NumericalParameter:
    return NumericalParameter(
        "r0_innovation_" + name.split(":", 1)[0],
        *bounds,
        shape=(width,) if width else (),
        scale="linear" if name in ("location", "offset") else "log",
    )


def innovation_category(role: str, values: tuple[Any, ...]) -> CategoricalParameter:
    return CategoricalParameter(
        "r0_innovation_" + role,
        tuple(f"option-{i}:{json.dumps(v, allow_nan=False)}" for i, v in enumerate(values)),
    )


@dataclass(frozen=True, slots=True)
class JointInnovationPlan:
    operation: tuple[str, ...]
    width: int
    prior: JointInnovationPrior
    family: str
    factor: str
    inputs: InputDistribution
    response: NoiseResponse
    scales: tuple[float, ...]
    version: str = "r0-joint-innovation-plan-v1"

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if (
            self.version != "r0-joint-innovation-plan-v1"
            or type(self.prior) is not JointInnovationPrior
            or type(self.width) is not int
            or not 1 <= self.width < 2**63
            or self.family not in self.prior.families
            or self.factor not in self.prior.factors
            or type(self.inputs) is not InputDistribution
            or type(self.response) is not NoiseResponse
            or type(self.scales) is not tuple
            or len(self.scales) != self.width
            or any(not finite_number(v) or v < 0 for v in self.scales)
        ):
            raise ValueError("innovation plan requires complete frozen geometry and laws")

    def validate(self, record: JointSamplingRecord) -> None:
        events = iter(operation_events(record, self.operation))

        def choose(role: str, values: tuple[Any, ...]) -> Any:
            event = next(events, None)
            if event is None or event.operation != self.operation or event.role != role:
                raise ValueError("innovation categorical decision ownership differs")
            draw = categorical_binding(event, record, innovation_category(role, values))
            return values[draw.index]

        def number(role: str, bounds: tuple[float, float], width: int = 0) -> tuple[float, ...]:
            event = next(events, None)
            if event is None or event.operation != self.operation or event.role != role:
                raise ValueError("innovation numerical decision ownership differs")
            return numerical_binding(event, record, innovation_number(role, bounds, width)).values

        expected = _plan(self.operation, self.width, self.prior, choose, number)
        if expected != self or next(events, None) is not None:
            raise ValueError("innovation law differs from retained decisions")


def _plan(
    operation: tuple[str, ...],
    width: int,
    prior: JointInnovationPrior,
    choose: Callable[[str, tuple[Any, ...]], Any],
    number: Callable[..., tuple[float, ...]],
) -> JointInnovationPlan:
    family = choose("family", prior.families)
    factor = choose("factor", prior.factors)
    # Ball is a complete direction/radius law, not a discarded selected base law.
    inputs = InputDistribution()
    if family != "ball":
        bound = choose("truncation", prior.truncation)
        location = number("location", prior.location)[0]
        scale = number("input_scale", prior.input_scale)[0]
        df = choose("degrees", prior.degrees) if family == "student_t" else 5.0
        gap = number("mixture_gap", prior.mixture_gap)[0] if family == "mixture" else 2.0
        inputs = InputDistribution(family, location, scale, df, gap, bound)
    strength = choose("strengths", prior.strengths)
    power = choose("powers", prior.powers)
    offset = number("offset", prior.offset)[0]
    scales = (
        tuple(number(f"structural_scale:{i}", prior.structural_scale)[0] for i in range(width))
        if prior.scale_scope == "channel"
        else number("structural_scale", prior.structural_scale) * width
    )
    return JointInnovationPlan(
        operation,
        width,
        prior,
        family,
        factor,
        inputs,
        NoiseResponse(offset, strength, power),
        scales,
    )


def sample_joint_innovation(
    context: JointSamplingContext, *, width: int, prior: JointInnovationPrior
) -> JointInnovationPlan:
    with context.transaction("innovation") as op:

        def choose(role: str, values: tuple[Any, ...]) -> Any:
            return values[op.categorical(role, innovation_category(role, values)).index]

        def number(role: str, bounds: tuple[float, float], width: int = 0) -> tuple[float, ...]:
            return op.numerical(role, innovation_number(role, bounds, width)).values

        result = _plan(op.operation, width, prior, choose, number)
        result.validate(op.freeze())
        return result
