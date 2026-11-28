"""Ordered ownership of decisions drawn from one shared preference scope."""

from dataclasses import dataclass
from typing import Any

from r0.modules.priors.lib.categorical_preferences import (
    CategoricalPreferences,
    categorical_preferences_from_dict,
    raw_categorical_admission,
)
from r0.modules.priors.lib.parameter_preferences import (
    NumericalParameter,
    ParameterPreferences,
    parameter_preferences_from_dict,
    raw_numerical_admission,
)
from r0.modules.priors.lib.preference_limits import PreferenceLimits


def numerical_policy(policy: str) -> str:
    if policy not in ("scalar-quantile-v1", "strict-contract-v1"):
        raise ValueError("unsupported joint global policy")
    return "quantile" if policy == "scalar-quantile-v1" else "strict"


def validate_name(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\0" in value
        or len(value.encode("utf-8")) > 128
    ):
        raise ValueError(f"joint {label} must be nonempty, trimmed and at most 128 UTF-8 bytes")


def validate_path(path: tuple[str, ...]) -> None:
    if type(path) is not tuple or not 1 <= len(path) <= 16:
        raise ValueError("joint operation path depth must be 1..16")
    for name in path:
        validate_name(name, "operation name")


def validate_event_paths(events: tuple["JointDecision", ...]) -> None:
    previous: tuple[str, ...] = ()
    closed: set[tuple[str, ...]] = set()
    for event in events:
        path = event.operation
        common = 0
        for left, right in zip(previous, path, strict=False):
            if left != right:
                break
            common += 1
        closed.update(previous[:n] for n in range(common + 1, len(previous) + 1))
        if any(path[:n] in closed for n in range(1, len(path) + 1)):
            raise ValueError("joint events cannot reopen an earlier operation")
        previous = path


def check_numerical_contract(
    request: NumericalParameter, mode: str, policy: str, previous: NumericalParameter | None
) -> None:
    if mode == "global" and policy == "scalar-quantile-v1" and request.shape != ():
        raise ValueError("joint quantile global policy requires scalar requests")
    if previous is not None and (request.scale, request.integer, request.boundary_mass) != (
        previous.scale,
        previous.integer,
        previous.boundary_mass,
    ):
        raise ValueError("joint numerical law contract changed for a cached name")


@dataclass(frozen=True, slots=True)
class JointDecision:
    operation: tuple[str, ...]
    role: str
    kind: str
    draw_index: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        validate_path(self.operation)
        validate_name(self.role, "role")
        if self.kind not in ("numerical", "categorical", "forced_identity"):
            raise ValueError("joint decisions require an operation, role and typed draw reference")
        if self.kind == "forced_identity":
            if self.draw_index is not None or type(self.reason) is not str or not self.reason:
                raise ValueError("forced identity requires a reason and no draw reference")
            validate_name(self.reason, "reason")
        elif type(self.draw_index) is not int or self.draw_index < 0 or self.reason is not None:
            raise ValueError("sampled joint decision requires a draw index and no forced reason")


@dataclass(frozen=True, slots=True)
class JointSamplingRecord:
    numerical: ParameterPreferences
    categorical: CategoricalPreferences
    events: tuple[JointDecision, ...]
    global_policy: str = "scalar-quantile-v1"
    limits: PreferenceLimits = PreferenceLimits()
    version: str = "joint-sampling-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "joint-sampling-v1"
            or type(self.numerical) is not ParameterPreferences
            or type(self.categorical) is not CategoricalPreferences
            or type(self.events) is not tuple
            or any(type(event) is not JointDecision for event in self.events)
        ):
            raise ValueError("joint sampling requires frozen preference records and decisions")
        if self.numerical.default_mode != self.categorical.default_mode:
            raise ValueError("joint samplers must use the same default mode")
        if type(self.limits) is not PreferenceLimits or (
            self.numerical.limits != self.limits or self.categorical.limits != self.limits
        ):
            raise ValueError("joint limits must agree with both child records")
        self.limits.admit(
            draws=self.draw_count, values=self.value_count, parameters=self.parameter_count
        )
        if self.numerical.global_policy != numerical_policy(self.global_policy):
            raise ValueError("joint global policy disagrees with its numerical record")
        contracts: dict[tuple[str, str], NumericalParameter] = {}
        for draw in self.numerical.draws:
            mode = self.numerical.default_mode if draw.request.mode == "meta" else draw.request.mode
            key = (draw.request.name, mode)
            check_numerical_contract(draw.request, mode, self.global_policy, contracts.get(key))
            contracts[key] = draw.request
        roles = [(event.operation, event.role) for event in self.events]
        if len(set(roles)) != len(roles):
            raise ValueError("joint decision roles must be unique within an operation")
        validate_event_paths(self.events)
        for kind, draws in (
            ("numerical", self.numerical.draws),
            ("categorical", self.categorical.draws),
        ):
            if tuple(e.draw_index for e in self.events if e.kind == kind) != tuple(
                range(len(draws))
            ):
                raise ValueError("joint decisions must own every draw exactly once and in order")

    @property
    def draw_count(self) -> int:
        return len(self.numerical.draws) + self.categorical.draw_count + self.forced_count

    @property
    def forced_count(self) -> int:
        return sum(e.kind == "forced_identity" for e in self.events)

    @property
    def value_count(self) -> int:
        return sum(len(d.values) for d in self.numerical.draws) + self.categorical.value_count

    @property
    def parameter_count(self) -> int:
        return (
            self.numerical.parameter_count
            + self.categorical.parameter_count
            + len(self.events)
            - self.forced_count
        )


def joint_sampling_from_dict(data: dict[str, Any]) -> JointSamplingRecord:
    fields = dict(data)
    limits = PreferenceLimits(**fields.get("limits", {}))
    events_raw = fields["events"]
    limits.admit(draws=len(events_raw), values=0, parameters=0)
    numerical = raw_numerical_admission(fields["numerical"])
    categorical = raw_categorical_admission(fields["categorical"])
    forced = sum(e.get("kind") == "forced_identity" for e in events_raw)
    limits.admit(
        draws=numerical["draws"] + categorical["draws"] + forced,
        values=numerical["values"] + categorical["values"],
        parameters=numerical["parameters"] + categorical["parameters"] + len(events_raw) - forced,
    )
    fields["limits"] = limits
    fields["numerical"] = parameter_preferences_from_dict(fields["numerical"])
    fields["categorical"] = categorical_preferences_from_dict(fields["categorical"])
    events: list[JointDecision] = []
    for raw in fields["events"]:
        event = dict(raw)
        if type(event["operation"]) not in (list, tuple):
            raise ValueError("joint operation path must be an array")
        if not 1 <= len(event["operation"]) <= 16:
            raise ValueError("joint operation path depth must be 1..16")
        event["operation"] = tuple(event["operation"])
        events.append(JointDecision(**event))
    fields["events"] = tuple(events)
    return JointSamplingRecord(**fields)
