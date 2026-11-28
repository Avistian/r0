"""Shared polynomial weight parameters with separately materialized innovations."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_plan_bindings import numerical_binding, operation_events
from r0.modules.priors.lib.joint_sampling import JointSamplingContext
from r0.modules.priors.lib.joint_sampling_records import JointSamplingRecord, validate_path
from r0.modules.priors.lib.matrix_preference_prior import vector_requests
from r0.modules.priors.lib.parameter_preferences import NumericalParameter
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transaction, recorded_draw
from r0.modules.priors.lib.weight_vectors import (
    PolynomialWeightVector,
    UniformWeightVector,
    WeightVector,
)

_DEFAULT_LIMITS = CausalLimits()
_SCALARIZED = "scalarized-weight-batches-v1"
_SOURCE_BATCHES = "source-weight-batches-v1"


def _requests(width: int, batch: int, policy: str) -> tuple[NumericalParameter, NumericalParameter]:
    decay, scale = vector_requests(width)
    if policy == _SOURCE_BATCHES:
        return replace(decay, shape=(batch,)), replace(scale, shape=(batch,))
    return decay, scale


@dataclass(frozen=True, slots=True)
class JointWeightPlan:
    operation: tuple[str, ...]
    width: int
    batch: int = 1
    family: str = "polynomial"
    decays: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    version: str = "joint-weight-plan-v1"
    scalarization: str = _SCALARIZED

    def __post_init__(self) -> None:
        validate_path(self.operation)
        if any(type(n) is not int or not 1 <= n < 2**63 for n in (self.width, self.batch)):
            raise ValueError("joint weight width and batch must lie in [1,2**63)")
        if self.version != "joint-weight-plan-v1" or self.family not in ("polynomial", "uniform"):
            raise ValueError("joint weights require their version and a supported family")
        if self.scalarization not in (_SCALARIZED, _SOURCE_BATCHES):
            raise ValueError("unsupported joint weight scalarization policy")
        expected = self.batch if self.family == "polynomial" else 0
        if any(
            type(values) is not tuple or len(values) != expected
            for values in (self.decays, self.scales)
        ):
            raise ValueError("weight hyperparameters must match the declared family and batch")
        if self.family == "polynomial":
            for values, request in zip(
                (self.decays, self.scales),
                _requests(self.width, self.batch, self.scalarization),
                strict=True,
            ):
                if any(
                    not finite_number(v) or not request.low <= v <= request.high for v in values
                ):
                    raise ValueError("polynomial hyperparameter lies outside its source bounds")

    @property
    def parameter_count(self) -> int:
        return 2 + len(self.decays) + len(self.scales)

    @property
    def realization_parameter_count(self) -> int:
        return self.batch * (2 * self.width + 2 if self.family == "polynomial" else self.width)

    def validate(self, decisions: JointSamplingRecord) -> None:
        events = operation_events(decisions, self.operation)
        scalar = self.scalarization == _SCALARIZED
        count = (2 * self.batch if scalar else 2) if self.family == "polynomial" else 0
        if len(events) != count:
            raise ValueError("weight plan decision count differs from its declared batch")
        roles = (
            (
                tuple(f"{role}:{i}" for role in ("decay", "scale") for i in range(self.batch))
                if scalar
                else ("decay", "scale")
            )
            if count
            else ()
        )
        if tuple(e.role for e in events) != roles or any(
            e.operation != self.operation for e in events
        ):
            raise ValueError("weight plan must own exactly its ordered hyperparameter decisions")
        if self.family == "polynomial":
            requests = _requests(self.width, self.batch, self.scalarization)
            for index, (values, request) in enumerate(
                zip((self.decays, self.scales), requests, strict=True)
            ):
                if scalar:
                    selected = events[index * self.batch : (index + 1) * self.batch]
                    expected = tuple(
                        numerical_binding(e, decisions, request).values[0] for e in selected
                    )
                else:
                    expected = numerical_binding(events[index], decisions, request).values
                if values != expected:
                    raise ValueError(
                        "weight hyperparameters differ from their recorded realizations"
                    )


def sample_joint_weights(
    context: JointSamplingContext,
    name: str,
    *,
    width: int,
    batch: int = 1,
    family: str = "polynomial",
    scalarization: str = _SCALARIZED,
) -> JointWeightPlan:
    if type(context) is not JointSamplingContext:
        raise ValueError("joint weight sampling requires a shared context")
    if any(type(n) is not int or not 1 <= n < 2**63 for n in (width, batch)) or family not in (
        "polynomial",
        "uniform",
    ):
        raise ValueError(
            "joint weight sampling requires a positive bounded width and supported family"
        )
    if scalarization not in (_SCALARIZED, _SOURCE_BATCHES):
        raise ValueError("unsupported joint weight scalarization policy")
    with context.transaction(name) as operation:
        decays: tuple[float, ...] = ()
        scales: tuple[float, ...] = ()
        if family == "polynomial":
            decay_request, scale_request = _requests(width, batch, scalarization)
            scalar = scalarization == _SCALARIZED
            repeats, draw_width = (batch, 1) if scalar else (1, batch)
            before = operation.freeze()
            # Admit batch cardinality before allocating retained parameter tuples.
            # Native draws then admit exact new/cached preference costs incrementally.
            minimum_parameters = repeats * sum(
                request.parameter_count + 2 + 2 * draw_width
                for request in (decay_request, scale_request)
            )
            before.limits.admit(
                draws=before.draw_count + 2 * repeats,
                values=before.value_count + 2 * batch,
                parameters=before.parameter_count + minimum_parameters,
            )
            if scalar:
                decays = tuple(
                    operation.numerical(f"decay:{i}", decay_request).values[0] for i in range(batch)
                )
                scales = tuple(
                    operation.numerical(f"scale:{i}", scale_request).values[0] for i in range(batch)
                )
            else:
                decays = operation.numerical("decay", decay_request).values
                scales = operation.numerical("scale", scale_request).values
        plan = JointWeightPlan(
            operation.operation, width, batch, family, decays, scales, scalarization=scalarization
        )
        plan.validate(operation.freeze())
    return plan


def materialize_joint_weights(
    plan: JointWeightPlan,
    decisions: JointSamplingRecord,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
    capture: RawDrawLedger | None = None,
) -> tuple[WeightVector, ...]:
    """Admit the full output batch before arrays and atomically commit its RNG.

    Graph-wide retained plans/decisions and simultaneous arrays need separate
    aggregate admission. Uniform plans have no named hyperparameter decisions.
    """
    if type(plan) is not JointWeightPlan:
        raise ValueError("weight materialization requires a frozen joint plan")
    plan.validate(decisions)
    if type(rng) is not np.random.Generator or type(limits) is not CausalLimits:
        raise ValueError("weight materialization requires injected RNG and frozen limits")
    limits.admit_dimensions(
        (plan.width,),
        ((),),
        rows=plan.batch,
        calibration_rows=0,
        features=plan.width,
        outputs=plan.width,
        extra_parameters=plan.realization_parameter_count,
        extra_working_channels=4 * plan.width,
    )
    with raw_transaction(rng, capture) as (staged_rng, capture):
        vectors: tuple[WeightVector, ...]
        if plan.family == "polynomial":
            draws = recorded_draw(
                staged_rng,
                capture,
                owner=plan.operation,
                role="weight-normals",
                distribution="normal",
                shape=(plan.batch, plan.width),
                parameters=(),
            )
            vectors = tuple(
                PolynomialWeightVector(
                    tuple(float(v) for v in row),
                    decay=decay,
                    log_noise_scale=scale,
                    permutation=tuple(
                        int(i)
                        for i in recorded_draw(
                            staged_rng,
                            capture,
                            owner=plan.operation,
                            role="weight-permutation",
                            distribution="permutation",
                            shape=(plan.width,),
                            parameters=(("count", plan.width),),
                        )
                    ),
                )
                for row, decay, scale in zip(draws, plan.decays, plan.scales, strict=True)
            )
        else:
            draws = recorded_draw(
                staged_rng,
                capture,
                owner=plan.operation,
                role="weight-uniforms",
                distribution="random",
                shape=(plan.batch, plan.width),
                parameters=(),
            )
            vectors = tuple(UniformWeightVector(tuple(float(v) for v in row)) for row in draws)
        return vectors


def joint_weight_plan_from_dict(
    data: dict[str, Any], *, decisions: JointSamplingRecord
) -> JointWeightPlan:
    fields = dict(data)
    path = fields["operation"]
    if type(path) not in (list, tuple) or not 1 <= len(path) <= 16:
        raise ValueError("weight operation path must be a bounded array")
    fields["operation"] = tuple(path)
    batch = fields.get("batch", 1)
    if type(batch) is not int or not 1 <= batch < 2**63:
        raise ValueError("weight batch must lie in [1,2**63)")
    count = batch if fields.get("family", "polynomial") == "polynomial" else 0
    source = fields.get("scalarization", _SCALARIZED) == _SOURCE_BATCHES
    draws = (2 if source else 2 * count) if count else 0
    if source and count > 100_000:
        raise ValueError("source weight batch exceeds the numerical shape limit")
    if type(decisions) is not JointSamplingRecord or draws > len(decisions.numerical.draws):
        raise ValueError("weight batch exceeds its available numerical decisions")
    if 2 * count > sum(len(d.values) for d in decisions.numerical.draws):
        raise ValueError("weight batch exceeds its available numerical values")
    for name in ("decays", "scales"):
        values = fields.get(name, ())
        if type(values) not in (list, tuple) or len(values) != count:
            raise ValueError("weight hyperparameters must be arrays matching the declared batch")
        fields[name] = tuple(values)
    plan = JointWeightPlan(**fields)
    plan.validate(decisions)
    return plan
