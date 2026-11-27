"""Independent prototype banks and batch-reference mixtures with owned raw replay."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.prototype_root_emission import PrototypeRootDraw, draw_prototype_root
from r0.modules.priors.lib.prototype_root_records import (
    FrozenPrototypeRoot,
    PrototypeRootPlan,
    admit_prototype_work,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.source_root_base import functional_root_values

_DEFAULT_LIMITS = CausalLimits()
_BANK_OWNER = ("prototype-bank",)
_WEIGHT_OWNER = ("prototype-emission",)
_SOURCE = "p2-convex-mixture-reconstruction-r0-v1"


def bank_draw_specs(plan: PrototypeRootPlan) -> tuple[RawDrawSpec, ...]:
    base = plan.base
    if isinstance(base, JointRootBase):
        uniform = base.family == "uniform" or base.covariance_base == "uniform"
        main = RawDrawSpec(
            _BANK_OWNER,
            "root-base-values",
            "random" if uniform else "normal",
            (plan.count, base.width),
        )
        return (
            (main, RawDrawSpec(_BANK_OWNER, "root-base-radii", "random", (plan.count,)))
            if base.family == "ball"
            else (main,)
        )
    specs: list[RawDrawSpec] = []
    for i, c in enumerate(base.law.coordinates):
        family = (
            "normal"
            if c.family == "gaussian"
            else "choice"
            if c.family == "categorical"
            else "zipf"
            if c.family == "zipf"
            else "random"
        )
        parameters = (
            (("count", len(c.probabilities)), ("probabilities", c.probabilities))
            if c.family == "categorical"
            else (("exponent", c.exponent),)
            if c.family == "zipf"
            else ()
        )
        specs.append(
            RawDrawSpec((*_BANK_OWNER, f"column:{i}"), "values", family, (plan.count,), parameters)
        )
    return tuple(specs)


def admit_prototype_plan(plan: PrototypeRootPlan, limits: CausalLimits, *, batch: bool) -> None:
    if type(plan) is not PrototypeRootPlan or type(limits) is not CausalLimits:
        raise ValueError("prototype construction requires frozen plan and limits")
    parameters, storage = function_record_cost(plan, limits)
    admit_prototype_work(
        plan.count,
        plan.base.width,
        plan.reference_rows if batch else 1,
        limits,
        metadata_parameters=parameters,
        metadata_bytes=4 * storage + 8 * plan.raw_limits.state_bytes,
    )


def preflight_raw(plan: PrototypeRootPlan, *, batch: bool) -> RawDrawLedger:
    ledger = RawDrawLedger(limits=plan.raw_limits)
    specs = bank_draw_specs(plan)
    if batch:
        specs += (plan.weights.draw_spec(plan.count, plan.reference_rows, _WEIGHT_OWNER),)
    # Charge the longest eventual stream name before any RNG snapshot or clone.
    ledger.admit_specs(specs, stream="prototype")
    return ledger


def realize_prototype_bank(
    plan: PrototypeRootPlan, rng: np.random.Generator, ledger: RawDrawLedger, limits: CausalLimits
) -> FrozenPrototypeRoot:
    ledger.admit_draws(rng, bank_draw_specs(plan))
    values = functional_root_values(
        plan.base, plan.count, rng, capture=ledger, owner=_BANK_OWNER, limits=limits
    )
    return FrozenPrototypeRoot(frozen_matrix(values), plan.weights)


@dataclass(frozen=True, slots=True)
class SampledPrototypeRoot:
    plan: PrototypeRootPlan
    root: FrozenPrototypeRoot
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits = _DEFAULT_LIMITS
    source: str = _SOURCE
    lifecycle: str = "frozen-independent-prototype-bank-r0-v1"
    version: str = "sampled-prototype-root-v1"

    def __post_init__(self) -> None:
        admit_prototype_plan(self.plan, self.limits, batch=False)
        if (
            self.version != "sampled-prototype-root-v1"
            or self.source != _SOURCE
            or self.lifecycle != "frozen-independent-prototype-bank-r0-v1"
            or type(self.root) is not FrozenPrototypeRoot
            or self.root.count != self.plan.count
            or self.root.width != self.plan.base.width
            or self.root.weights != self.plan.weights
            or type(self.initial) is not GeneratorSnapshot
            or type(self.final) is not GeneratorSnapshot
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
        ):
            raise ValueError("invalid sampled prototype bank envelope")
        function_record_cost(self, self.limits)
        self.validate_replay(limits=self.limits)

    @property
    def construction_moment_order(self) -> float:
        return self.plan.construction_moment_order

    def validate_replay(self, *, limits: CausalLimits = _DEFAULT_LIMITS) -> None:
        admit_prototype_plan(self.plan, limits, batch=False)
        ledger = preflight_raw(self.plan, batch=False)
        rng = self.initial.restore(byte_limit=self.plan.raw_limits.state_bytes)
        with ledger.transactions({"prototype": rng}) as (streams, capture):
            root = realize_prototype_bank(self.plan, streams["prototype"], capture, limits)
            final = snapshot_generator(
                streams["prototype"], byte_limit=self.plan.raw_limits.state_bytes
            )
            if root != self.root or final != self.final or capture.freeze() != self.raw:
                raise ValueError("sampled prototype bank differs from raw replay")


def sample_prototype_root(
    plan: PrototypeRootPlan, *, rng: np.random.Generator, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledPrototypeRoot:
    admit_prototype_plan(plan, limits, batch=False)
    ledger = preflight_raw(plan, batch=False)
    with ledger.transactions({"prototype": rng}) as (streams, capture):
        active = streams["prototype"]
        initial = snapshot_generator(active, byte_limit=plan.raw_limits.state_bytes)
        root = realize_prototype_bank(plan, active, capture, limits)
        return SampledPrototypeRoot(
            plan,
            root,
            initial,
            snapshot_generator(active, byte_limit=plan.raw_limits.state_bytes),
            capture.freeze(),
            limits,
        )


def realize_prototype_batch(
    plan: PrototypeRootPlan,
    streams: dict[str, np.random.Generator],
    ledger: RawDrawLedger,
    limits: CausalLimits,
) -> tuple[FrozenPrototypeRoot, PrototypeRootDraw]:
    # "prototype" is the longer name, conservatively charging all bound stream metadata.
    ledger.admit_draws(
        streams["prototype"],
        (
            *bank_draw_specs(plan),
            plan.weights.draw_spec(plan.count, plan.reference_rows, _WEIGHT_OWNER),
        ),
    )
    root = realize_prototype_bank(plan, streams["prototype"], ledger, limits)
    draw = draw_prototype_root(
        root,
        rows=plan.reference_rows,
        rng=streams["weights"],
        capture=ledger,
        limits=limits,
        owner=_WEIGHT_OWNER,
    )
    return root, draw


@dataclass(frozen=True, slots=True)
class PrototypeBatchReference:
    plan: PrototypeRootPlan
    root: FrozenPrototypeRoot
    draw: PrototypeRootDraw
    initial: tuple[GeneratorSnapshot, ...]
    final: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits = _DEFAULT_LIMITS
    source: str = _SOURCE
    lifecycle: str = "prototype-batch-reference-r0-v1"
    version: str = "prototype-batch-reference-v1"

    def __post_init__(self) -> None:
        admit_prototype_plan(self.plan, self.limits, batch=True)
        if (
            self.version != "prototype-batch-reference-v1"
            or self.source != _SOURCE
            or self.lifecycle != "prototype-batch-reference-r0-v1"
            or type(self.root) is not FrozenPrototypeRoot
            or type(self.draw) is not PrototypeRootDraw
            or len(self.draw.values) != self.plan.reference_rows
            or type(self.raw) is not tuple
            or any(type(r) is not RawDrawRecord for r in self.raw)
            or any(
                type(seq) is not tuple
                or len(seq) != 2
                or any(type(s) is not GeneratorSnapshot for s in seq)
                for seq in (self.initial, self.final)
            )
        ):
            raise ValueError("invalid prototype batch reference")
        function_record_cost(self, self.limits)
        self.validate_replay(limits=self.limits)

    def validate_replay(self, *, limits: CausalLimits = _DEFAULT_LIMITS) -> None:
        admit_prototype_plan(self.plan, limits, batch=True)
        ledger = preflight_raw(self.plan, batch=True)
        streams: dict[str, np.random.Generator] = dict(
            zip(
                ("prototype", "weights"),
                (s.restore(byte_limit=self.plan.raw_limits.state_bytes) for s in self.initial),
                strict=True,
            )
        )
        with ledger.transactions(streams) as (active, capture):
            result = realize_prototype_batch(self.plan, active, capture, limits)
            final = tuple(
                snapshot_generator(active[n], byte_limit=self.plan.raw_limits.state_bytes)
                for n in ("prototype", "weights")
            )
            if (
                result != (self.root, self.draw)
                or final != self.final
                or capture.freeze() != self.raw
            ):
                raise ValueError("prototype batch differs from raw replay")


def sample_prototype_batch_reference(
    plan: PrototypeRootPlan,
    *,
    prototype_rng: np.random.Generator,
    weights_rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PrototypeBatchReference:
    admit_prototype_plan(plan, limits, batch=True)
    ledger = preflight_raw(plan, batch=True)
    with ledger.transactions({"prototype": prototype_rng, "weights": weights_rng}) as (
        active,
        capture,
    ):
        initial = tuple(
            snapshot_generator(active[n], byte_limit=plan.raw_limits.state_bytes)
            for n in ("prototype", "weights")
        )
        root, draw = realize_prototype_batch(plan, active, capture, limits)
        final = tuple(
            snapshot_generator(active[n], byte_limit=plan.raw_limits.state_bytes)
            for n in ("prototype", "weights")
        )
        return PrototypeBatchReference(plan, root, draw, initial, final, capture.freeze(), limits)
