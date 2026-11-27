"""Atomic synthetic-only fits retaining backend export and three RNG streams."""

import importlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan
from r0.modules.priors.lib.xgboost_vector import FrozenXGBoostVector
from r0.modules.priors.lib.xgboost_vector_export import export_xgboost_vector

_NAMES = ("input", "target", "backend")
_DEFAULT_LIMITS = CausalLimits()
_DEFAULT_RAW = RawDrawLimits()


def canonical_xgboost_record(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class XGBoostFit:
    plan: XGBoostFitPlan
    function: FrozenXGBoostVector
    model_json: str
    backend_parameters: str
    initial_rng: tuple[GeneratorSnapshot, ...]
    final_rng: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "independent-xgboost-fit-v1"
    adaptation: str = "r0-iid-normal-inputs-normal-or-uniform-labels-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "independent-xgboost-fit-v1"
            or self.adaptation != "r0-iid-normal-inputs-normal-or-uniform-labels-v1"
            or type(self.plan) is not XGBoostFitPlan
            or type(self.function) is not FrozenXGBoostVector
            or type(self.model_json) is not str
            or not self.model_json.isascii()
            or not 1 <= len(self.model_json) <= 16_000_000
            or type(self.backend_parameters) is not str
            or len(self.backend_parameters) > 8192
            or any(
                type(a) is not tuple or len(a) != 3
                for a in (self.initial_rng, self.final_rng, self.raw)
            )
            or any(type(s) is not GeneratorSnapshot for s in (*self.initial_rng, *self.final_rng))
            or any(type(r) is not RawDrawRecord for r in self.raw)
        ):
            raise ValueError("invalid XGBoost fit custody fields")
        self.validate()

    def validate(self, *, limits: CausalLimits | None = None) -> None:
        active = self.limits if limits is None else limits
        cost = self.plan.admit(active, self.raw_limits)
        function_record_cost(self, active)
        # Charge retained native text and both parser representations before decoding.
        if cost + 12 * len(self.model_json) > active.working_bytes:
            raise ValueError("XGBoost fit native model replay workspace limit exceeded")
        for i, (spec, name, raw) in enumerate(
            zip(self.plan.raw_specs(), _NAMES, self.raw, strict=True)
        ):
            if (
                raw.sequence != i
                or raw.stream != name
                or raw.owner != spec.owner
                or raw.role != spec.role
                or raw.distribution != spec.distribution
                or raw.shape != spec.shape
                or canonical_xgboost_record(raw.parameters)
                != canonical_xgboost_record(spec.parameters)
            ):
                raise ValueError("XGBoost fit raw declaration differs from plan")
        streams = tuple(s.restore(byte_limit=self.raw_limits.state_bytes) for s in self.initial_rng)
        for raw, rng in zip(self.raw, streams, strict=True):
            raw.replay(rng)
        final = tuple(
            snapshot_generator(r, byte_limit=self.raw_limits.state_bytes) for r in streams
        )
        if final != self.final_rng:
            raise ValueError("XGBoost fit final RNG custody differs")
        seed = self.raw[-1].values[0]
        if type(seed) is not int or self.backend_parameters != canonical_xgboost_record(
            self.plan.backend_parameters(seed)
        ):
            raise ValueError("XGBoost fit backend controls differ from frozen plan")
        actual = export_xgboost_vector(self.model_json.encode("ascii"))
        if canonical_xgboost_record(asdict(actual)) != canonical_xgboost_record(
            asdict(self.function)
        ):
            raise ValueError("XGBoost fit function differs from native model export")
        if (actual.input_width, actual.output_width, actual.objective, len(actual.trees)) != (
            self.plan.input_width,
            self.plan.output_width,
            self.plan.objective,
            self.plan.estimators,
        ):
            raise ValueError("XGBoost fitted model geometry differs from plan")
        for tree in actual.trees:
            depth = [0] * len(tree.features)
            for i, f in enumerate(tree.features):
                if f >= 0:
                    depth[tree.left[i]] = depth[tree.right[i]] = depth[i] + 1
            if max(depth) > self.plan.depth:
                raise ValueError("XGBoost fitted tree exceeds declared depth")


def fit_xgboost(
    plan: XGBoostFitPlan,
    *,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
    raw_limits: RawDrawLimits = _DEFAULT_RAW,
) -> XGBoostFit:
    """Draw only synthetic fit rows, fit once and freeze; failures roll back all streams.

    This is the independent construction lifecycle. Episode/support/batch refits
    require separately named entrypoints; no query rows enter this API.
    """
    if type(plan) is not XGBoostFitPlan:
        raise ValueError("XGBoost fitting requires its immutable plan")
    plan.admit(limits, raw_limits)
    backend = importlib.import_module("xgboost")
    if backend.__version__ != "3.4.1":
        raise ValueError("XGBoost fitting requires CPU backend version3.4.1")
    ledger = RawDrawLedger(limits=raw_limits)
    with ledger.transactions(
        dict(zip(_NAMES, (input_rng, target_rng, backend_rng), strict=True))
    ) as (streams, capture):
        initial = tuple(
            snapshot_generator(r, byte_limit=raw_limits.state_bytes) for r in streams.values()
        )
        draws: list[NDArray[Any]] = []
        for name, spec in zip(_NAMES, plan.raw_specs(), strict=True):
            draws.append(
                capture.draw(
                    streams[name],
                    owner=spec.owner,
                    role=spec.role,
                    distribution=spec.distribution,
                    shape=spec.shape,
                    parameters=spec.parameters,
                )
            )
        parameters = plan.backend_parameters(int(draws[-1][0]))
        model = backend.train(
            parameters, backend.DMatrix(draws[0], label=draws[1]), num_boost_round=plan.estimators
        )
        model_json = bytes(model.save_raw(raw_format="json")).decode("ascii")
        function = export_xgboost_vector(model_json.encode("ascii"))
        result = XGBoostFit(
            plan,
            function,
            model_json,
            canonical_xgboost_record(parameters),
            initial,
            tuple(
                snapshot_generator(r, byte_limit=raw_limits.state_bytes) for r in streams.values()
            ),
            capture.freeze(),
            limits,
            raw_limits,
        )
    return result
