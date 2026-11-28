"""Atomic XGBoost fitting to retained actual inputs with immutable model custody."""

import importlib
from dataclasses import InitVar, asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape
from r0.modules.priors.lib.xgboost_fitting import canonical_xgboost_record
from r0.modules.priors.lib.xgboost_population_plan import PopulationXGBoostPlan
from r0.modules.priors.lib.xgboost_vector import FrozenXGBoostVector
from r0.modules.priors.lib.xgboost_vector_export import export_xgboost_vector

_DEFAULT = CausalLimits()
_RAW = RawDrawLimits()
_NAMES = ("target", "backend")
_CONTRACTS = {
    "support-only": "r0-actual-synthetic-support-fixed-xgboost-v1",
    "batch-refit-reference": "tabicl-pinned-actual-batch-xgboost-r0-settings-v1",
    "independent-ancestors": "r0-independent-synthetic-ancestor-xgboost-v1",
}


@dataclass(frozen=True, slots=True)
class PopulationXGBoostMetadata:
    plan: PopulationXGBoostPlan
    function: FrozenXGBoostVector
    inputs: Matrix
    targets: Matrix
    population: str
    model_json: str
    backend_parameters: str
    initial_rng: tuple[GeneratorSnapshot, ...]
    final_rng: tuple[GeneratorSnapshot, ...]
    raw: tuple[RawDrawRecord, ...]
    limits: CausalLimits
    raw_limits: RawDrawLimits
    source_contract: str
    version: str = "population-xgboost-fit-v1"

    caller_limits: InitVar[CausalLimits | None] = None

    def __post_init__(self, caller_limits: CausalLimits | None) -> None:
        self.preflight(self.limits if caller_limits is None else caller_limits)

    def preflight(self, limits: CausalLimits) -> None:
        if (
            self.version != "population-xgboost-fit-v1"
            or type(self.plan) is not PopulationXGBoostPlan
            or self.population not in _CONTRACTS
            or self.source_contract != _CONTRACTS[self.population]
            or type(self.function) is not FrozenXGBoostVector
            or type(self.limits) is not CausalLimits
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid population XGBoost lifecycle or records")
        cost = self.plan.admit(limits, self.raw_limits)
        function_record_cost(self, limits)
        if (
            type(self.model_json) is not str
            or not self.model_json.isascii()
            or not 1 <= len(self.model_json) <= min(16_000_000, self.plan.model_byte_bound)
            or cost + 12 * len(self.model_json) > limits.working_bytes
            or type(self.backend_parameters) is not str
            or len(self.backend_parameters) > 8192
        ):
            raise ValueError("population XGBoost model metadata/workspace exceeds limits")
        if matrix_shape(self.inputs) != (self.plan.rows, self.plan.input_width) or matrix_shape(
            self.targets
        ) != (self.plan.rows, self.plan.target_width):
            raise ValueError("population XGBoost actual fit table geometry differs")
        if any(
            type(v) is not tuple or len(v) != 2
            for v in (self.initial_rng, self.final_rng, self.raw)
        ):
            raise ValueError("population XGBoost requires two complete stream records")
        for state in (*self.initial_rng, *self.final_rng):
            if type(state) is not GeneratorSnapshot:
                raise ValueError("invalid XGBoost snapshot")
            state.validate_metadata(byte_limit=self.raw_limits.state_bytes)
        for i, (spec, name, raw) in enumerate(
            zip(self.plan.raw_specs(), _NAMES, self.raw, strict=True)
        ):
            if (
                type(raw) is not RawDrawRecord
                or raw.sequence != i
                or raw.stream != name
                or raw.owner != spec.owner
                or raw.role != spec.role
                or raw.distribution != spec.distribution
                or raw.shape != spec.shape
                or canonical_xgboost_record(raw.parameters)
                != canonical_xgboost_record(spec.parameters)
            ):
                raise ValueError("population XGBoost raw declaration differs from plan")
        if tuple(v for row in self.targets for v in row) != self.raw[0].values:
            raise ValueError("population XGBoost targets differ from retained raw draws")
        seed = self.raw[1].values[0]
        if type(seed) is not int or self.backend_parameters != canonical_xgboost_record(
            self.plan.backend_parameters(seed)
        ):
            raise ValueError("population XGBoost backend parameters differ")
        actual = export_xgboost_vector(self.model_json.encode("ascii"), limits=limits)
        if canonical_xgboost_record(asdict(actual)) != canonical_xgboost_record(
            asdict(self.function)
        ):
            raise ValueError("population XGBoost native export differs")
        if (actual.input_width, actual.output_width, actual.objective, len(actual.trees)) != (
            self.plan.input_width,
            self.plan.output_width,
            self.plan.objective,
            self.plan.estimators,
        ):
            raise ValueError("population XGBoost native model geometry differs")
        for tree in actual.trees:
            depth = [0] * len(tree.features)
            for i, feature in enumerate(tree.features):
                if feature >= 0:
                    depth[tree.left[i]] = depth[tree.right[i]] = depth[i] + 1
            if max(depth) > self.plan.depth:
                raise ValueError("population XGBoost tree depth differs")


@dataclass(frozen=True, slots=True)
class PopulationXGBoostFit(PopulationXGBoostMetadata):
    def __post_init__(self, caller_limits: CausalLimits | None) -> None:
        self.validate(limits=caller_limits)

    def validate(self, *, limits: CausalLimits | None = None) -> None:
        cap = self.limits if limits is None else limits
        self.preflight(cap)
        streams = tuple(s.restore(byte_limit=self.raw_limits.state_bytes) for s in self.initial_rng)
        for raw, rng in zip(self.raw, streams, strict=True):
            raw.replay(rng)
        if (
            tuple(snapshot_generator(r, byte_limit=self.raw_limits.state_bytes) for r in streams)
            != self.final_rng
        ):
            raise ValueError("population XGBoost final RNG custody differs")


def fit_population_xgboost(
    plan: PopulationXGBoostPlan,
    *,
    inputs: NDArray[np.float64],
    population: str,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT,
    raw_limits: RawDrawLimits = _RAW,
) -> PopulationXGBoostFit:
    if type(plan) is not PopulationXGBoostPlan or population not in _CONTRACTS:
        raise ValueError("population XGBoost needs an explicit plan and lifecycle")
    cost = plan.admit(limits, raw_limits)
    if (
        cost + 12 * plan.model_byte_bound + 256 * plan.rows * (plan.input_width + plan.target_width)
        > limits.working_bytes
        or plan.retained_number_bound > limits.parameters
    ):
        raise ValueError("population XGBoost complete prospective fit exceeds limits")
    if (
        type(inputs) is not np.ndarray
        or inputs.shape != (plan.rows, plan.input_width)
        or inputs.dtype != np.float64
        or not np.isfinite(inputs).all()
        or (np.abs(inputs) > np.finfo(np.float32).max).any()
    ):
        raise ValueError("population XGBoost inputs must be finite actual float64 rows")
    retained = frozen_matrix(inputs)
    backend = importlib.import_module("xgboost")
    if backend.__version__ != "3.4.1":
        raise ValueError("population XGBoost requires CPU backend3.4.1")
    ledger = RawDrawLedger(limits=raw_limits)
    with ledger.transactions(dict(zip(_NAMES, (target_rng, backend_rng), strict=True))) as (
        streams,
        capture,
    ):
        initial = tuple(
            snapshot_generator(r, byte_limit=raw_limits.state_bytes) for r in streams.values()
        )
        draws = [
            capture.draw(
                streams[name],
                owner=spec.owner,
                role=spec.role,
                distribution=spec.distribution,
                shape=spec.shape,
                parameters=spec.parameters,
            )
            for name, spec in zip(_NAMES, plan.raw_specs(), strict=True)
        ]
        parameters = plan.backend_parameters(int(draws[1][0]))
        model = backend.train(
            parameters, backend.DMatrix(inputs, label=draws[0]), num_boost_round=plan.estimators
        )
        model_json = bytes(model.save_raw(raw_format="json")).decode("ascii")
        result = PopulationXGBoostFit(
            plan,
            export_xgboost_vector(model_json.encode("ascii"), limits=limits),
            retained,
            frozen_matrix(draws[0]),
            population,
            model_json,
            canonical_xgboost_record(parameters),
            initial,
            tuple(
                snapshot_generator(r, byte_limit=raw_limits.state_bytes) for r in streams.values()
            ),
            capture.freeze(),
            limits,
            raw_limits,
            _CONTRACTS[population],
        )
    return result
