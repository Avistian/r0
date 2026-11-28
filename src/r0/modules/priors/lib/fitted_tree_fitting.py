"""Atomic independent synthetic fits with raw input/target/seed custody."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_validation import validate_fitted_custody
from r0.modules.priors.lib.fitted_tree_backend import fit_backend
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.fitted_tree_records import FrozenFittedTreeFunction
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class FittedTreeRealization:
    plan: FittedTreePlan
    function: FrozenFittedTreeFunction
    inputs: Matrix
    targets: Matrix
    input_law: str
    backend_seeds: tuple[int, ...]
    backend_parameters: tuple[str, ...]
    backend_version: str
    initial_rng: tuple[GeneratorSnapshot, ...]
    final_rng: tuple[GeneratorSnapshot, ...]
    raw_draws: tuple[RawDrawRecord, ...]
    version: str = "independent-fitted-tree-realization-v1"
    adaptation: str = "r0-independent-synthetic-fit-normal-or-uniform-class-targets-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "independent-fitted-tree-realization-v1"
            or self.adaptation != "r0-independent-synthetic-fit-normal-or-uniform-class-targets-v1"
            or type(self.plan) is not FittedTreePlan
            or type(self.function) is not FrozenFittedTreeFunction
            or self.input_law not in ("iid-standard-normal", "independent-frozen-ancestor-prefix")
            or matrix_shape(self.inputs) != (self.plan.rows, self.plan.input_width)
            or matrix_shape(self.targets) != (self.plan.rows, self.plan.output_width)
            or (self.function.input_width, self.function.output_width)
            != (self.plan.input_width, self.plan.output_width)
            or type(self.backend_seeds) is not tuple
            or len(self.backend_seeds) != self.plan.output_width
            or any(type(s) is not int or not 0 <= s < 2**31 for s in self.backend_seeds)
            or type(self.backend_parameters) is not tuple
            or len(self.backend_parameters) != self.plan.output_width
            or type(self.backend_version) is not str
            or not self.backend_version
            or any(type(v) is not str for v in self.backend_parameters)
            or any(type(v) is not tuple for v in (self.initial_rng, self.final_rng, self.raw_draws))
            or len(self.initial_rng) != 3
            or len(self.final_rng) != 3
            or any(type(v) is not GeneratorSnapshot for v in (*self.initial_rng, *self.final_rng))
            or any(type(v) is not RawDrawRecord for v in self.raw_draws)
        ):
            raise ValueError("fitted realization geometry, provenance or version differs")
        validate_fitted_custody(self)

    @property
    def input_digest(self) -> str:
        return record_fingerprint(self.inputs)

    @property
    def target_digest(self) -> str:
        return record_fingerprint(self.targets)


def fit_fitted_tree(
    plan: FittedTreePlan,
    *,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT_LIMITS,
    inputs: NDArray[np.float64] | None = None,
) -> FittedTreeRealization:
    """Fit synthetic-only rows once. Provided inputs must be independent ancestor rows.

    Callers must not pass episode/query rows. The sampled graph constructor owns
    this boundary. Backend seeds describe sklearn randomness, not captured internal draws.
    """
    if type(plan) is not FittedTreePlan or type(limits) is not CausalLimits:
        raise ValueError("fitting requires its frozen plan and caller limits")
    plan.admit(limits)
    if inputs is not None and (
        type(inputs) is not np.ndarray
        or inputs.dtype != np.float64
        or inputs.shape != (plan.rows, plan.input_width)
        or not np.isfinite(inputs).all()
    ):
        raise ValueError("synthetic parent fitting population has invalid geometry or values")
    ledger = RawDrawLedger()
    with ledger.transactions(
        {"input": input_rng, "target": target_rng, "backend": backend_rng}
    ) as (streams, capture):
        initial = tuple(snapshot_generator(rng) for rng in streams.values())
        owner = ("fitted-tree", "independent-fit")
        x = inputs
        if x is None:
            x = recorded_draw(
                streams["input"],
                capture,
                owner=owner,
                role="synthetic-inputs",
                distribution="normal",
                shape=(plan.rows, plan.input_width),
                parameters=(),
            ).astype(np.float64)
        classifier = plan.task == "classification"
        y = recorded_draw(
            streams["target"],
            capture,
            owner=owner,
            role="synthetic-targets",
            distribution="integers" if classifier else "normal",
            shape=(plan.rows, plan.output_width),
            parameters=(("high", plan.classes),) if classifier else (),
        ).astype(np.float64)
        seeds = tuple(
            map(
                int,
                recorded_draw(
                    streams["backend"],
                    capture,
                    owner=owner,
                    role="backend-seeds",
                    distribution="integers",
                    shape=(plan.output_width,),
                    parameters=(("high", 2**31),),
                ),
            )
        )
        outputs, params, backend_version = fit_backend(plan, x, y, seeds)
        function = FrozenFittedTreeFunction(plan.input_width, outputs)
        return FittedTreeRealization(
            plan,
            function,
            frozen_matrix(x),
            frozen_matrix(y),
            "iid-standard-normal" if inputs is None else "independent-frozen-ancestor-prefix",
            seeds,
            params,
            backend_version,
            initial,
            tuple(snapshot_generator(rng) for rng in streams.values()),
            capture.freeze(),
        )
