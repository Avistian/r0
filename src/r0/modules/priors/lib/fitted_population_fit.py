"""Atomic fitting with explicit population custody and immutable exported functions."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_validation import validate_backend_fit
from r0.modules.priors.lib.fitted_population_draws import population_draws
from r0.modules.priors.lib.fitted_population_plan import MixedFitPopulation, PopulationTreePlan
from r0.modules.priors.lib.fitted_tree_backend import fit_backend
from r0.modules.priors.lib.fitted_tree_records import FrozenFittedTreeFunction
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_point_realization import frozen_matrix
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT_LIMITS = CausalLimits()

_POPULATION_SOURCES = {
    "support-only": "oprior-v2-support-contract-r0-fit-settings-v1",
    "batch-refit-reference": "tabicl-pinned-batch-refit-r0-backend-settings-v1",
    "mixed-independent": "mitra-v1-mixed-fit-laws-r0-explicit-settings-v1",
}


@dataclass(frozen=True, slots=True)
class PopulationTreeFit:
    plan: PopulationTreePlan
    function: FrozenFittedTreeFunction
    inputs: Matrix
    targets: Matrix
    population: str
    mixed: MixedFitPopulation | None
    cardinalities: tuple[int, ...]
    backend_seeds: tuple[int, ...]
    backend_parameters: tuple[str, ...]
    backend_version: str
    initial_rng: tuple[GeneratorSnapshot, ...]
    final_rng: tuple[GeneratorSnapshot, ...]
    raw_draws: tuple[RawDrawRecord, ...]
    version: str = "population-fitted-tree-realization-v1"
    source_contract: str = ""
    backend_semantics: str = "sklearn-independent-output-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "population-fitted-tree-realization-v1"
            or self.source_contract != _POPULATION_SOURCES.get(self.population)
            or self.backend_semantics != "sklearn-independent-output-r0-v1"
            or type(self.plan) is not PopulationTreePlan
            or type(self.function) is not FrozenFittedTreeFunction
            or self.population not in ("support-only", "batch-refit-reference", "mixed-independent")
            or (self.mixed is not None) != (self.population == "mixed-independent")
            or self.mixed is not None
            and type(self.mixed) is not MixedFitPopulation
            or matrix_shape(self.inputs) != (self.plan.rows, self.plan.input_width)
            or matrix_shape(self.targets) != (self.plan.rows, self.plan.output_width)
            or (self.function.input_width, self.function.output_width)
            != (self.plan.input_width, self.plan.output_width)
            or type(self.cardinalities) is not tuple
            or len(self.cardinalities) != self.plan.input_width
            or type(self.backend_version) is not str
            or not self.backend_version
            or any(
                type(v) is not tuple
                for v in (
                    self.backend_seeds,
                    self.backend_parameters,
                    self.initial_rng,
                    self.final_rng,
                    self.raw_draws,
                )
            )
            or len(self.backend_seeds) != self.plan.output_width
            or len(self.backend_parameters) != self.plan.output_width
            or any(type(v) is not int or not 0 <= v < 2**31 for v in self.backend_seeds)
            or any(type(v) is not str for v in self.backend_parameters)
            or len(self.initial_rng) != 3
            or len(self.final_rng) != 3
            or any(type(v) is not GeneratorSnapshot for v in (*self.initial_rng, *self.final_rng))
            or any(type(v) is not RawDrawRecord for v in self.raw_draws)
        ):
            raise ValueError("population fit geometry, version or provenance differs")
        streams: dict[str, np.random.Generator] = dict(
            zip(
                ("input", "target", "backend"), (s.restore() for s in self.initial_rng), strict=True
            )
        )
        ledger = RawDrawLedger()
        with ledger.transactions(streams) as (staged, capture):
            x, y, seeds, cardinalities = population_draws(
                self.plan,
                self.mixed,
                None if self.mixed else np.asarray(self.inputs, dtype=np.float64),
                staged,
                capture,
            )
        if (
            frozen_matrix(x) != self.inputs
            or frozen_matrix(y) != self.targets
            or seeds != self.backend_seeds
            or cardinalities != self.cardinalities
            or ledger.freeze() != self.raw_draws
            or tuple(snapshot_generator(r) for r in streams.values()) != self.final_rng
        ):
            raise ValueError("population fit differs from retained raw draw custody")
        validate_backend_fit(
            self.plan, self.function, self.backend_seeds, self.backend_parameters, self.targets
        )


def admit_population_fit(plan: PopulationTreePlan, limits: CausalLimits) -> None:
    plan.admit(limits)
    # Both numerical tables, Python raw/derived copies, worst-case class masses and
    # three initial/final engine snapshots are admitted before allocating tables.
    cells = plan.rows * (plan.input_width + plan.output_width)
    limits.admit_dimensions(
        (plan.input_width, plan.output_width),
        ((), ()),
        rows=plan.rows,
        calibration_rows=plan.rows,
        features=plan.input_width,
        outputs=plan.output_width,
        extra_parameters=8 * cells + 6 * 700 + 400 * (plan.input_width + plan.output_width),
        extra_working_channels=8 * (plan.input_width + plan.output_width + plan.classes),
    )


def fit_population_tree(
    plan: PopulationTreePlan,
    *,
    population: str,
    input_rng: np.random.Generator,
    target_rng: np.random.Generator,
    backend_rng: np.random.Generator,
    inputs: NDArray[np.float64] | None = None,
    mixed: MixedFitPopulation | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> PopulationTreeFit:
    """Fit exactly the selected synthetic rows; never pad, clip, retry or add classes."""
    if type(plan) is not PopulationTreePlan or type(limits) is not CausalLimits:
        raise ValueError("population fitting requires its plan and caller limits")
    if population not in ("support-only", "batch-refit-reference", "mixed-independent"):
        raise ValueError("unknown fitting lifecycle")
    if (mixed is not None) != (population == "mixed-independent"):
        raise ValueError("mixed population and lifecycle differ")
    if mixed is not None and type(mixed) is not MixedFitPopulation:
        raise ValueError("mixed fitting requires its explicit population settings")
    if (
        (inputs is None) != (mixed is not None)
        or inputs is not None
        and (
            type(inputs) is not np.ndarray
            or inputs.dtype != np.float64
            or inputs.shape != (plan.rows, plan.input_width)
            or not np.isfinite(inputs).all()
        )
    ):
        raise ValueError("selected fitting inputs have invalid shape or values")
    admit_population_fit(plan, limits)
    ledger = RawDrawLedger()
    with ledger.transactions(
        {"input": input_rng, "target": target_rng, "backend": backend_rng}
    ) as (streams, capture):
        initial = tuple(snapshot_generator(r) for r in streams.values())
        x, y, seeds, cardinalities = population_draws(plan, mixed, inputs, streams, capture)
        if (
            plan.family == "gbdt"
            and plan.task == "classification"
            and any(len(np.unique(y[:, column])) < 2 for column in range(plan.output_width))
        ):
            raise ValueError("boosted classifier requires two observed classes; no retry")
        outputs, parameters, version = fit_backend(plan, x, y, seeds)
        result = PopulationTreeFit(
            plan,
            FrozenFittedTreeFunction(plan.input_width, outputs),
            frozen_matrix(x),
            frozen_matrix(y),
            population,
            mixed,
            cardinalities,
            seeds,
            parameters,
            version,
            initial,
            tuple(snapshot_generator(r) for r in streams.values()),
            capture.freeze(),
            source_contract=_POPULATION_SOURCES[population],
        )
        admit_function_metadata(asdict(result), limits)
        return result
