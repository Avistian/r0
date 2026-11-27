"""Distinct ordered-population requests and realizations, never frozen row-wise functions."""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_forest_plan import (
    RecursiveForestPlan,
    recursive_effective_limits,
)
from r0.modules.priors.lib.recursive_forest_sampling import (
    SampledRecursiveForest,
    sample_recursive_forest,
)
from r0.modules.priors.lib.recursive_population_ancestors import (
    ANCESTOR_TYPES,
    PopulationAncestors,
    population_ancestor_values,
    population_state_widths,
)
from r0.modules.priors.lib.representatives import record_fingerprint


def population_identity(seed: int, task_id: int) -> None:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("population identity requires exact integers in [0, 2**63)")


def population_matrix(value: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(map(float, row)) for row in value.cpu().numpy())


@dataclass(frozen=True, slots=True)
class RecursivePopulationRequest:
    ancestors: PopulationAncestors
    parents: tuple[tuple[int, int], ...]
    plan: RecursiveForestPlan
    vocabulary: tuple[str, ...] = ()
    target_channel: int = 0
    limits: CausalLimits = CausalLimits()
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "recursive-population-request-v1"

    @property
    def task(self) -> LinearRecipe:
        return replace(
            self.ancestors.task,
            task_type="regression"
            if self.plan.task == "regression"
            else ("binary" if self.plan.classes == 2 else "multiclass"),
            classes=self.plan.classes,
            noise_ratio=0.0,
            separation=1.0,
            class_offsets=(),
        )

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-request-v1"
            or type(self.ancestors) not in ANCESTOR_TYPES
            or type(self.plan) is not RecursiveForestPlan
            or self.plan.lifecycle != "fresh-forest-per-row"
            or self.plan.populations != self.ancestors.task.rows
            or type(self.target_channel) is not int
            or not 0 <= self.target_channel < self.plan.output_width
            or type(self.vocabulary) is not tuple
            or any(type(v) is not str or not v or len(v) > 256 for v in self.vocabulary)
            or len(set(self.vocabulary)) != len(self.vocabulary)
            or len(self.vocabulary) != self.plan.classes
        ):
            raise ValueError(
                "population request requires explicit ordered per-row lifecycle and target"
            )
        widths = population_state_widths(self.ancestors)
        if (
            type(self.parents) is not tuple
            or len(self.parents) != self.plan.input_width
            or any(
                type(p) is not tuple
                or len(p) != 2
                or any(type(v) is not int for v in p)
                or not 0 <= p[0] < len(widths)
                or not 0 <= p[1] < widths[p[0]]
                for p in self.parents
            )
            or len(set(self.parents)) != len(self.parents)
        ):
            raise ValueError(
                "population parent slices must identify distinct encoded state channels"
            )
        self.admit()

    def admit(self, *, limits: CausalLimits | None = None) -> dict[str, int]:
        effective = recursive_effective_limits(self.limits, limits or self.limits)
        effective = recursive_effective_limits(effective, self.ancestors.limits)
        forest = self.plan.admit(effective, self.raw_limits)
        ancestor = self.ancestors.admit()
        numbers, storage = function_record_cost(self, effective)
        rows = self.plan.populations
        retained = rows * (self.plan.input_width + self.task.features + self.plan.output_width)
        parameters = forest["parameters"] + ancestor["parameters"] + numbers + 4 * retained
        working = (
            forest["estimated_working_bytes"]
            + ancestor["estimated_working_bytes"]
            + 6 * storage
            + 1024 * retained
            + rows * 2048 * self.plan.estimators
        )
        if (
            parameters > effective.parameters
            or working > effective.working_bytes
            or max(population_state_widths(self.ancestors)) > effective.node_channels
            or sum(population_state_widths(self.ancestors)) + self.plan.output_width
            > effective.total_channels
            or forest["nodes"] + len(self.ancestors.nodes) > effective.nodes
        ):
            raise ValueError(
                "aggregate population construction, retained rows or replay exceeds limits"
            )
        return {"parameters": parameters, "estimated_working_bytes": working}


@dataclass(frozen=True, slots=True)
class RecursivePopulationRecipe:
    request: RecursivePopulationRequest
    sample: SampledRecursiveForest
    parent_rows: tuple[tuple[float, ...], ...]
    observed_rows: tuple[tuple[float, ...], ...]
    seed: int
    task_id: int
    version: str = "recursive-population-v1"

    @property
    def task(self) -> LinearRecipe:
        return self.request.task

    @property
    def limits(self) -> CausalLimits:
        return self.request.limits

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-v1"
            or type(self.request) is not RecursivePopulationRequest
        ):
            raise ValueError("population realization requires its explicit request")
        population_identity(self.seed, self.task_id)
        self.validate_replay()

    def validate_replay(self, *, limits: CausalLimits | None = None) -> None:
        self.request.admit(limits=limits)
        function_record_cost(self, limits or self.limits)
        if (
            type(self.sample) is not SampledRecursiveForest
            or not exact_frozen_equal(self.sample.plan, self.request.plan)
            or not exact_frozen_equal(self.sample.limits, self.request.limits)
            or not exact_frozen_equal(self.sample.raw_limits, self.request.raw_limits)
        ):
            raise ValueError("population forest receipt differs from request")
        for matrix, width in (
            (self.parent_rows, self.request.plan.input_width),
            (self.observed_rows, self.task.features),
        ):
            if (
                type(matrix) is not tuple
                or len(matrix) != self.task.rows
                or any(type(row) is not tuple or len(row) != width for row in matrix)
                or any(type(v) is not float or not np.isfinite(v) for row in matrix for v in row)
            ):
                raise ValueError("population rows require exact finite float matrices")
        self.sample.validate_replay(limits=limits or self.limits)
        values = population_ancestor_values(
            self.request.ancestors, seed=self.seed, task_id=self.task_id
        )
        parents = torch.stack([values.states[i][:, j] for i, j in self.request.parents], 1)
        if (
            population_matrix(parents) != self.parent_rows
            or population_matrix(values.observed) != self.observed_rows
        ):
            raise ValueError("population rows differ from original ordered ancestor realization")


def realize_recursive_population(
    request: RecursivePopulationRequest,
    *,
    seed: int,
    task_id: int,
    rng: np.random.Generator | None = None,
) -> RecursivePopulationRecipe:
    """A new realization is explicit; frozen generation never changes its row ownership."""
    if type(request) is not RecursivePopulationRequest:
        raise ValueError("population construction requires its request")
    population_identity(seed, task_id)
    if rng is not None and type(rng) is not np.random.Generator:
        raise ValueError("population construction requires an explicit NumPy generator")
    request.admit()
    values = population_ancestor_values(request.ancestors, seed=seed, task_id=task_id)
    parents = torch.stack([values.states[i][:, j] for i, j in request.parents], 1)
    owned = (
        np.random.default_rng(
            int(
                record_fingerprint(
                    (asdict(request), seed, task_id, "recursive-population-construction-v1")
                ),
                16,
            )
        )
        if rng is None
        else rng
    )
    from r0.modules.priors.lib.generator_snapshot import snapshot_generator

    initial = snapshot_generator(owned, byte_limit=request.raw_limits.state_bytes)
    stream = initial.restore(byte_limit=request.raw_limits.state_bytes)
    sample = sample_recursive_forest(
        request.plan, rng=stream, limits=request.limits, raw_limits=request.raw_limits
    )
    result = RecursivePopulationRecipe(
        request,
        sample,
        population_matrix(parents),
        population_matrix(values.observed),
        seed,
        task_id,
    )
    if snapshot_generator(owned, byte_limit=request.raw_limits.state_bytes) != initial:
        raise ValueError("caller population construction stream changed during realization")
    owned.bit_generator.state = stream.bit_generator.state
    return result
