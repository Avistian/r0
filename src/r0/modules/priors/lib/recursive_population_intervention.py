"""Explicit matched versus new-population comparisons for row-owned terminal forests."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.recursive_forest_evaluation import evaluate_recursive_forest
from r0.modules.priors.lib.recursive_forest_sampling import (
    SampledRecursiveForest,
    sample_recursive_forest,
)
from r0.modules.priors.lib.recursive_population_ancestors import (
    PopulationInputs,
    population_ancestor_values,
)
from r0.modules.priors.lib.recursive_population_output import population_output
from r0.modules.priors.lib.recursive_population_records import (
    RecursivePopulationRecipe,
    population_matrix,
)
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask


def population_intervention_seed(
    base: RecursivePopulationRecipe, population: int, stream: str
) -> int:
    return (
        int(
            record_fingerprint(
                (
                    "recursive-population-intervention-v1",
                    asdict(base.sample.initial),
                    base.seed,
                    base.task_id,
                    population,
                    stream,
                )
            ),
            16,
        )
        % 2**63
    )


def admit_population_intervention(
    base: RecursivePopulationRecipe, ancestors: CausalInterventionRecipe | None
) -> None:
    from r0.modules.priors.lib.joint_function_admission import function_record_cost

    report = base.request.admit()
    numbers, storage = function_record_cost((base, ancestors, base.sample), base.limits)
    extra = ancestors.admit() if ancestors else {"parameters": 0, "estimated_working_bytes": 0}
    retained = base.task.rows * (
        base.request.plan.input_width + base.task.features + base.request.plan.output_width
    )
    if (
        report["estimated_working_bytes"]
        + 8 * storage
        + 2048 * retained
        + extra["estimated_working_bytes"]
        > base.limits.working_bytes
        or report["parameters"] + numbers + 4 * retained + extra["parameters"]
        > base.limits.parameters
    ):
        raise ValueError("population intervention aggregate custody exceeds limits")


@dataclass(frozen=True, slots=True)
class RecursivePopulationIntervention:
    base: RecursivePopulationRecipe
    ancestors: CausalInterventionRecipe | None
    mode: str
    population: int
    sample: SampledRecursiveForest
    parent_rows: tuple[tuple[float, ...], ...]
    observed_rows: tuple[tuple[float, ...], ...]
    forced_value: float | None = None
    forced_category: str | None = None
    version: str = "recursive-population-intervention-v1"

    @property
    def task(self):
        return self.base.task

    @property
    def limits(self):
        return self.base.limits

    def __post_init__(self) -> None:
        if (
            self.version != "recursive-population-intervention-v1"
            or type(self.base) is not RecursivePopulationRecipe
            or self.ancestors is not None
            and type(self.ancestors) is not CausalInterventionRecipe
            or self.mode not in ("matched", "population")
            or type(self.population) is not int
            or not 1 <= self.population < 2**63
            or self.forced_value is not None
            and (
                type(self.forced_value) is not float
                or not np.isfinite(self.forced_value)
                or self.base.request.plan.classes != 0
            )
            or self.forced_category is not None
            and (
                type(self.forced_category) is not str
                or self.forced_category not in self.base.request.vocabulary
            )
            or self.ancestors is None
            and self.forced_value is None
            and self.forced_category is None
            and self.mode == "matched"
        ):
            raise ValueError(
                "population intervention requires explicit changes and population policy"
            )
        self.validate_replay()

    def values(self) -> PopulationInputs:
        seed = (
            self.base.seed
            if self.mode == "matched"
            else population_intervention_seed(self.base, self.population, "ancestors")
        )
        return population_ancestor_values(
            self.base.request.ancestors,
            seed=seed,
            task_id=self.base.task_id,
            intervention=self.ancestors,
        )

    def validate_replay(self) -> None:
        from r0.modules.priors.lib.joint_function_admission import function_record_cost

        function_record_cost(self, self.limits)
        admit_population_intervention(self.base, self.ancestors)
        self.base.validate_replay()
        if self.ancestors is not None and (
            type(self.ancestors) is not CausalInterventionRecipe
            or self.ancestors.mode != "matched"
            or not exact_frozen_equal(self.ancestors.base, self.base.request.ancestors)
        ):
            raise ValueError(
                "ancestor intervention must bind the exact original graph in matched mode"
            )
        if (
            type(self.sample) is not SampledRecursiveForest
            or not exact_frozen_equal(self.sample.plan, self.base.sample.plan)
            or not exact_frozen_equal(self.sample.limits, self.base.sample.limits)
            or not exact_frozen_equal(self.sample.raw_limits, self.base.sample.raw_limits)
        ):
            raise ValueError("intervention forest plan differs")
        if self.mode == "matched":
            if not exact_frozen_equal(self.sample, self.base.sample):
                raise ValueError("matched intervention must retain every row-owned forest")
        else:
            initial = snapshot_generator(
                np.random.default_rng(
                    population_intervention_seed(self.base, self.population, "forests")
                ),
                byte_limit=self.base.request.raw_limits.state_bytes,
            )
            if not exact_frozen_equal(initial, self.sample.initial):
                raise ValueError(
                    "population forest stream differs from explicit population identity"
                )
        self.sample.validate_replay(limits=self.limits)
        values = self.values()
        parents = torch.stack([values.states[i][:, j] for i, j in self.base.request.parents], 1)
        if not exact_frozen_equal(
            population_matrix(parents), self.parent_rows
        ) or not exact_frozen_equal(population_matrix(values.observed), self.observed_rows):
            raise ValueError("intervened original-row custody differs")


def intervene_recursive_population(
    base: RecursivePopulationRecipe,
    *,
    ancestors: CausalInterventionRecipe | None = None,
    mode: str = "matched",
    population: int = 1,
    forced_value: float | None = None,
    forced_category: str | None = None,
) -> RecursivePopulationIntervention:
    if (
        type(base) is not RecursivePopulationRecipe
        or mode not in ("matched", "population")
        or (type(population) is not int or not 1 <= population < 2**63)
    ):
        raise ValueError("invalid explicit population intervention")
    base.request.admit()
    if ancestors is not None and (
        type(ancestors) is not CausalInterventionRecipe
        or ancestors.mode != "matched"
        or not exact_frozen_equal(ancestors.base, base.request.ancestors)
    ):
        raise ValueError("ancestor intervention differs from original graph")
    admit_population_intervention(base, ancestors)
    values = population_ancestor_values(
        base.request.ancestors,
        seed=base.seed
        if mode == "matched"
        else population_intervention_seed(base, population, "ancestors"),
        task_id=base.task_id,
        intervention=ancestors,
    )
    parents = torch.stack([values.states[i][:, j] for i, j in base.request.parents], 1)
    sample = (
        base.sample
        if mode == "matched"
        else sample_recursive_forest(
            base.request.plan,
            rng=np.random.default_rng(population_intervention_seed(base, population, "forests")),
            limits=base.limits,
            raw_limits=base.request.raw_limits,
        )
    )
    return RecursivePopulationIntervention(
        base,
        ancestors,
        mode,
        population,
        sample,
        population_matrix(parents),
        population_matrix(values.observed),
        forced_value,
        forced_category,
    )


def generate_population_intervention(
    recipe: RecursivePopulationIntervention, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    from r0.modules.priors.lib.recursive_population_records import population_identity

    population_identity(seed, task_id)
    if device != "cpu" or (seed, task_id) != (recipe.base.seed, recipe.base.task_id):
        raise ValueError(
            "intervened population retains original identity and explicit CPU execution"
        )
    recipe.validate_replay()
    request = recipe.base.request
    if recipe.forced_category is not None:
        emitted = np.full(
            request.task.rows, request.vocabulary.index(recipe.forced_category), dtype=np.int64
        )
    elif recipe.forced_value is not None and request.plan.output_width == 1:
        emitted = np.full((request.task.rows, 1), recipe.forced_value, dtype=np.float64)
    else:
        emitted = evaluate_recursive_forest(
            recipe.sample, np.asarray(recipe.parent_rows, dtype=np.float64), limits=recipe.limits
        )
        if recipe.forced_value is not None:
            emitted[:, request.target_channel] = recipe.forced_value
    metadata: dict[str, Any] = {
        "mode": recipe.mode,
        "population": recipe.population,
        "forest_ownership": "retained original forests"
        if recipe.mode == "matched"
        else "fresh keyed population",
        "forced_value": recipe.forced_value,
        "forced_category": recipe.forced_category,
    }
    values = recipe.values()
    return population_output(
        request,
        asdict(recipe),
        values.observed,
        values.schema,
        emitted,
        seed=seed,
        task_id=task_id,
        intervention=metadata,
    )


def generate_recursive_population_pair(
    base: RecursivePopulationRecipe,
    *,
    ancestors: CausalInterventionRecipe | None = None,
    mode: str = "matched",
    population: int = 1,
    forced_value: float | None = None,
    forced_category: str | None = None,
):
    """Keep baseline bytes unchanged and record explicit row-wise population ownership."""
    from r0.modules.priors.lib.pairs import TaskPair
    from r0.modules.priors.lib.recursive_population_output import generate_recursive_population
    from r0.modules.priors.lib.representatives import encode

    changed = intervene_recursive_population(
        base,
        ancestors=ancestors,
        mode=mode,
        population=population,
        forced_value=forced_value,
        forced_category=forced_category,
    )
    before = generate_recursive_population(base, seed=base.seed, task_id=base.task_id)
    after = generate_population_intervention(changed, seed=base.seed, task_id=base.task_id)
    return TaskPair(
        before,
        after,
        encode(
            {
                "version": "recursive-population-pair-v1",
                "mode": mode,
                "population": population,
                "before_episode": before.episode_id,
                "after_episode": after.episode_id,
                "base_archive_unchanged": asdict(changed.base) == asdict(base),
                "forests_equal": exact_frozen_equal(base.sample, changed.sample),
                "parent_rows_equal": base.parent_rows == changed.parent_rows,
                "inputs_equal": bool(torch.equal(before.inputs, after.inputs)),
                "targets_equal": bool(torch.equal(before.targets, after.targets)),
                "split_equal": bool(torch.equal(before.context_indices, after.context_indices)),
                "variance_semantics": (
                    "raw population outputs; no variance matching or query calibration"
                ),
            }
        ),
    )
