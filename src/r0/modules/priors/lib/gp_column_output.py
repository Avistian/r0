"""Shared task adapters for a GP-feature population, separate from GP target draws."""

from dataclasses import asdict, dataclass, replace

import torch

from r0.modules.priors.lib.gp_column_population import (
    GPColumnPopulation,
    GPColumnRequest,
    gp_stream,
    realize_gp_columns,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class GPColumnRecipe:
    source: GPColumnRequest | GPColumnPopulation
    task: LinearRecipe
    version: str = "gp-column-recipe-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "gp-column-recipe-v1"
            or type(self.source) not in (GPColumnRequest, GPColumnPopulation)
            or type(self.task) is not LinearRecipe
        ):
            raise ValueError("invalid GP-column recipe")
        r = self.request
        if (
            (self.task.rows, self.task.features) != (r.rows, r.columns)
            or self.task.active_features != r.columns
            or self.task.signal_features != r.columns
            or r.outputs != (self.task.classes if self.task.task_type == "multiclass" else 1)
        ):
            raise ValueError(
                "GP population and adapter geometry disagree; all columns are structural"
            )
        r.admit()

    @property
    def request(self) -> GPColumnRequest:
        return self.source.request if isinstance(self.source, GPColumnPopulation) else self.source

    def with_task(self, task: LinearRecipe) -> "GPColumnRecipe":
        if task.rows != self.task.rows:
            return GPColumnRecipe(replace(self.request, rows=task.rows), task)
        return replace(self, task=task)


def generate_gp_columns(
    recipe: GPColumnRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    from r0.modules.priors.lib.gp_column_decoding import validate_gp_population

    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("GP generation requires exact integer identity")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")
    recipe.request.admit()
    if isinstance(recipe.source, GPColumnPopulation):
        population = recipe.source
        if (seed, task_id) != (population.seed, population.task_id):
            raise ValueError("realized GP population identity differs")
        validate_gp_population(population, limits=recipe.request.limits)
    else:
        population = realize_gp_columns(recipe.request, seed=seed, task_id=task_id)
    realized = replace(recipe, source=population)
    mechanism = {
        "family": "gp_columns",
        "version": "gp-column-population-v1",
        "population": asdict(population),
        "calibration": "independent single-row GP marginals; constant scale fallback=1",
        "target_law": "r0 scaled Gaussian weights/bias; full quadratic or tanh hidden layer",
        "lifecycle": "finite joint rows; not an independently evaluable GP function",
    }
    return complete_task(
        recipe.task,
        asdict(realized),
        mechanism,
        torch.tensor(population.draw.values, dtype=torch.float64, device=device),
        torch.tensor(population.scores, dtype=torch.float64, device=device),
        seed=seed,
        task_id=task_id,
        rng=lambda *role: gp_stream(seed, task_id, "adapter", *role),
    )
