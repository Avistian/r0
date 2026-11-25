"""Controlled recipe changes with explicit checks of the generated task pair."""

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from r0.modules.priors.lib.compositions import CompositeRecipe
from r0.modules.priors.lib.generate import Recipe, generate_task, recipe_from_dict
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask

_FIELDS = {
    "noise": "noise_ratio",
    "irrelevant": "features",
    "separation": "separation",
    "sparsity": "active_features",
}


@dataclass(frozen=True, slots=True)
class TaskPair:
    before: TrainingTask
    after: TrainingTask
    card_json: str


def paired_recipe(recipe: Recipe, *, intervention: str, value: int | float) -> Recipe:
    """Construct one explicit change; preserve all other recipe controls."""
    if intervention not in _FIELDS:
        raise ValueError("unsupported pair intervention")
    controls = recipe if isinstance(recipe, LinearRecipe) else recipe.task
    field = _FIELDS[intervention]
    if value == getattr(controls, field):
        raise ValueError("pair intervention must change the selected value")
    changed = replace(controls, **{field: value})
    if intervention == "sparsity" and isinstance(recipe, CompositeRecipe):
        components = tuple(
            component
            if component.axes
            else replace(
                component,
                recipe=paired_recipe(component.recipe, intervention="sparsity", value=value),
            )
            for component in recipe.components
        )
        return replace(recipe, task=changed, components=components)
    return changed if isinstance(recipe, LinearRecipe) else replace(recipe, task=changed)


def generate_pair(
    recipe: Recipe,
    *,
    intervention: str,
    value: int | float,
    seed: int,
    task_id: int,
    device: str = "cpu",
    allow_estimated_variance: bool = False,
) -> TaskPair:
    """Generate two coupled variants; neither failures nor changed properties are hidden."""
    _variance_control(recipe, intervention, allow_estimated_variance)
    changed = paired_recipe(recipe, intervention=intervention, value=value)
    before = generate_task(recipe, seed=seed, task_id=task_id, device=device)
    after = generate_task(changed, seed=seed, task_id=task_id, device=device)
    return TaskPair(
        before,
        after,
        encode(
            compare_pair(
                before,
                after,
                intervention=intervention,
                allow_estimated_variance=allow_estimated_variance,
            )
        ),
    )


def compare_pair(
    before: TrainingTask,
    after: TrainingTask,
    *,
    intervention: str,
    allow_estimated_variance: bool = False,
) -> dict[str, Any]:
    """Verify a saved pair through the same interface used by local/notebook inspection."""
    records = [json.loads(task.recipe_json) for task in (before, after)]
    recipes = [recipe_from_dict(record["recipe"]) for record in records]
    controls = [recipe if isinstance(recipe, LinearRecipe) else recipe.task for recipe in recipes]
    if intervention not in _FIELDS:
        raise ValueError("unsupported pair intervention")
    variance_control = _variance_control(recipes[0], intervention, allow_estimated_variance)
    field = _FIELDS[intervention]
    expected = paired_recipe(
        recipes[0], intervention=intervention, value=getattr(controls[1], field)
    )
    if asdict(expected) != asdict(recipes[1]):
        raise ValueError("pair changes recipe properties outside the declared intervention")
    common = min(before.inputs.shape[1], after.inputs.shape[1])
    equalities = {
        "seed": records[0]["seed"] == records[1]["seed"],
        "task_id": records[0]["task_id"] == records[1]["task_id"],
        "execution": records[0]["execution"] == records[1]["execution"],
        "common_inputs": torch.equal(before.inputs[:, :common], after.inputs[:, :common]),
        "scores": torch.equal(before.scores, after.scores),
        "mechanism": before.mechanism_json == after.mechanism_json,
        "context_indices": torch.equal(before.context_indices, after.context_indices),
        "query_indices": torch.equal(before.query_indices, after.query_indices),
        "classes": before.classes == after.classes,
    }
    if intervention == "irrelevant":
        equalities["targets"] = torch.equal(before.targets, after.targets)
        equalities["probabilities"] = (
            before.probabilities is None and after.probabilities is None
        ) or (
            before.probabilities is not None
            and after.probabilities is not None
            and torch.equal(before.probabilities, after.probabilities)
        )
    required = dict(equalities)
    if intervention == "sparsity":
        required.pop("scores")
        required.pop("mechanism")
        active = [set(json.loads(task.mechanism_json)["active"]) for task in (before, after)]
        required["nested_active_features"] = active[0] < active[1] or active[1] < active[0]
    failed = [name for name, same in required.items() if not same]
    if failed:
        raise ValueError(f"pair violated required equalities: {', '.join(failed)}")
    return {
        "version": "controlled-pair-v1",
        "pair_id": record_fingerprint(
            ("controlled-pair-v1", intervention, before.episode_id, after.episode_id)
        ),
        "intervention": intervention,
        "control": {
            "field": field,
            "before": getattr(controls[0], field),
            "after": getattr(controls[1], field),
        },
        "seed": records[0]["seed"],
        "task_id": records[0]["task_id"],
        "independent_source_draws": 1,
        "variant_count": 2,
        "changed_target_fraction": float(
            (before.targets != after.targets).to(torch.float64).mean()
        ),
        "required_equalities": required,
        "observed_equalities": equalities,
        "variance_control": variance_control,
        "allow_estimated_variance": allow_estimated_variance,
        "before": _measure(before),
        "after": _measure(after),
    }


def _variance_control(recipe: Recipe, intervention: str, allow_estimate: bool) -> str:
    if type(allow_estimate) is not bool:
        raise ValueError("allow_estimated_variance must be a boolean")
    if intervention != "sparsity":
        return "unchanged_mechanism"
    if isinstance(recipe, LinearRecipe):
        return "analytic_unit_variance"
    if not allow_estimate:
        raise ValueError(
            "exact expected-variance equality is available for Gaussian linear sparsity pairs; "
            "set allow_estimated_variance=True to request an explicitly calibration-normalized pair"
        )
    return "independent_calibration_unit_variance_estimate"


def _measure(task: TrainingTask) -> dict[str, Any]:
    scores = task.scores.detach().cpu().numpy()
    mechanism = json.loads(task.mechanism_json)
    result: dict[str, Any] = {
        "source_task_id": task.source_task_id,
        "recipe": json.loads(task.recipe_json),
        "mechanism_sha256": record_fingerprint(json.loads(task.mechanism_json)),
        "episode_id": task.episode_id,
        "rows": len(task.inputs),
        "features": task.inputs.shape[1],
        "active_features": len(mechanism["active"]),
        "normalization": mechanism.get(
            "normalization", "analytic unit variance under independent standard Gaussian inputs"
        ),
        "center": mechanism.get("center", [0.0] * scores.shape[1]),
        "scale": mechanism.get("scale", [1.0] * scores.shape[1]),
        "expected_signal_variance": mechanism.get("expected_signal_variance"),
        "signal_std": [_std(scores[:, channel]) for channel in range(scores.shape[1])],
    }
    if task.task_type == "regression":
        residual = task.targets.detach().cpu().numpy() - scores[:, 0]
        result["residual_std"] = _std(residual)
    else:
        if task.probabilities is None:
            raise ValueError("classification pair is missing probabilities")
        probabilities = task.probabilities.detach().cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(
                np.where(probabilities > 0, probabilities * np.log(probabilities), 0), axis=1
            )
        result["entropy_units"] = "nats"
        result["mean_entropy"] = float(entropy.mean())
        result["class_counts"] = np.bincount(
            task.targets.cpu().numpy(), minlength=len(task.classes)
        ).tolist()
        result["expected_class_counts"] = probabilities.sum(axis=0).tolist()
    return result


def _std(values: NDArray[np.float64]) -> float:
    magnitude = float(np.max(np.abs(values)))
    return 0.0 if magnitude == 0 else float(np.std(values / magnitude) * magnitude)
