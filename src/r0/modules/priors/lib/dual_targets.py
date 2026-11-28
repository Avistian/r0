"""Named r0 dual-target mapping: continuous labels plus fixed-threshold categories."""

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.training_task import TrainingTask

if TYPE_CHECKING:
    from r0.modules.priors.lib.generate import Recipe


def validate_boundaries(boundaries: tuple[float, ...]) -> None:
    if (
        type(boundaries) is not tuple
        or not 1 <= len(boundaries) <= 99
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in boundaries)
        or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
    ):
        raise ValueError("dual-target boundaries must be 1..99 finite increasing cutpoints")


@dataclass(frozen=True, slots=True)
class DualTargetRecipe:
    base: "Recipe"
    boundaries: tuple[float, ...] = (0.0,)
    version: str = "dual-target-fixed-boundaries-r0-v1"

    def __post_init__(self) -> None:
        validate_boundaries(self.boundaries)
        task = getattr(self.base, "task", self.base)
        if (
            self.version != "dual-target-fixed-boundaries-r0-v1"
            or isinstance(self.base, DualTargetRecipe)
            or not isinstance(task, LinearRecipe)
            or task.task_type != "regression"
        ):
            raise ValueError("dual targets require an original regression generator")

    @property
    def task(self) -> LinearRecipe:
        base = getattr(self.base, "task", self.base)
        assert isinstance(base, LinearRecipe)
        return replace(
            base,
            task_type="binary" if len(self.boundaries) == 1 else "multiclass",
            classes=len(self.boundaries) + 1,
            noise_ratio=0.0,
            class_offsets=(),
            separation=1.0,
        )


@dataclass(frozen=True, slots=True)
class DualTargetTable:
    regression: TrainingTask
    classification: TrainingTask


def classification_view(recipe: DualTargetRecipe, base: TrainingTask) -> TrainingTask:
    if base.task_type != "regression" or not torch.isfinite(base.targets).all():
        raise ValueError("dual targets require finite continuous regression labels")
    boundaries = torch.tensor(
        recipe.boundaries, dtype=base.targets.dtype, device=base.targets.device
    )
    labels = torch.bucketize(base.targets, boundaries, right=True)
    classes = tuple(f"class-{i}" for i in range(len(recipe.boundaries) + 1))
    probabilities = torch.eye(len(classes), dtype=base.scores.dtype, device=base.inputs.device)[
        labels
    ]
    record = json.loads(base.recipe_json)
    record["recipe"] = asdict(recipe)
    record["execution"]["implementation"] = recipe.version
    identity = record_fingerprint((base.source_task_id, recipe.version, recipe.boundaries))
    return replace(
        base,
        source_task_id=identity,
        episode_id=record_fingerprint((identity, base.episode_id)),
        task_type=recipe.task.task_type,
        targets=labels,
        scores=probabilities[:, 1:] if len(classes) == 2 else probabilities,
        probabilities=probabilities,
        classes=classes,
        target_origin="adapter-v1",
        recipe_json=encode(record),
        mechanism_json=encode(
            {
                "version": recipe.version,
                "base": json.loads(base.mechanism_json),
                "regression_source_task_id": base.source_task_id,
                "boundaries": recipe.boundaries,
                "equality": "upper-class",
                "mapping": (
                    "fixed thresholds on emitted regression targets; no fitting or rebalancing"
                ),
                "source": "named r0 dual-target reconstruction; peer mapping undisclosed",
            }
        ),
    )


def generate_dual_targets(
    base: "Recipe",
    *,
    boundaries: tuple[float, ...] = (0.0,),
    seed: int,
    task_id: int,
    device: str = "cpu",
) -> DualTargetTable:
    from r0.modules.priors.lib.generate import generate_task

    recipe = DualTargetRecipe(base, boundaries)
    regression = generate_task(base, seed=seed, task_id=task_id, device=device)
    return DualTargetTable(regression, classification_view(recipe, regression))


def generate_dual_target(
    recipe: DualTargetRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    return generate_dual_targets(
        recipe.base, boundaries=recipe.boundaries, seed=seed, task_id=task_id, device=device
    ).classification


def decode_dual_target(data: dict[str, Any], *, limits: CausalLimits) -> DualTargetRecipe:
    from r0.modules.priors.lib.generate import recipe_from_dict

    if set(data) != {"base", "boundaries", "version"} or type(data["boundaries"]) not in (
        tuple,
        list,
    ):
        raise ValueError("dual-target recipe requires complete base and boundaries")
    boundaries = tuple(cast(list[float] | tuple[float, ...], data["boundaries"]))
    validate_boundaries(boundaries)
    return DualTargetRecipe(
        recipe_from_dict(data["base"], limits=limits), boundaries, data["version"]
    )
