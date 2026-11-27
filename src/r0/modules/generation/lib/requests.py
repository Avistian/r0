"""Run requests and deterministic per-task recipe selection."""

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.priors import (
    CausalLimits,
    LinearRecipe,
    Recipe,
    TrainingTask,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)


@dataclass(frozen=True, slots=True)
class RecipeSampling:
    family: str
    profile: str = "broad-v1"

    def __post_init__(self) -> None:
        if type(self.family) is not str or type(self.profile) is not str:
            raise ValueError("sampling family and profile must be strings")


@dataclass(frozen=True, slots=True)
class RunRequest:
    recipe: Recipe
    seed: int
    count: int
    max_seconds: float
    max_bytes: int
    start: int = 0
    group_size: int = 64
    device: str = "cpu"
    sampling: RecipeSampling | None = None
    causal_limits: CausalLimits | None = None
    construction: GraphConstruction | None = None

    def __post_init__(self) -> None:
        if self.causal_limits is not None and type(self.causal_limits) is not CausalLimits:
            raise ValueError("run exploration limits require caller-owned CausalLimits")
        if any(
            type(v) is not int
            for v in (self.seed, self.count, self.start, self.group_size, self.max_bytes)
        ):
            raise ValueError("run counts, identities and storage limits must be integers")
        if not (0 <= self.seed < 2**63 and 0 <= self.start < self.start + self.count <= 2**63):
            raise ValueError("invalid seed or task interval")
        if not (1 <= self.count <= 100_000_000 and 1 <= self.group_size <= 1024):
            raise ValueError("count must be 1..100M and group_size 1..1024")
        if self.max_bytes < 65_536 or self.device not in ("cpu", "cuda"):
            raise ValueError("at least 64 KiB storage and an explicit CPU/CUDA device are required")
        if type(self.max_seconds) not in (int, float) or not (
            math.isfinite(self.max_seconds) and self.max_seconds > 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.construction is not None:
            if type(self.construction) is not GraphConstruction or self.sampling is not None:
                raise ValueError("graph construction cannot also use recipe sampling")
            if json.loads(json.dumps(self.construction.task_record())) != json.loads(
                json.dumps(asdict(self.recipe))
            ):
                raise ValueError("construction task controls differ from the run recipe")
        if self.sampling is not None:
            if type(self.sampling) is not RecipeSampling or not isinstance(
                self.recipe, LinearRecipe
            ):
                raise ValueError("sampling needs RecipeSampling and LinearRecipe task controls")
            # Validate family/profile and fixed geometry before creating a run directory.
            selected_recipe(self, self.start)


def selected_recipe(request: RunRequest, task_id: int) -> Recipe:
    if request.construction is not None:
        return request.construction.build(request.causal_limits or CausalLimits())
    if request.sampling is None:
        return request.recipe
    assert isinstance(request.recipe, LinearRecipe)
    return sample_recipe(
        request.sampling.family,
        profile=request.sampling.profile,
        task=request.recipe,
        seed=request.seed,
        task_id=task_id,
    )


def sampling_record(request: RunRequest) -> dict[str, Any] | None:
    if request.sampling is None:
        return None
    return json.loads(json.dumps({**asdict(request.sampling), "task": asdict(request.recipe)}))


def generate_requested(request: RunRequest, task_id: int) -> TrainingTask:
    task = generate_task(
        selected_recipe(request, task_id), seed=request.seed, task_id=task_id, device=request.device
    )
    sampling = sampling_record(request)
    if sampling is None:
        return task
    record = {**json.loads(task.recipe_json), "sampling": sampling}
    return replace(
        task, recipe_json=json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def record_matches(record: dict[str, Any], request: RunRequest, task_id: int) -> bool:
    return (
        record["recipe"] == json.loads(json.dumps(asdict(selected_recipe(request, task_id))))
        and record.get("sampling") == sampling_record(request)
        and record["seed"] == request.seed
        and record["task_id"] == task_id
    )


def request_from_dict(data: dict[str, Any]) -> RunRequest:
    fields = dict(data)
    raw_limits = fields.pop("causal_limits", None)
    limits = None if raw_limits is None else CausalLimits(**raw_limits)
    recipe = recipe_from_dict(fields.pop("recipe"), limits=limits or CausalLimits())
    sampling = fields.pop("sampling", None)
    construction = fields.pop("construction", None)
    return RunRequest(
        recipe,
        sampling=None if sampling is None else RecipeSampling(**sampling),
        causal_limits=limits,
        construction=None if construction is None else GraphConstruction(**construction),
        **fields,
    )
