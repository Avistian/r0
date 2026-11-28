"""Run requests and deterministic per-task recipe selection."""

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.priors import (
    CalibratedPopulationGraph,
    CausalLimits,
    LinearRecipe,
    Recipe,
    TrainingTask,
    generate_task,
    preflight_calibrated_generation,
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
        if isinstance(self.recipe, CalibratedPopulationGraph):
            preflight_calibrated_generation(
                self.recipe,
                seed=self.seed,
                task_id=self.start,
                device=self.device,
                count=self.count,
            )
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
        return request.construction.build(
            request.causal_limits or CausalLimits(), episode_seed=request.seed, task_id=task_id
        )
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
    from r0.modules.priors import (
        PrototypeBatchNetworkRecipe,
        RealizedPrototypeBatchNetworkRecipe,
    )

    selected = selected_recipe(request, task_id)
    from r0.modules.priors import GPColumnPopulation, GPColumnRecipe, GPColumnTargetRecipe

    if isinstance(selected, GPColumnTargetRecipe):
        if any(
            type(record.get(k)) is not int or record[k] != v
            for k, v in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        if json.dumps(record.get("recipe"), sort_keys=True, allow_nan=False) != json.dumps(
            asdict(selected), sort_keys=True, allow_nan=False
        ) or record.get("sampling") != sampling_record(request):
            return False
        recipe_from_dict(record["recipe"], limits=request.causal_limits or selected.request.limits)
        return True
    if isinstance(selected, GPColumnRecipe):
        if any(
            type(record.get(key)) is not int or record[key] != identity
            for key, identity in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        raw = record.get("recipe")
        if type(raw) is not dict:
            return False
        raw = cast(dict[str, Any], raw)
        source = raw.get("source")
        if type(source) is not dict:
            return False
        source = cast(dict[str, Any], source)
        if (
            raw.get("version") != selected.version
            or source.get("version") != "gp-column-population-v1"
            or any(
                type(source.get(key)) is not int or source[key] != identity
                for key, identity in (("seed", request.seed), ("task_id", task_id))
            )
            or record.get("sampling") != sampling_record(request)
        ):
            return False
        # These controls are known without any numerical population replay.
        actual_controls = {"task": raw.get("task"), "request": source.get("request")}
        expected_controls = {"task": asdict(selected.task), "request": asdict(selected.request)}
        if json.dumps(actual_controls, sort_keys=True, allow_nan=False) != json.dumps(
            expected_controls, sort_keys=True, allow_nan=False
        ):
            return False
        if isinstance(selected.source, GPColumnPopulation) and json.dumps(
            raw, sort_keys=True, allow_nan=False
        ) != json.dumps(asdict(selected), sort_keys=True, allow_nan=False):
            return False
        realized = recipe_from_dict(raw, limits=request.causal_limits or selected.request.limits)
        if not isinstance(realized, GPColumnRecipe) or not isinstance(
            realized.source, GPColumnPopulation
        ):
            return False
        if (realized.source.seed, realized.source.task_id) != (request.seed, task_id):
            return False
        expected = (
            selected
            if isinstance(selected.source, GPColumnPopulation)
            else replace(selected, source=realized.source)
        )
        return (
            json.dumps(asdict(realized), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(expected), sort_keys=True, allow_nan=False)
            and json.dumps(asdict(realized.request), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(selected.request), sort_keys=True, allow_nan=False)
            and record.get("sampling") == sampling_record(request)
        )
    if isinstance(selected, (PrototypeBatchNetworkRecipe, RealizedPrototypeBatchNetworkRecipe)):
        # Python numeric equality aliases bool/float identities to integers. Check
        # transport primitives before replay so a persisted record can replay itself.
        raw = record.get("recipe")
        if type(raw) is not dict:
            return False
        raw = cast(dict[str, Any], raw)
        if any(
            type(record.get(key)) is not int
            or type(raw.get(key)) is not int
            or record[key] != expected_id
            or raw[key] != expected_id
            for key, expected_id in (("seed", request.seed), ("task_id", task_id))
        ):
            return False
        realized = recipe_from_dict(
            record["recipe"], limits=request.causal_limits or selected.limits
        )
        expected = (
            selected.request
            if isinstance(selected, RealizedPrototypeBatchNetworkRecipe)
            else selected
        )
        return (
            isinstance(realized, RealizedPrototypeBatchNetworkRecipe)
            and json.dumps(raw.get("request"), sort_keys=True, allow_nan=False)
            == json.dumps(asdict(expected), sort_keys=True, allow_nan=False)
            and (realized.seed, realized.task_id) == (request.seed, task_id)
            and record.get("sampling") == sampling_record(request)
            and record["seed"] == request.seed
            and record["task_id"] == task_id
        )
    return (
        record["recipe"] == json.loads(json.dumps(asdict(selected)))
        and record.get("sampling") == sampling_record(request)
        and record["seed"] == request.seed
        and record["task_id"] == task_id
    )


def request_from_dict(data: dict[str, Any]) -> RunRequest:
    fields = dict(data)
    candidate = fields.get("recipe")
    if (
        type(candidate) is dict
        and cast(dict[str, Any], candidate).get("version") == "calibrated-population-graph-v1"
        and any(fields.get(key) is not None for key in ("construction", "sampling"))
    ):
        raise ValueError("fixed calibrated population cannot use construction or sampling")
    raw_limits = fields.pop("causal_limits", None)
    limits = None if raw_limits is None else CausalLimits(**raw_limits)
    raw_construction = fields.pop("construction", None)
    construction = None if raw_construction is None else GraphConstruction(**raw_construction)
    # Construction-only transport derives its task inside the guarded worker.
    # Explicit recipe controls still take precedence and must pass RunRequest's
    # geometry binding check, so contradictory controls cannot be ignored.
    raw_recipe = (
        construction.task_record()
        if "recipe" not in fields and construction is not None
        else fields.pop("recipe", None)
    )
    if (
        type(raw_recipe) is dict
        and cast(dict[str, Any], raw_recipe).get("version") == "calibrated-population-graph-v1"
    ):
        raw_recipe = cast(dict[str, Any], raw_recipe)
        preflight_calibrated_generation(
            raw_recipe,
            seed=cast(int, fields.get("seed")),
            task_id=fields.get("start", 0),
            device=fields.get("device", "cpu"),
            count=cast(int, fields.get("count")),
        )
        if construction is not None or fields.get("sampling") is not None:
            raise ValueError("fixed calibrated population cannot use construction or sampling")
        try:
            task_fields = dict(raw_recipe["original"]["request"]["base"]["task"])
            if type(task_fields["class_offsets"]) not in (tuple, list):
                raise ValueError("class_offsets must be an array")
            task_fields["class_offsets"] = tuple(task_fields["class_offsets"])
            preview = LinearRecipe(**task_fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid calibrated population task metadata") from exc
        # Validate the complete outer RunRequest schema/budgets against cheap task
        # controls, without decoding either retained population. Discard preview.
        RunRequest(preview, causal_limits=limits, **fields)
    recipe = recipe_from_dict(cast(Any, raw_recipe), limits=limits or CausalLimits())
    sampling = fields.pop("sampling", None)
    return RunRequest(
        recipe,
        sampling=None if sampling is None else RecipeSampling(**sampling),
        causal_limits=limits,
        construction=construction,
        **fields,
    )
