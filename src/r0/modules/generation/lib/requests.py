"""Run requests and deterministic per-task recipe selection."""

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from r0.modules.generation.lib.construction import GraphConstruction
from r0.modules.generation.lib.gp_root_run_identity import validate_gp_root_run_identity
from r0.modules.priors import (
    CalibratedPopulationGraph,
    CalibratedPopulationIntervention,
    CalibratedPopulationQuery,
    CausalLimits,
    GPRootCalibration,
    GPRootCalibrationQuery,
    GPRootDAGRecipe,
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
        from r0.modules.generation.lib.xgboost_population_transport import (
            validate_xgboost_population_run,
        )

        validate_xgboost_population_run(self)
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
        if (
            isinstance(self.recipe, (GPRootCalibration, GPRootCalibrationQuery))
            and self.device != "cpu"
        ):
            raise ValueError("GP calibrated runs require CPU replay")
        if isinstance(self.recipe, (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery)):
            validate_gp_root_run_identity(
                asdict(self.recipe),
                seed=self.seed,
                start=self.start,
                count=self.count,
                sampling=self.sampling,
                construction=self.construction,
            )
        if isinstance(
            self.recipe,
            (
                CalibratedPopulationGraph,
                CalibratedPopulationIntervention,
                CalibratedPopulationQuery,
            ),
        ):
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
            controls = json.loads(self.construction.controls_json)
            if controls.get("topology") == "query-calibrated-gp-root" and (
                self.device != "cpu" or self.count != 1
            ):
                raise ValueError("GP query construction requires one population and CPU")
            if controls.get("topology") == "calibrate-gp-root":
                source = controls.get("source", {}).get("source", {})
                if (
                    self.device != "cpu"
                    or self.count != 1
                    or type(source.get("seed")) is not int
                    or source["seed"] != self.seed
                    or type(source.get("task_id")) is not int
                    or source["task_id"] != self.start
                ):
                    raise ValueError(
                        "GP fitting requires one exact retained source identity and CPU"
                    )
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
    if (
        request.construction is not None
        and json.loads(request.construction.controls_json).get("topology") == "tree-extension"
    ):
        from r0.modules.generation.lib.tree_extension_construction import generate_tree_extension

        return generate_tree_extension(
            request.construction,
            limits=request.causal_limits or CausalLimits(),
            seed=request.seed,
            task_id=task_id,
            device=request.device,
        )
    task = generate_task(
        selected_recipe(request, task_id), seed=request.seed, task_id=task_id, device=request.device
    )
    sampling = sampling_record(request)
    if sampling is None:
        return task
    record = json.loads(task.recipe_json)
    record["sampling"] = sampling
    return replace(
        task, recipe_json=json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def record_matches(record: dict[str, Any], request: RunRequest, task_id: int) -> bool:
    from r0.modules.generation.lib.request_matching import record_matches as matches

    return matches(record, request, task_id)


def request_from_dict(data: dict[str, Any]) -> RunRequest:
    fields = dict(data)
    candidate = fields.get("recipe")
    if type(candidate) is dict:
        candidate = cast(dict[str, Any], candidate)
        if candidate.get("version") in (
            "gp-root-dag-v1",
            "gp-root-calibration-v1",
            "gp-root-calibration-query-v1",
        ):
            validate_gp_root_run_identity(
                candidate,
                seed=fields.get("seed"),
                start=fields.get("start", 0),
                count=fields.get("count"),
                sampling=fields.get("sampling"),
                construction=fields.get("construction"),
            )
    if (
        type(candidate) is dict
        and cast(dict[str, Any], candidate).get("version")
        in (
            "calibrated-population-graph-v1",
            "calibrated-population-intervention-v1",
            "calibrated-population-query-v1",
        )
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
    raw_recipe = cast(
        dict[str, Any],
        construction.task_record()
        if "recipe" not in fields and construction is not None
        else fields.pop("recipe", None),
    )
    if type(raw_recipe) is dict and raw_recipe.get("version") in (
        "gp-root-dag-v1",
        "gp-root-calibration-v1",
        "gp-root-calibration-query-v1",
    ):
        if (
            raw_recipe["version"] in ("gp-root-calibration-v1", "gp-root-calibration-query-v1")
            and fields.get("device", "cpu") != "cpu"
        ):
            raise ValueError("GP calibrated runs require CPU replay")
        validate_gp_root_run_identity(
            raw_recipe,
            seed=fields.get("seed"),
            start=fields.get("start", 0),
            count=fields.get("count"),
            sampling=fields.get("sampling"),
            construction=construction,
        )
        # Admit the ordinary run header before decoding any retained covariance.
        header = {k: v for k, v in fields.items() if k != "sampling"}
        RunRequest(LinearRecipe(), causal_limits=limits, **header)
    if type(raw_recipe) is dict and raw_recipe.get("version") in (
        "calibrated-population-graph-v1",
        "calibrated-population-intervention-v1",
        "calibrated-population-query-v1",
    ):
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
            task_source = (
                raw_recipe["base"]
                if raw_recipe["version"] == "calibrated-population-intervention-v1"
                else raw_recipe
            )
            task_fields = dict(task_source["original"]["request"]["base"]["task"])
            if type(task_fields["class_offsets"]) not in (tuple, list):
                raise ValueError("class_offsets must be an array")
            task_fields["class_offsets"] = tuple(task_fields["class_offsets"])
            preview = LinearRecipe(**task_fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid calibrated population task metadata") from exc
        # Validate the complete outer RunRequest schema/budgets against cheap task
        # controls, without decoding either retained population. Discard preview.
        RunRequest(preview, causal_limits=limits, **fields)
    from r0.modules.generation.lib.xgboost_population_transport import (
        preflight_xgboost_population_run,
    )

    preflight_xgboost_population_run(raw_recipe, fields, limits, construction)
    recipe = recipe_from_dict(raw_recipe, limits=limits or CausalLimits())
    sampling = fields.pop("sampling", None)
    return RunRequest(
        recipe,
        sampling=None if sampling is None else RecipeSampling(**sampling),
        causal_limits=limits,
        construction=construction,
        **fields,
    )
