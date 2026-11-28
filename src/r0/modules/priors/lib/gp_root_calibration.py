"""Fixed target calibration from a separately keyed finite GP-root DAG."""

import json
from dataclasses import asdict, dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_target_calibration import (
    FunctionalTargetCalibration,
    fit_target_calibration,
)
from r0.modules.priors.lib.gp_column_population import realize_gp_columns
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe, generate_gp_root_dag
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.targets import adapt_targets
from r0.modules.priors.lib.training_task import TrainingTask
from r0.modules.priors.lib.weight_geometry import Matrix


def _fit_base(original: GPRootDAGRecipe, rows: int):
    if type(rows) is not int or not 32 <= rows <= 1024:
        raise ValueError("GP root calibration requires 32..1024 independent finite GP rows")
    if (
        not isinstance(original.base, FunctionalCausalRecipe)
        and original.task.task_type != "regression"
    ):
        raise ValueError("direct categorical targets require category-preserving calibration")
    if isinstance(original.base, FunctionalCausalRecipe) and original.base.target_calibration_rows:
        raise ValueError(
            "GP roots require explicit retained GP calibration, not placeholder iid fitting"
        )
    return replace(original.base, task=replace(original.task, rows=rows, context_rows=rows // 2))


def _identity(original: GPRootDAGRecipe, seed: int, task_id: int) -> None:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("GP calibration identity must contain exact nonnegative integers")
    if (seed, task_id) == (original.source.seed, original.source.task_id):
        raise ValueError("GP calibration requires a distinct independent population identity")


def _admit(original: GPRootDAGRecipe, fit: GPRootDAGRecipe, limits: CausalLimits) -> None:
    for cap in (limits, original.limits, fit.limits):
        original.admit()
        fit.admit()
        count, storage = function_record_cost((original, fit), cap)
        reports = [r.source.request.admit(cap) for r in (original, fit)]
        graphs = [replace(r.base, limits=cap).admit() for r in (original, fit)]
        if (
            count + sum(r["parameters"] for r in reports + graphs) > cap.parameters
            or 8 * storage + 4 * sum(r["estimated_working_bytes"] for r in reports + graphs)
            > cap.working_bytes
        ):
            raise ValueError("GP independent calibration aggregate exceeds limits")


@dataclass(frozen=True, slots=True)
class GPRootCalibration:
    original: GPRootDAGRecipe
    fit: GPRootDAGRecipe
    calibration: FunctionalTargetCalibration
    limits: CausalLimits
    version: str = "gp-root-calibration-v1"

    @property
    def task(self):
        return self.original.task

    def validate(self) -> None:
        if (
            type(self.original) is not GPRootDAGRecipe
            or type(self.fit) is not GPRootDAGRecipe
            or type(self.calibration) is not FunctionalTargetCalibration
            or type(self.limits) is not CausalLimits
            or self.version != "gp-root-calibration-v1"
        ):
            raise ValueError("invalid retained GP root calibration")
        from r0.modules.priors.lib.gp_root_calibration_decoding import (
            preflight_gp_root_calibration,
        )

        preflight_gp_root_calibration(asdict(self), limits=self.limits)
        _identity(self.original, self.fit.source.seed, self.fit.source.task_id)
        expected_base = _fit_base(self.original, self.calibration.rows)
        request = replace(
            self.original.source.request,
            rows=self.calibration.rows,
            locations=self.fit.source.request.locations,
        )
        if (
            not exact_frozen_equal(self.fit.base, expected_base)
            or not exact_frozen_equal(self.fit.source.request, request)
            or not exact_frozen_equal(self.fit.bindings, self.original.bindings)
        ):
            raise ValueError("GP calibration must preserve the graph and GP population law")
        _admit(self.original, self.fit, self.limits)
        generate_gp_root_dag(
            self.original, seed=self.original.source.seed, task_id=self.original.source.task_id
        )
        replay = generate_gp_root_dag(
            self.fit, seed=self.fit.source.seed, task_id=self.fit.source.task_id
        )
        wanted = fit_target_calibration(replay.scores, record_fingerprint(asdict(self.original)))
        if not exact_frozen_equal(self.calibration, wanted):
            raise ValueError("GP target calibration differs from independent population replay")


def calibrate_gp_root_dag(
    original: GPRootDAGRecipe,
    *,
    calibration_rows: int,
    calibration_seed: int,
    calibration_task_id: int,
    calibration_locations: Matrix | None = None,
    limits: CausalLimits | None = None,
) -> GPRootCalibration:
    if type(original) is not GPRootDAGRecipe:
        raise ValueError("GP calibration requires a retained GP root graph")
    _identity(original, calibration_seed, calibration_task_id)
    cap = limits or original.limits
    base = _fit_base(original, calibration_rows)
    request = replace(
        original.source.request,
        rows=calibration_rows,
        locations=original.source.request.locations
        if calibration_locations is None
        else calibration_locations,
    )
    # Both complete runtime allocations and retained original metadata precede draws.
    original.admit()
    request.admit(cap)
    graph = replace(base, limits=cap).admit()
    count, storage = function_record_cost((original, request, base), cap)
    gp = request.admit(cap)
    if (
        count + graph["parameters"] + gp["parameters"] > cap.parameters
        or 8 * storage + 8 * (graph["estimated_working_bytes"] + gp["estimated_working_bytes"])
        > cap.working_bytes
    ):
        raise ValueError("GP calibration construction aggregate exceeds limits")
    generate_gp_root_dag(original, seed=original.source.seed, task_id=original.source.task_id)
    source = realize_gp_columns(request, seed=calibration_seed, task_id=calibration_task_id)
    fit = GPRootDAGRecipe(source, base, original.bindings, cap)
    _admit(original, fit, cap)
    raw = generate_gp_root_dag(fit, seed=calibration_seed, task_id=calibration_task_id)
    calibration = fit_target_calibration(raw.scores, record_fingerprint(asdict(original)))
    result = GPRootCalibration(original, fit, calibration, cap)
    result.validate()
    return result


def generate_calibrated_gp_root(
    recipe: GPRootCalibration, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if (
        any(type(v) is not int for v in (seed, task_id))
        or (seed, task_id) != (recipe.original.source.seed, recipe.original.source.task_id)
        or device != "cpu"
    ):
        raise ValueError("retained GP calibration requires exact source identity and CPU replay")
    recipe.validate()
    original = generate_gp_root_dag(recipe.original, seed=seed, task_id=task_id, device=device)
    scores = recipe.calibration.apply(original.scores)
    if isinstance(recipe.original.base, FunctionalCausalRecipe):
        targets, probabilities, classes = adapt_targets(
            recipe.task,
            scores,
            noise_rng=functional_rng(seed, task_id, "noise"),
            label_rng=functional_rng(seed, task_id, "labels"),
        )
    else:
        if original.classes:
            raise ValueError("direct categories cannot receive numeric GP calibration")
        targets, probabilities, classes = scores[:, 0], None, original.classes
    config = asdict(recipe)
    record = json.loads(original.recipe_json)
    record["recipe"] = config
    record["execution"]["implementation"] = recipe.version
    identity = record_fingerprint((config, seed, task_id))
    return replace(
        original,
        source_task_id=identity,
        episode_id=record_fingerprint((identity, recipe.task.split, recipe.task.context_rows)),
        targets=targets,
        scores=scores,
        probabilities=probabilities,
        classes=classes,
        recipe_json=encode(record),
        mechanism_json=encode(
            {
                "family": "gp_root_calibration",
                "version": recipe.version,
                "original": json.loads(original.mechanism_json),
                "calibration": asdict(recipe.calibration),
                "fit_population_id": recipe.fit.source.population_id,
                "query_fit": False,
            }
        ),
    )
