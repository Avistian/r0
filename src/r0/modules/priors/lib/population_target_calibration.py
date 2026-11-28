"""Retain an independent row-owned graph population for fixed target calibration."""

from dataclasses import asdict, dataclass, replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_target_calibration import (
    FunctionalTargetCalibration,
    fit_target_calibration,
)
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.recursive_population_graph_plan import RecursivePopulationGraphRequest
from r0.modules.priors.lib.recursive_population_graph_records import (
    RecursivePopulationGraph,
    realize_recursive_population_graph,
)
from r0.modules.priors.lib.recursive_population_records import population_identity
from r0.modules.priors.lib.representatives import record_fingerprint


def calibration_request(
    original: RecursivePopulationGraph, rows: int
) -> RecursivePopulationGraphRequest:
    return calibration_request_from_source(original.request, rows)


def calibration_request_from_source(
    request: RecursivePopulationGraphRequest, rows: int
) -> RecursivePopulationGraphRequest:
    """Plan the independent population using metadata-only authored previews too."""
    if type(rows) is not int or not 32 <= rows <= 4096:
        raise ValueError(
            "population target calibration requires 32..4096 independent rows "
            "under the retained forest population cap"
        )
    if (
        not isinstance(request.base, FunctionalCausalRecipe)
        and request.task.task_type != "regression"
    ):
        raise ValueError(
            "direct categorical targets retain category semantics, not numeric scaling"
        )
    task = replace(request.task, rows=rows, context_rows=rows // 2)
    return replace(
        request,
        base=replace(request.base, task=task),
        replacements=tuple(
            replace(binding, forest=replace(binding.forest, populations=rows))
            for binding in request.replacements
        ),
    )


def admit_population_calibration(
    original: RecursivePopulationGraph,
    request: RecursivePopulationGraphRequest,
    limits: CausalLimits,
) -> None:
    if type(limits) is not CausalLimits:
        raise ValueError("population calibration requires explicit caller limits")
    reports = [r.admit(limits=limits) for r in (original.request, request)]
    numbers, storage = function_record_cost((original, request), limits)
    if (
        4 * sum(r["parameters"] for r in reports) + numbers > limits.parameters
        or 4 * sum(r["estimated_working_bytes"] for r in reports) + 8 * storage
        > limits.working_bytes
    ):
        raise ValueError("aggregate independent population calibration exceeds limits")


@dataclass(frozen=True, slots=True)
class CalibratedPopulationGraph:
    original: RecursivePopulationGraph
    calibration_population: RecursivePopulationGraph
    calibration: FunctionalTargetCalibration
    limits: CausalLimits
    version: str = "calibrated-population-graph-v1"

    @property
    def task(self):
        return self.original.task

    def validate(self) -> None:
        if (
            self.version != "calibrated-population-graph-v1"
            or type(self.original) is not RecursivePopulationGraph
            or type(self.calibration_population) is not RecursivePopulationGraph
            or type(self.calibration) is not FunctionalTargetCalibration
        ):
            raise ValueError("invalid retained population calibration envelope")
        from r0.modules.priors.lib.population_calibration_decoding import (
            preflight_calibrated_population,
        )

        function_record_cost(self, self.limits)
        preflight_calibrated_population(asdict(self), limits=self.limits)
        fit = self.calibration_population
        expected = calibration_request(self.original, self.calibration.rows)
        if not exact_frozen_equal(fit.request, expected) or (fit.seed, fit.task_id) == (
            self.original.seed,
            self.original.task_id,
        ):
            raise ValueError("calibration must retain a distinct independent same-law population")
        admit_population_calibration(self.original, expected, self.limits)
        self.original.replay(limits=self.limits)
        evaluation = fit.replay(limits=self.limits)
        expected_fit = fit_target_calibration(
            evaluation.task.scores, record_fingerprint(asdict(self.original.request))
        )
        if not exact_frozen_equal(self.calibration, expected_fit):
            raise ValueError("population target calibration differs from independent replay")


def calibrate_population_graph(
    original: RecursivePopulationGraph,
    *,
    calibration_rows: int,
    calibration_seed: int,
    calibration_task_id: int,
    limits: CausalLimits | None = None,
) -> CalibratedPopulationGraph:
    if type(original) is not RecursivePopulationGraph:
        raise ValueError("target calibration requires an original retained population DAG")
    population_identity(calibration_seed, calibration_task_id)
    if (calibration_seed, calibration_task_id) == (original.seed, original.task_id):
        raise ValueError("calibration must use a distinct independent population identity")
    cap = limits or original.limits
    request = calibration_request(original, calibration_rows)
    admit_population_calibration(original, request, cap)
    original.replay(limits=cap)
    fit = realize_recursive_population_graph(
        request, seed=calibration_seed, task_id=calibration_task_id
    )
    calibration = fit_target_calibration(
        fit.replay(limits=cap).task.scores, record_fingerprint(asdict(original.request))
    )
    result = CalibratedPopulationGraph(original, fit, calibration, cap)
    result.validate()
    return result
