"""Same-law finite GP queries retain their complete source and one independent fit."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_target_calibration import FunctionalTargetCalibration
from r0.modules.priors.lib.gp_column_population import realize_gp_columns
from r0.modules.priors.lib.gp_population_metadata_cost import retained_cost
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe, generate_gp_root_dag
from r0.modules.priors.lib.weight_geometry import Matrix


@dataclass(frozen=True, slots=True)
class GPRootCalibrationQuery:
    source: GPRootCalibration | GPRootCalibrationQuery
    original: GPRootDAGRecipe
    limits: CausalLimits
    version: str = "gp-root-calibration-query-v1"

    @property
    def task(self):
        return self.original.task

    @property
    def fit(self) -> GPRootDAGRecipe:
        return self.source.fit

    @property
    def calibration(self) -> FunctionalTargetCalibration:
        return self.source.calibration

    def validate(self) -> None:
        from r0.modules.priors.lib.gp_root_query_metadata import query_source_metadata

        if type(self.limits) is not CausalLimits:
            raise ValueError("GP query requires explicit limits")
        retained_cost(self, self.limits)
        query_source_metadata(asdict(self), limits=self.limits)
        self.source.validate()
        generate_gp_root_dag(
            self.original, seed=self.original.source.seed, task_id=self.original.source.task_id
        )


def realize_gp_root_calibration_query(
    source: GPRootCalibration | GPRootCalibrationQuery,
    *,
    seed: int,
    task_id: int,
    locations: Matrix | None = None,
    limits: CausalLimits | None = None,
) -> GPRootCalibrationQuery:
    from r0.modules.priors.lib.gp_root_query_admission import query_construction_plan

    if type(source) not in (GPRootCalibration, GPRootCalibrationQuery):
        raise ValueError("GP query requires a complete retained calibration or query")
    cap = source.limits if limits is None else limits
    if type(cap) is not CausalLimits:
        raise ValueError("GP query requires explicit caller limits")
    retained_cost(source, cap)
    metadata, request = query_construction_plan(
        asdict(source), seed=seed, task_id=task_id, locations=locations, limits=cap
    )
    source.validate()
    population = realize_gp_columns(request, seed=seed, task_id=task_id)
    original = GPRootDAGRecipe(
        population, source.original.base, source.original.bindings, metadata.limits
    )
    result = GPRootCalibrationQuery(source, original, metadata.limits)
    result.validate()
    return result
