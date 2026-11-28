"""Fresh same-law query populations retain complete calibration and source lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.population_target_calibration import CalibratedPopulationGraph
from r0.modules.priors.lib.recursive_population_graph_records import (
    RecursivePopulationGraph,
    realize_recursive_population_graph,
)
from r0.modules.priors.lib.representatives import record_fingerprint

ENGINES = ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937")


def query_stream(
    request: object, seed: int, task_id: int, engine: str, construction_seed: int
) -> np.random.Generator:
    if type(engine) is not str or engine not in ENGINES:
        raise ValueError("calibrated query requires an explicit supported NumPy engine")
    key = int(
        record_fingerprint(
            ("fixed-calibration-query-stream-v1", request, seed, task_id, engine, construction_seed)
        ),
        16,
    )
    return np.random.Generator(getattr(np.random, engine)(key))


@dataclass(frozen=True, slots=True)
class CalibratedPopulationQuery:
    source: CalibratedPopulationGraph | CalibratedPopulationQuery
    original: RecursivePopulationGraph
    engine: str
    construction_seed: int
    version: str = "calibrated-population-query-v1"

    def lineage(self) -> tuple[CalibratedPopulationGraph, tuple[CalibratedPopulationQuery, ...]]:
        current: CalibratedPopulationGraph | CalibratedPopulationQuery = self
        queries: list[CalibratedPopulationQuery] = []
        seen: set[int] = set()
        while type(current) is CalibratedPopulationQuery:
            if id(current) in seen or len(queries) >= 64:
                raise ValueError("calibrated query lineage exceeds bounded acyclic custody")
            seen.add(id(current))
            queries.append(current)
            current = current.source
        if type(current) is not CalibratedPopulationGraph:
            raise ValueError("calibrated query requires its complete source calibration")
        return current, tuple(reversed(queries))

    @property
    def task(self):
        return self.original.task

    @property
    def limits(self):
        return self.lineage()[0].limits

    @property
    def calibration(self):
        return self.lineage()[0].calibration

    @property
    def calibration_population(self):
        return self.lineage()[0].calibration_population

    def validate(self, *, limits: CausalLimits | None = None) -> None:
        from r0.modules.priors.lib.calibrated_query_decoding import query_source_metadata

        cap = limits or self.limits
        function_record_cost(self, cap)
        query_source_metadata(asdict(self), limits=cap)
        base, lineage = self.lineage()
        base.validate()
        for item in lineage:
            item.original.validate_replay(limits=cap)


def realize_calibrated_population_query(
    source: CalibratedPopulationGraph | CalibratedPopulationQuery,
    *,
    seed: int,
    task_id: int,
    engine: str | None = None,
    construction_seed: int | None = None,
) -> CalibratedPopulationQuery:
    from r0.modules.priors.lib.calibrated_query_admission import preflight_calibrated_query

    if type(source) not in (CalibratedPopulationGraph, CalibratedPopulationQuery):
        raise ValueError("fresh calibrated query requires a retained calibrated source")
    preflight_calibrated_query(
        source, seed=seed, task_id=task_id, engine=engine, construction_seed=construction_seed
    )
    owned_seed = seed if construction_seed is None else construction_seed
    selected = engine if engine is not None else source.original.bindings[0].sample.initial.engine
    request = source.original.request
    source.validate()
    rng = query_stream(asdict(request), seed, task_id, selected, owned_seed)
    original = realize_recursive_population_graph(request, seed=seed, task_id=task_id, rng=rng)
    result = CalibratedPopulationQuery(source, original, selected, owned_seed)
    result.validate()
    return result
