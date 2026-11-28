"""Strict actual-parent XGBoost controls before source replay or stream creation."""

from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.generator_snapshot import snapshot_generator
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_population_ancestors import ANCESTOR_TYPES, PopulationAncestors
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.xgboost_population_graph import (
    admit_xgboost_population_graph,
    fit_xgboost_population_graph,
)
from r0.modules.priors.lib.xgboost_population_graph_decoding import (
    preflight_xgboost_population_graph,
)
from r0.modules.priors.lib.xgboost_population_plan import PopulationXGBoostPlan
from r0.modules.priors.lib.xgboost_population_task import (
    preflight_xgboost_population_generation,
    preflight_xgboost_population_record,
)

_ENGINES = ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937")
_FIELDS = {
    "topology",
    "base",
    "plans",
    "population",
    "fit_seed",
    "fit_task_id",
    "engine",
    "raw_limits",
}


def population_xgboost_controls(
    data: Any, *, construction_seed: int, seed: int, task_id: int, limits: CausalLimits
):
    from r0.modules.priors.lib.generate import recipe_from_dict

    observation_metadata_cost(data, limits)
    if type(data) is not dict:
        raise ValueError("population XGBoost construction requires complete controls")
    raw = cast(dict[str, Any], data)
    if set(raw) != _FIELDS or raw["topology"] != "xgboost-population":
        raise ValueError("population XGBoost construction fields differ")
    if type(construction_seed) is not int or not 0 <= construction_seed < 2**63:
        raise ValueError("population XGBoost construction identity must be exact")
    if type(raw["engine"]) is not str or raw["engine"] not in _ENGINES:
        raise ValueError("population XGBoost construction engine is unsupported")
    preflight_xgboost_population_generation(raw, seed=seed, task_id=task_id)
    try:
        budget = RawDrawLimits(**gp_fields(raw["raw_limits"], RawDrawLimits))
        entries = raw["plans"]
        if type(entries) not in (list, tuple) or not entries:
            raise ValueError("population XGBoost construction needs node plans")
        plans: list[tuple[int, PopulationXGBoostPlan]] = []
        for entry in entries:
            if type(entry) is not dict:
                raise ValueError("population XGBoost node plan requires an object")
            entry = cast(dict[str, Any], entry)
            if set(entry) != {"node", "plan"}:
                raise ValueError("population XGBoost construction requires complete node plans")
            plans.append(
                (
                    entry["node"],
                    PopulationXGBoostPlan(**gp_fields(entry["plan"], PopulationXGBoostPlan)),
                )
            )
        preflight_calibration_custody(raw["base"], limits)
        preview = recipe_from_dict(
            authored_semantic_preview(recursive_authored_preview(raw["base"])), limits=limits
        )
        if type(preview) not in ANCESTOR_TYPES:
            raise ValueError("population XGBoost construction requires authored graph")
        preview = cast(PopulationAncestors, preview)
        admit_xgboost_population_graph(
            preview,
            tuple(plans),
            raw["population"],
            raw["fit_seed"],
            raw["fit_task_id"],
            limits,
            budget,
        )
        return raw, tuple(plans), budget
    except (TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed population XGBoost construction") from exc


def population_xgboost_streams(raw: dict[str, Any], construction_seed: int):
    engine = getattr(np.random, raw["engine"])
    return tuple(
        np.random.Generator(
            engine(
                np.random.SeedSequence(
                    [construction_seed, raw["fit_seed"], raw["fit_task_id"], index]
                )
            )
        )
        for index in (0, 1)
    )


def construct_xgboost_population(
    data: Any, *, construction_seed: int, seed: int, task_id: int, limits: CausalLimits
):
    from r0.modules.priors.lib.generate import recipe_from_dict

    raw, plans, budget = population_xgboost_controls(
        data, construction_seed=construction_seed, seed=seed, task_id=task_id, limits=limits
    )
    base = recipe_from_dict(raw["base"], limits=limits)
    if type(base) not in ANCESTOR_TYPES:
        raise ValueError("population XGBoost requires authored base")
    base = replace(cast(PopulationAncestors, base), limits=CausalLimits(**raw["base"]["limits"]))
    if not gp_payload_equal(asdict(base), raw["base"]):
        raise ValueError("population XGBoost authored source custody differs")
    target, backend = population_xgboost_streams(raw, construction_seed)
    return fit_xgboost_population_graph(
        base,
        plans,
        population=raw["population"],
        seed=raw["fit_seed"],
        task_id=raw["fit_task_id"],
        target_rng=target,
        backend_rng=backend,
        limits=limits,
        raw_limits=budget,
    )


def xgboost_population_construction_matches(
    data: Any, record: Any, *, construction_seed: int, seed: int, task_id: int, limits: CausalLimits
) -> bool:
    """Verify producer construction custody without fitting a second native model."""
    record = preflight_xgboost_population_record(record)
    if (record["seed"], record["task_id"]) != (seed, task_id):
        return False
    raw, plans, budget = population_xgboost_controls(
        data, construction_seed=construction_seed, seed=seed, task_id=task_id, limits=limits
    )
    graph, _ = preflight_xgboost_population_graph(record["recipe"], limits=limits)
    expected = {
        "base": raw["base"],
        "population": raw["population"],
        "fit_seed": raw["fit_seed"],
        "fit_task_id": raw["fit_task_id"],
        "limits": asdict(limits),
        "raw_limits": asdict(budget),
    }
    if not gp_payload_equal({key: graph[key] for key in expected}, expected):
        return False
    actual_plans = [(fit["node"], fit["realization"]["plan"]) for fit in graph["fits"]]
    if not gp_payload_equal(actual_plans, [(node, asdict(plan)) for node, plan in sorted(plans)]):
        return False
    states = [
        asdict(snapshot_generator(r, byte_limit=budget.state_bytes))
        for r in population_xgboost_streams(raw, construction_seed)
    ]
    return gp_payload_equal(states, graph["fits"][0]["realization"]["initial_rng"])
