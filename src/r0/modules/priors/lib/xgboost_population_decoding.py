"""Strict actual-population XGBoost replay; never import or refit the backend."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.gp_column_decoding import gp_fields as _complete
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.xgboost_fitting import canonical_xgboost_record
from r0.modules.priors.lib.xgboost_population_fit import (
    PopulationXGBoostFit,
    PopulationXGBoostMetadata,
)
from r0.modules.priors.lib.xgboost_population_plan import PopulationXGBoostPlan
from r0.modules.priors.lib.xgboost_vector_graph import decode_xgboost_vector

_DEFAULT = CausalLimits()


def population_xgboost_fit_metadata(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> PopulationXGBoostMetadata:
    if type(limits) is not CausalLimits:
        raise ValueError("population XGBoost replay needs caller limits")
    admit_function_metadata(data, limits)
    try:
        raw = _complete(data, PopulationXGBoostFit)
        declared = CausalLimits(**_complete(raw["limits"], CausalLimits))
        cap = CausalLimits(
            **{
                f.name: min(getattr(limits, f.name), getattr(declared, f.name))
                for f in fields(CausalLimits)
            }
        )
        plan = PopulationXGBoostPlan(**_complete(raw["plan"], PopulationXGBoostPlan))
        budget = RawDrawLimits(**_complete(raw["raw_limits"], RawDrawLimits))
        cost = plan.admit(cap, budget)
        if (
            type(raw["model_json"]) is not str
            or cost + 12 * len(raw["model_json"]) > cap.working_bytes
        ):
            raise ValueError("population XGBoost native model workspace limit exceeded")
        raw.update(plan=plan, limits=declared, raw_limits=budget)
        for key, width in (("inputs", plan.input_width), ("targets", plan.target_width)):
            rows = fitted_array(raw[key], plan.rows)
            if len(rows) != plan.rows:
                raise ValueError("population XGBoost table row count differs")
            raw[key] = tuple(tuple(fitted_array(row, width)) for row in rows)
        for key in ("initial_rng", "final_rng"):
            raw[key] = tuple(
                GeneratorSnapshot(**_complete(v, GeneratorSnapshot))
                for v in fitted_array(raw[key], 2)
            )
        records = fitted_array(raw["raw"], 2)
        for r in records:
            _complete(r, RawDrawRecord)
        raw["raw"] = tuple(decode_fitted_raw_draw(r) for r in records)
        raw["function"] = decode_xgboost_vector(raw["function"])
        result = PopulationXGBoostMetadata(**raw, caller_limits=cap)
        if canonical_xgboost_record(asdict(result)) != canonical_xgboost_record(data):
            raise ValueError("population XGBoost differs from exact archive reconstruction")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed population XGBoost fit archive") from exc


def population_xgboost_fit_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> PopulationXGBoostFit:
    metadata = population_xgboost_fit_metadata(data, limits=limits)
    return PopulationXGBoostFit(
        **{f.name: getattr(metadata, f.name) for f in fields(PopulationXGBoostMetadata)},
        caller_limits=limits,
    )
