"""Strict bounded fit archive reconstruction without importing/refitting the backend."""

from dataclasses import asdict, fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan
from r0.modules.priors.lib.xgboost_fitting import XGBoostFit, canonical_xgboost_record
from r0.modules.priors.lib.xgboost_vector_graph import decode_xgboost_vector

_DEFAULT = CausalLimits()


def xgboost_fit_from_dict(data: dict[str, Any], *, limits: CausalLimits = _DEFAULT) -> XGBoostFit:
    if type(limits) is not CausalLimits:
        raise ValueError("XGBoost fit decoder requires caller limits")
    admit_function_metadata(data, limits)
    try:
        raw = _complete(data, XGBoostFit)
        if set(raw) != {f.name for f in fields(XGBoostFit)}:
            raise ValueError("XGBoost fit archive requires complete fields")
        declared = CausalLimits(**_complete(raw["limits"], CausalLimits))
        effective = CausalLimits(
            **{
                f.name: min(getattr(limits, f.name), getattr(declared, f.name))
                for f in fields(CausalLimits)
            }
        )
        plan = XGBoostFitPlan(**_complete(raw["plan"], XGBoostFitPlan))
        raw_limits = RawDrawLimits(**_complete(raw["raw_limits"], RawDrawLimits))
        cost = plan.admit(effective, raw_limits)
        model = raw["model_json"]
        if type(model) is not str or cost + 12 * len(model) > effective.working_bytes:
            raise ValueError("XGBoost fit native model replay workspace limit exceeded")
        raw["plan"], raw["limits"], raw["raw_limits"] = plan, declared, raw_limits
        raw["function"] = decode_xgboost_vector(raw["function"])
        for key in ("initial_rng", "final_rng"):
            raw[key] = tuple(
                GeneratorSnapshot(**_complete(v, GeneratorSnapshot))
                for v in fitted_array(raw[key], 3)
            )
        records = fitted_array(raw["raw"], 3)
        specs = plan.raw_specs()
        if len(records) != len(specs):
            raise ValueError("XGBoost raw record count differs")
        for record, spec in zip(records, specs, strict=True):
            entry = _complete(record, RawDrawRecord)
            if len(fitted_array(entry["values"], spec.numbers)) != spec.numbers:
                raise ValueError("XGBoost raw record length differs")
        raw["raw"] = tuple(decode_fitted_raw_draw(v) for v in records)
        result = XGBoostFit(**raw)
        result.validate(limits=effective)
        if canonical_xgboost_record(asdict(result)) != canonical_xgboost_record(data):
            raise ValueError("XGBoost fit archive differs from exact reconstruction")
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed XGBoost fit archive") from exc


def _complete(value: Any, kind: type[Any]) -> dict[str, Any]:
    record = fitted_object(value, kind)
    if set(record) != {field.name for field in fields(kind)}:
        raise ValueError("XGBoost fit nested archive requires complete fields")
    return record
