"""Reserve exact retained-fit storage and replay workspace before graph decoding."""

from dataclasses import fields, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.xgboost_fit_plan import XGBoostFitPlan
from r0.modules.priors.lib.xgboost_fitting import XGBoostFit


def reserve_fit_workspace(
    limits: CausalLimits, numbers: int, storage: int, replay: int
) -> CausalLimits:
    if numbers >= limits.parameters or 2 * storage + replay >= limits.working_bytes:
        raise ValueError("XGBoost graph aggregate fit replay limit exceeded")
    return replace(
        limits,
        parameters=limits.parameters - numbers,
        working_bytes=limits.working_bytes - 2 * storage - replay,
    )


def raw_fit_reservation(value: Any, limits: CausalLimits, node_count: int) -> CausalLimits:
    """Same metadata cost as function_record_cost on the eventual fit dataclasses."""
    from r0.modules.priors.lib.xgboost_fit_graph import XGBoostNodeFit

    admit_function_metadata(value, limits)
    raw_fits = fitted_array(value, min(limits.nodes, node_count))
    if not raw_fits:
        raise ValueError("XGBoost graph requires frozen fits")
    indices: list[int] = []
    replay = 0
    for raw in raw_fits:
        entry = _complete(raw, XGBoostNodeFit)
        index = entry["node"]
        if type(index) is not int or not 0 <= index < node_count:
            raise ValueError("XGBoost fit node exceeds graph")
        indices.append(index)
        fit = _complete(entry["realization"], XGBoostFit)
        plan = XGBoostFitPlan(**_complete(fit["plan"], XGBoostFitPlan))
        raw_limits = RawDrawLimits(**_complete(fit["raw_limits"], RawDrawLimits))
        model = fit["model_json"]
        if type(model) is not str:
            raise ValueError("XGBoost native model must be text")
        replay += plan.admit(limits, raw_limits) + 12 * len(model)
    if indices != sorted(set(indices)):
        raise ValueError("XGBoost fits must be ordered and unique")
    numbers, storage = _raw_cost(raw_fits)
    return reserve_fit_workspace(limits, numbers, storage, replay)


def _complete(value: Any, kind: type[Any]) -> dict[str, Any]:
    record = fitted_object(value, kind)
    if set(record) != {f.name for f in fields(kind)}:
        raise ValueError("XGBoost fit archive requires complete fields")
    return record


def _raw_cost(value: Any) -> tuple[int, int]:
    if type(value) in (int, float, bool):
        return 1, 64
    if type(value) is str:
        return 0, len(value.encode("utf-8"))
    if value is None:
        return 0, 0
    numbers, storage = 0, 32
    if type(value) is dict:
        mapping = cast(dict[str, Any], value)
        for key, item in mapping.items():
            n, s = _raw_cost(item)
            numbers += n
            storage += s + len(key.encode("utf-8"))
    elif type(value) in (list, tuple):
        for item in cast(list[Any] | tuple[Any, ...], value):
            n, s = _raw_cost(item)
            numbers += n
            storage += s
    else:
        raise ValueError("XGBoost fit metadata must be JSON")
    return numbers, storage
