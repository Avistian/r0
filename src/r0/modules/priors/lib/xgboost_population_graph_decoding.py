"""Complete XGBoost population outer semantics precede source and fit replay."""

from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.population_calibration_custody import preflight_calibration_custody
from r0.modules.priors.lib.population_graph_semantic_preflight import authored_semantic_preview
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_population_ancestors import ANCESTOR_TYPES, PopulationAncestors
from r0.modules.priors.lib.recursive_task_preflight import recursive_authored_preview
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.xgboost_population_decoding import (
    population_xgboost_fit_from_dict,
    population_xgboost_fit_metadata,
)
from r0.modules.priors.lib.xgboost_population_graph import (
    PopulationXGBoostNodeFit,
    XGBoostPopulationGraph,
    admit_xgboost_population_graph,
)

_DEFAULT = CausalLimits()


def preflight_xgboost_population_graph(data: Any, *, limits: CausalLimits = _DEFAULT):
    from r0.modules.priors.lib.generate import recipe_from_dict

    observation_metadata_cost(data, limits)
    try:
        raw = gp_fields(data, XGBoostPopulationGraph)
        if raw["version"] != "population-xgboost-graph-v1":
            raise ValueError("invalid population XGBoost graph version")
        cap = CausalLimits(**gp_fields(raw["limits"], CausalLimits))
        if any(getattr(cap, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
            raise ValueError("population XGBoost graph exceeds caller limits")
        budget = RawDrawLimits(**gp_fields(raw["raw_limits"], RawDrawLimits))
        preflight_calibration_custody(raw["base"], cap)
        preview = recipe_from_dict(
            authored_semantic_preview(recursive_authored_preview(raw["base"])), limits=cap
        )
        if type(preview) not in ANCESTOR_TYPES:
            raise ValueError("population XGBoost requires authored base")
        preview = cast(PopulationAncestors, preview)
        if type(raw["fits"]) not in (list, tuple) or not raw["fits"]:
            raise ValueError("population XGBoost needs retained node fits")
        nodes = [gp_fields(v, PopulationXGBoostNodeFit) for v in raw["fits"]]
        fits = [population_xgboost_fit_metadata(n["realization"], limits=cap) for n in nodes]
        plans = tuple((n["node"], f.plan) for n, f in zip(nodes, fits, strict=True))
        admit_xgboost_population_graph(
            preview,
            plans,
            raw["population"],
            raw["fit_seed"],
            raw["fit_task_id"],
            cap,
            budget,
            raw["fits"],
        )
        if tuple(n["node"] for n in nodes) != tuple(sorted({n["node"] for n in nodes})):
            raise ValueError("population XGBoost node order differs")
        runtime_nodes = list(raw["base"]["nodes"])
        for i, (node, fit) in enumerate(zip(nodes, fits, strict=True)):
            rows = node["row_ids"]
            count = fit.plan.rows
            total = preview.task.rows if raw["population"] == "support-only" else count
            if (
                node["version"] != "population-xgboost-node-fit-v1"
                or type(rows) not in (list, tuple)
                or len(rows) != count
                or len(set(rows)) != count
                or any(type(v) is not int or not 0 <= v < total for v in rows)
            ):
                raise ValueError("population XGBoost row identity metadata differs")
            if raw["population"] != "support-only" and tuple(rows) != tuple(range(count)):
                raise ValueError("population XGBoost batch row order differs")
            if fit.population != raw["population"] or fit.limits != cap or fit.raw_limits != budget:
                raise ValueError("population XGBoost fit lifecycle/caps differ")
            if i and fits[i - 1].final_rng != fit.initial_rng:
                raise ValueError("population XGBoost stream chain metadata differs")
            if node["prefix_digest"] != record_fingerprint(runtime_nodes[: node["node"]]):
                raise ValueError("population XGBoost original prefix identity differs")
            previous = runtime_nodes[node["node"]]
            runtime_nodes[node["node"]] = (
                {**previous, "function": asdict(fit.function)}
                if "equation" not in previous
                else {
                    **previous,
                    "equation": {**previous["equation"], "function": asdict(fit.function)},
                }
            )
        return raw, cap
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed population XGBoost graph metadata") from exc


def xgboost_population_graph_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT
) -> XGBoostPopulationGraph:
    from r0.modules.priors.lib.generate import recipe_from_dict

    raw, cap = preflight_xgboost_population_graph(data, limits=limits)
    base = recipe_from_dict(raw["base"], limits=cap)
    if type(base) not in ANCESTOR_TYPES:
        raise ValueError("population XGBoost requires authored base")
    base = cast(PopulationAncestors, base)
    fits = tuple(
        PopulationXGBoostNodeFit(
            n["node"],
            n["prefix_digest"],
            tuple(n["row_ids"]),
            population_xgboost_fit_from_dict(n["realization"], limits=cap),
            n["version"],
        )
        for n in raw["fits"]
    )
    result = XGBoostPopulationGraph(
        base,
        fits,
        raw["population"],
        raw["fit_seed"],
        raw["fit_task_id"],
        cap,
        RawDrawLimits(**raw["raw_limits"]),
    )
    if not gp_payload_equal(asdict(result), data):
        raise ValueError("population XGBoost differs from canonical archive")
    return result
