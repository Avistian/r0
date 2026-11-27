"""Independent synthetic XGBoost fitting inside the guarded worker."""

from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_xgboost(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.XGBoostFittedGraph:
    if set(fields) != {"skeleton", "fits"}:
        raise ValueError("XGBoost construction requires a skeleton and selected fit plans")
    skeleton = priors.recipe_from_dict(fields["skeleton"], limits=limits)
    if type(skeleton) is not priors.FunctionalCausalRecipe:
        raise ValueError("XGBoost construction requires an authored functional skeleton")
    raw_fits = fields["fits"]
    if (
        type(raw_fits) is not tuple
        or not raw_fits
        or len(cast(tuple[Any, ...], raw_fits)) > limits.nodes
    ):
        raise ValueError("XGBoost construction requires bounded selected node plans")
    plans: list[tuple[int, priors.XGBoostFitPlan]] = []
    raw_limits = priors.RawDrawLimits()
    workspace = skeleton.admit()["estimated_working_bytes"]
    for raw in cast(tuple[Any, ...], raw_fits):
        if type(raw) is not dict or set(cast(dict[str, Any], raw)) != {"node", "plan"}:
            raise ValueError("each XGBoost selection requires exactly node and plan")
        entry = cast(dict[str, Any], raw)
        index = entry["node"]
        if type(index) is not int or not 0 <= index < len(skeleton.nodes):
            raise ValueError("XGBoost selection requires an existing integer node index")
        if plans and index <= plans[-1][0]:
            raise ValueError("XGBoost selections must be unique and ordered")
        if type(entry["plan"]) is not dict:
            raise ValueError("XGBoost selection plan must be an object")
        try:
            plan = priors.XGBoostFitPlan(**cast(dict[str, Any], entry["plan"]))
        except TypeError as exc:
            raise ValueError("invalid XGBoost plan fields") from exc
        node = skeleton.nodes[index]
        if node.pipeline is not None or (node.function.input_width, node.function.output_width) != (
            plan.input_width,
            plan.output_width,
        ):
            raise ValueError("XGBoost plan must match an equation without a fitted pipeline")
        workspace += plan.admit(limits, raw_limits)
        plans.append((index, plan))
    if workspace > limits.working_bytes:
        raise ValueError("aggregate XGBoost construction workspace limit exceeded")
    # All selections and known workspace are admitted before streams or backend work.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    fits = tuple(
        priors.XGBoostNodeFit(
            index,
            priors.fit_xgboost(
                plan,
                input_rng=streams[0],
                target_rng=streams[1],
                backend_rng=streams[2],
                limits=limits,
                raw_limits=raw_limits,
            ),
        )
        for index, plan in plans
    )
    return priors.with_xgboost_fits(skeleton, fits)
