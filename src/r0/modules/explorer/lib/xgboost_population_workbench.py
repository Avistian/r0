"""Retained XGBoost fits: explicit construction, scoped inspection and frozen queries."""

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.identities import identity_numbers
from r0.modules.explorer.lib.saved import run_saved_document

if TYPE_CHECKING:
    from r0.modules.explorer.lib.session import ExplorerSession

_VERSION = "population-xgboost-graph-v1"


def xgboost_population_action(
    session: "ExplorerSession", action: str, data: dict[str, Any]
) -> dict[str, Any] | None:
    if session.task is None:
        return None
    record = json.loads(session.task.recipe_json)
    source = record["recipe"]
    if source.get("version") != _VERSION:
        return None
    if action in (
        "intervention_controls",
        "intervene",
        "sample_intervention_policy",
        "edit",
        "pair",
        "curve",
        "population_calibrate",
        "population_query",
    ):
        raise ValueError(
            "Population XGBoost paired, calibration, reference and policy composition remains "
            "pending. Use explicit construction to fit a changed mechanism."
        )
    if action in ("mechanism", "xgboost_population_controls"):
        return dict(
            xgboost_population=True,
            controls=dict(
                topology="xgboost-population",
                base=source["base"],
                plans=[dict(node=f["node"], plan=f["realization"]["plan"]) for f in source["fits"]],
                population=source["population"],
                fit_seed=source["fit_seed"],
                fit_task_id=source["fit_task_id"],
                engine=source["fits"][0]["realization"]["initial_rng"][0]["engine"],
                raw_limits=source["raw_limits"],
            ),
            fit_rows=[f["row_ids"] for f in source["fits"]],
            notice=(
                "Explicit construction fits actual synthetic parent rows. Replay and "
                "independent queries keep retained native models fixed. Fit tables "
                "use their own original row IDs."
            ),
        )
    if action not in ("xgboost_population_replay", "xgboost_population_query"):
        return None
    data = identity_numbers(data, for_browser=False)
    seed, task_id = record["seed"], record["task_id"]
    if action == "xgboost_population_query":
        if source["population"] != "independent-ancestors":
            raise ValueError(
                "Support and batch-reference mechanisms require an explicit new fit; "
                "frozen Next queries require independent ancestors"
            )
        seed, task_id = data.get("seed"), data.get("task_id")
    task, archive, run = run_saved_document(
        session.output_dir,
        dict(
            recipe=source,
            seed=seed,
            start=task_id,
            count=1,
            group_size=1,
            device="cpu",
            causal_limits=asdict(caller_limits(data)),
            max_seconds=data.get("max_seconds", 60),
            max_bytes=data.get("max_bytes", 256 * 1024**2),
        ),
    )
    session.task, session.archive, session.run = task, archive, run
    session.pair = session.edit = session.intervention = None
    return session.request("view", {})
