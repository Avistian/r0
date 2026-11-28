"""Inspect authored GP-root drafts and replay retained finite populations."""

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.interventions import intervention_controls
from r0.modules.explorer.lib.saved import run_saved_document

if TYPE_CHECKING:
    from r0.modules.explorer.lib.session import ExplorerSession


def gp_root_action(
    session: "ExplorerSession",
    action: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    if session.task is None:
        return None
    record = json.loads(session.task.recipe_json)
    recipe = record["recipe"]
    if recipe.get("version") not in (
        "gp-root-dag-v1",
        "gp-root-calibration-v1",
        "gp-root-calibration-query-v1",
    ):
        return None
    original = (
        recipe["original"]
        if recipe["version"] in ("gp-root-calibration-v1", "gp-root-calibration-query-v1")
        else recipe
    )
    if action == "gp_root_query":
        if recipe["version"] == "gp-root-dag-v1":
            raise ValueError("Fit an independent GP calibration before fixed-fit continuation")
        return session.request(
            "construct",
            {
                **data,
                "controls": {
                    "topology": "query-calibrated-gp-root",
                    "source": recipe,
                    "locations": data.get("locations"),
                },
                "construction_seed": 0,
            },
        )
    if action == "gp_root_calibrate":
        if recipe["version"] != "gp-root-dag-v1":
            raise ValueError(
                "Open the original GP-root dataset to fit a new independent calibration"
            )
        return session.request(
            "construct",
            {
                **data,
                "controls": {
                    "topology": "calibrate-gp-root",
                    "source": original,
                    "calibration_rows": data["calibration_rows"],
                    "calibration_seed": data["calibration_seed"],
                    "calibration_task_id": data["calibration_task_id"],
                    "calibration_locations": data.get("calibration_locations"),
                },
                "construction_seed": 0,
                "seed": record["seed"],
                "task_id": record["task_id"],
            },
        )
    if action == "gp_root_replay":
        task, archive, run = run_saved_document(
            session.output_dir,
            {
                "recipe": recipe,
                "seed": record["seed"],
                "start": record["task_id"],
                "count": 1,
                "group_size": 1,
                "device": data.get("device", "cpu"),
                "causal_limits": asdict(caller_limits(data)),
                "max_seconds": data.get("max_seconds", 60),
                "max_bytes": data.get("max_bytes", 256 * 1024**2),
            },
        )
        session.task, session.archive, session.run = task, archive, run
        session.pair = session.edit = session.intervention = None
        return session.request("view", {})
    if action in ("gp_root_controls", "intervention_controls", "mechanism"):
        editable = replace(session.task, recipe_json=json.dumps({"recipe": original["base"]}))
        info = intervention_controls(editable)
        info.update(
            gp_root_dag=True,
            controls={
                "topology": "gp-root-dag",
                "request": original["source"]["request"],
                "base": original["base"],
                "bindings": original["bindings"],
            },
            calibrated=recipe["version"]
            in ("gp-root-calibration-v1", "gp-root-calibration-query-v1"),
            population_id=original["source"]["population_id"],
            notice=(
                "Retained GP columns feed the declared root inputs; the DAG defines targets. "
                "Column order and aliases are explicit. Replay keeps the exact population. "
                "Construct uses the episode seed and task identity for new GP draws; "
                "the generic construction seed does not change these authored equations. "
                "Apply edits to the construction draft, then construct a new population."
            ),
        )
        return info
    if action in ("curve", "sample_intervention_policy", "edit", "intervene"):
        raise ValueError(
            "GP-root graphs require retained-population-aware references and interventions; "
            "these workbench operations are pending integration"
        )
    return None
