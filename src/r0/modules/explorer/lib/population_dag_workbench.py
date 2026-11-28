"""Population-specific explorer actions preserve row-owned archives and bounded replay."""

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.interventions import intervention_controls
from r0.modules.explorer.lib.saved import run_saved_document

if TYPE_CHECKING:
    from r0.modules.explorer.lib.session import ExplorerSession


POPULATION_VERSION = "recursive-population-graph-v1"
CALIBRATED_VERSIONS = ("calibrated-population-graph-v1", "calibrated-population-query-v1")


def population_dag_action(
    session: "ExplorerSession", action: str, data: dict[str, Any]
) -> dict[str, Any] | None:
    if session.task is None:
        return None
    record = json.loads(session.task.recipe_json)
    stored = record["recipe"]
    if action == "curve" and stored.get("version") in (
        POPULATION_VERSION,
        "recursive-population-graph-intervention-v1",
    ):
        from r0.modules.explorer.lib.population_reference import population_reference_action

        return population_reference_action(session, data)
    calibrated = stored.get("version") in CALIBRATED_VERSIONS
    if stored.get("version") != POPULATION_VERSION and not calibrated:
        return None
    original = stored["original"] if calibrated else stored
    if action in ("population_calibrate", "population_query"):
        if action == "population_calibrate":
            if calibrated:
                raise ValueError(
                    "Fit a new raw population explicitly; retained calibration is immutable"
                )
            controls = {
                "topology": "calibrate-population",
                "source": stored,
                "calibration_rows": data.get("calibration_rows", 64),
                "calibration_seed": data.get("calibration_seed"),
                "calibration_task_id": data.get("calibration_task_id"),
            }
            seed, task_id = record["seed"], record["task_id"]
            construction_seed = data.get("calibration_seed")
        else:
            if not calibrated:
                raise ValueError("Create an independent calibration before fixed-fit Next")
            controls = {
                "topology": "calibrated-population-query",
                "source": stored,
                "engine": data.get(
                    "engine", original["bindings"][0]["sample"]["initial"]["engine"]
                ),
            }
            seed, task_id = data.get("seed"), data.get("task_id")
            construction_seed = data.get("construction_seed", seed)
        return session.request(
            "construct",
            {
                **data,
                "controls": controls,
                "construction_seed": construction_seed,
                "seed": seed,
                "task_id": task_id,
            },
        )
    if action == "population_dag_replay":
        task, archive, run = run_saved_document(
            session.output_dir,
            {
                "recipe": stored,
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
    if action in ("intervention_controls", "mechanism", "population_dag_controls"):
        base = original["request"]["base"]
        editable = replace(session.task, recipe_json=json.dumps({"recipe": base}))
        info = intervention_controls(editable)
        for binding in original["request"]["replacements"]:
            info["nodes"][binding["node"]]["population_forest"] = binding["forest"]
        calibration = None
        if calibrated:
            root = stored
            while root["version"] == "calibrated-population-query-v1":
                root = root["source"]
            fit = root["calibration_population"]
            calibration = {
                "query_fit": False,
                "constants": root["calibration"],
                "fit": {
                    "seed": fit["seed"],
                    "task_id": fit["task_id"],
                    "rows": fit["request"]["base"]["task"]["rows"],
                },
                "source_version": stored["version"],
            }
        info.update(
            calibration=calibration,
            population_dag=True,
            original_version=stored["version"],
            controls={
                "topology": "recursive-population-dag",
                "request": original["request"],
                "engine": original["bindings"][0]["sample"]["initial"]["engine"],
            },
            notice=(
                "Each original row owns its forest at each selected node. Matched comparisons "
                "retain these forests; fresh comparisons draw new row forests with the explicit "
                "population key. Hard cuts bypass selected mechanisms, soft replacements disable "
                "that node's population forest. Stored pipelines retain calibration constants; "
                "no query fit. Explicit independent calibration retains fixed constants across "
                "Next and matched/fresh comparisons. Changing laws requires a new explicit fit."
            ),
        )
        return info
    if calibrated and action == "pair":
        raise ValueError(
            "Use fixed-calibration causal comparisons; "
            "task-only adapter comparisons remain required"
        )
    if action in ("curve", "sample_intervention_policy", "edit"):
        raise ValueError(
            "Population DAGs require original-row traces and explicit retained interventions; "
            "sampled policies and linked mechanism editing remain required "
            "follow-ons. Use construction controls for a new population."
        )
    return None
