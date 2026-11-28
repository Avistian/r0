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


def population_dag_action(
    session: "ExplorerSession", action: str, data: dict[str, Any]
) -> dict[str, Any] | None:
    if session.task is None:
        return None
    record = json.loads(session.task.recipe_json)
    original = record["recipe"]
    if original.get("version") != POPULATION_VERSION:
        return None
    if action == "population_dag_replay":
        task, archive, run = run_saved_document(
            session.output_dir,
            {
                "recipe": original,
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
        info.update(
            population_dag=True,
            original_version=POPULATION_VERSION,
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
                "no query fit. Independent target calibration remains a required follow-on."
            ),
        )
        return info
    if action in ("curve", "sample_intervention_policy", "edit"):
        raise ValueError(
            "Population DAGs require original-row traces and explicit retained interventions; "
            "reference curves, sampled policies and linked mechanism editing remain required "
            "follow-ons. Use construction controls for a new population."
        )
    return None
