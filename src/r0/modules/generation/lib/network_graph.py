"""Saved original-row network trace under the bounded process guard."""

import json
from pathlib import Path
from typing import Any

from r0.modules import priors
from r0.modules.generation.lib.loading import load_group


def network_graph(data: dict[str, Any]) -> dict[str, Any]:
    population = data.get("population", "query")
    if population not in ("query", "fit"):
        raise ValueError("network trace population must be query or fit")
    maximum = data.get("max_bytes", 256 * 1024**2)
    if type(maximum) is not int or not 1 <= maximum <= 256 * 1024**2:
        raise ValueError("network trace byte cap must be an integer in 1..268435456")
    path = Path(data["archive"])
    if path.stat().st_size > maximum:
        raise ValueError("network trace archive exceeds byte cap")
    for task in load_group(path):
        record = json.loads(task.recipe_json)
        if record["task_id"] != data["task_id"]:
            continue
        if task.episode_id != data["episode_id"] or record["recipe"].get("version") not in (
            "recursive-population-graph-v1",
            "recursive-population-graph-intervention-v1",
            "functional-causal-v1",
            "functional-causal-v2",
            "native-causal-v1",
            "native-vector-causal-v1",
            "converted-causal-v1",
            "sampled-intervention-policy-v1",
            "sampled-typed-intervention-policy-v1",
            "prototype-batch-network-realized-v1",
            "gp-root-dag-v1",
            "gp-root-calibration-v1",
            "gp-column-recipe-v1",
            "gp-column-target-recipe-v1",
            "gp-column-target-link-v1",
            "network-observation-recipe-v1",
            "standalone-network-innovations-v1",
            "standalone-network-v1",
            "network-input-view-v1",
            "network-input-view-v2",
        ):
            raise ValueError("network trace archive identity differs")
        explain = (
            priors.explain_gp_calibration_fit if population == "fit" else priors.explain_generation
        )
        graph = explain(
            task,
            data["indices"],
            component=data.get("component"),
            column_start=data.get("column_start", 0),
            limits=priors.CausalLimits(**data.get("causal_limits", {})),
        )
        if len(json.dumps(graph, allow_nan=False).encode()) > maximum:
            raise ValueError("network trace exceeds output byte cap")
        return {"status": "VALIDATED", "graph": graph}
    raise ValueError("network trace task identity was not found")
