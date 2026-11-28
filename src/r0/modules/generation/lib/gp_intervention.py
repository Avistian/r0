"""Admit complete GP comparison sources inside the guarded causal worker."""

import json
from typing import Any

from r0.modules import priors


def construct_gp_intervention(
    record: dict[str, Any], request: dict[str, Any], *, limits: priors.CausalLimits
) -> priors.GPRootIntervention:
    source = record["recipe"]
    current = source if source["version"] == "gp-root-dag-v1" else source["original"]
    return priors.construct_gp_root_intervention(
        {
            "source": source,
            "change": {
                "version": "causal-intervention-v1",
                "base": current["base"],
                "interventions": json.loads(request["interventions_json"]),
                "mode": request["mode"],
                "population": request["population"],
            },
        },
        seed=record["seed"],
        task_id=record["task_id"],
        limits=limits,
    )
