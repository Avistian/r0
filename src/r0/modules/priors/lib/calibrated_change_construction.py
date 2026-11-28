"""Admit a complete fixed-calibration comparison before decoding its retained baseline."""

from typing import Any

from r0.modules.priors.lib.calibrated_population_intervention import intervene_calibrated_population
from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
    preflight_calibrated_intervention,
)
from r0.modules.priors.lib.calibrated_query_decoding import calibrated_source_from_dict
from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata


def construct_calibrated_population_change(
    source: dict[str, Any], interventions: Any, *, mode: str, population: int, limits: CausalLimits
):
    admit_function_metadata((source, interventions), limits)
    try:
        current = source["original"]
        change = {
            "version": "causal-intervention-v1",
            "base": current["request"]["base"],
            "interventions": interventions,
            "mode": mode,
            "population": population,
        }
        preview = {
            "version": "calibrated-population-intervention-v1",
            "base": source,
            "original": {
                "version": "recursive-population-graph-intervention-v1",
                "base": current,
                "change": change,
                "samples": [b["sample"] for b in current["bindings"]],
                "inputs": [(b["node"], b["inputs"]) for b in current["bindings"]],
            },
        }
        preflight_calibrated_intervention(preview, limits=limits)
        base = calibrated_source_from_dict(source, limits=limits)
        return intervene_calibrated_population(base, intervention_from_dict(change, limits=limits))
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(
            "calibrated comparison requires complete source and change metadata"
        ) from exc
