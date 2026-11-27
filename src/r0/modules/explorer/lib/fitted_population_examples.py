"""Broad explicit fitting lifecycle choices, without conflating source fidelity."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def fitted_population_examples(explicit: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("dt", "extra_trees", "rf", "gbdt"):
        original = explicit[f"Fitted structural tree: {family}"]
        for population in ("support-only", "batch-refit-reference", "mixed-independent"):
            for task in ("regression", "classification"):
                selections: list[dict[str, Any]] = []
                for fit in original["fits"]:
                    plan = {
                        **fit["plan"],
                        "version": "population-fitted-tree-plan-v1",
                        "task": task,
                    }
                    if population != "mixed-independent":
                        del plan["rows"]
                    selections.append({"node": fit["node"], "plan": plan})
                result[f"Population fit: {population} {family} {task} (r0)"] = {
                    "topology": "population-fitted",
                    "population": population,
                    "skeleton": original["skeleton"],
                    "fits": selections,
                    "mixed": (
                        asdict(p.MixedFitPopulation())
                        if population == "mixed-independent"
                        else None
                    ),
                }
    return result
