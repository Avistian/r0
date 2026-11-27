"""Sampled laws crossed with every supported lifecycle, backend and task contract."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p


def _laws(
    family: str, task: str, population: str, pinned: bool
) -> tuple[tuple[str, p.FittedParameterLaw], ...]:
    laws: list[tuple[str, p.FittedParameterLaw]] = []
    if not pinned:
        laws.append(("depth", p.FittedParameterLaw(kind="integer-uniform", low=1, high=5)))
        if family != "dt":
            laws.append(
                ("estimators", p.FittedParameterLaw(kind="categorical", values=(1, 2, 4, 8)))
            )
    if population == "mixed-independent":
        laws.append(("rows", p.FittedParameterLaw(kind="categorical", values=(32, 64, 128))))
    if family == "dt":
        laws.append(
            ("splitter", p.FittedParameterLaw(kind="categorical", values=("best", "random")))
        )
    if family == "gbdt":
        laws.append(("learning_rate", p.FittedParameterLaw(kind="log-uniform", low=0.03, high=0.8)))
    if task == "classification":
        laws.append(("classes", p.FittedParameterLaw(kind="categorical", values=(2, 3, 5))))
        if family != "gbdt":
            laws.append(
                (
                    "classification_rule",
                    p.FittedParameterLaw(
                        kind="categorical", values=("probability_mean", "majority")
                    ),
                )
            )
    return tuple(laws)


def sampled_population_examples(explicit: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for population in ("support-only", "batch-refit-reference", "mixed-independent"):
        for family in ("dt", "extra_trees", "rf", "gbdt", "backend-mixture"):
            original = explicit[
                f"Fitted structural tree: {'dt' if family == 'backend-mixture' else family}"
            ]
            for task in ("regression", "classification"):
                for pinned in (True,) if family == "backend-mixture" else (False, True):
                    selections: list[dict[str, Any]] = []
                    for fit in original["fits"]:
                        base_fields: dict[str, Any] = {
                            **fit["plan"],
                            "task": task,
                            "version": "population-fitted-tree-plan-v1",
                        }
                        base = p.PopulationTreePlan(**base_fields)
                        laws = (
                            _laws(family, task, population, pinned)
                            if family != "backend-mixture"
                            else (
                                (
                                    "family",
                                    p.FittedParameterLaw(
                                        kind="categorical",
                                        values=("dt", "extra_trees", "rf", "gbdt"),
                                        probabilities=(0.1, 0.2, 0.3, 0.4),
                                    ),
                                ),
                            )
                        )
                        prior = asdict(
                            p.PopulationTreeParameterPrior(
                                base=base,
                                population=population,
                                laws=laws,
                                profile="tabicl-pinned-depth-count-v1"
                                if pinned
                                else "r0-explicit-parameter-laws-v1",
                            )
                        )
                        if population != "mixed-independent":
                            del prior["base"]["rows"]
                        selections.append({"node": fit["node"], "prior": prior})
                    profile = "pinned depth/count" if pinned else "explicit laws"
                    result[f"Sampled population: {population} {family} {task}, {profile} (r0)"] = {
                        "topology": "sampled-population-fitted",
                        "population": population,
                        "skeleton": original["skeleton"],
                        "fits": selections,
                        "mixed": asdict(p.MixedFitPopulation())
                        if population == "mixed-independent"
                        else None,
                    }
    return result
