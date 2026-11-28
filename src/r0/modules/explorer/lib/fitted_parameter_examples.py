"""Source-law profiles and broad compatible r0 fit options, all separately labelled."""

from dataclasses import asdict, replace
from typing import Any

from r0.modules import priors as p


def fitted_parameter_examples(explicit: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in ("dt", "extra_trees", "rf", "gbdt"):
        original = explicit[f"Fitted structural tree: {family}"]
        for task in ("regression", "classification"):
            laws = [
                ("depth", p.FittedParameterLaw(kind="integer-uniform", low=1, high=5)),
                ("rows", p.FittedParameterLaw(kind="categorical", values=(32, 64, 128))),
            ]
            if family == "dt":
                laws.append(
                    (
                        "splitter",
                        p.FittedParameterLaw(kind="categorical", values=("best", "random")),
                    )
                )
            else:
                laws.append(
                    ("estimators", p.FittedParameterLaw(kind="categorical", values=(1, 2, 4, 8)))
                )
            if family == "gbdt":
                laws.append(
                    ("learning_rate", p.FittedParameterLaw(kind="log-uniform", low=0.03, high=0.8))
                )
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
            selections: list[dict[str, Any]] = []
            for fit in original["fits"]:
                base = replace(p.FittedTreePlan(**fit["plan"]), task=task)
                selections.append(
                    {
                        "node": fit["node"],
                        "prior": asdict(p.FittedTreeParameterPrior(base, tuple(laws))),
                    }
                )
            result[f"Sampled fitted parameters: {family} {task} (r0)"] = {
                "topology": "sampled-fitted",
                "skeleton": original["skeleton"],
                "fits": selections,
            }
        # Pin fidelity is solely the disclosed two parameter laws. Backend and
        # frozen independent calibration remain explicitly labelled adaptations.
        result[f"TabICL depth/count laws: {family} independent fit (r0)"] = {
            "topology": "sampled-fitted",
            "skeleton": original["skeleton"],
            "fits": [
                {
                    "node": fit["node"],
                    "prior": asdict(
                        p.FittedTreeParameterPrior(
                            p.FittedTreePlan(**fit["plan"]), profile="tabicl-pinned-depth-count-v1"
                        )
                    ),
                }
                for fit in original["fits"]
            ],
        }
    return result
