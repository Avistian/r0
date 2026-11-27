"""Deterministic declared backend settings, independent of sklearn imports/refits."""

from typing import Any

from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan


def fitted_backend_contract(plan: FittedTreePlan, seed: int) -> dict[str, Any]:
    classifier = plan.task == "classification"
    settings: dict[str, Any] = {
        "random_state": seed,
        "max_depth": plan.depth,
    }
    if plan.family != "gbdt":
        settings["criterion"] = "gini" if classifier else "squared_error"
    if plan.family == "dt":
        settings["splitter"] = plan.splitter
    else:
        settings["n_estimators"] = plan.estimators
    if plan.family in ("rf", "extra_trees"):
        settings.update(
            n_jobs=1, bootstrap=plan.family == "rf", max_features="sqrt" if classifier else 1.0
        )
    if plan.family == "gbdt":
        settings.update(
            learning_rate=plan.learning_rate,
            loss="log_loss" if classifier else "squared_error",
            init=None,
        )
    return settings
