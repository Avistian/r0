"""CPU sklearn fitting/export boundary; no estimator survives this operation."""

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r0.modules.priors.lib.fitted_backend_contract import fitted_backend_contract
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.fitted_tree_records import FittedSplitTable, FittedTreeOutput


def _table(estimator: Any) -> FittedSplitTable:
    tree = estimator.tree_
    return FittedSplitTable(
        tuple(map(int, tree.feature)),
        tuple(map(float, tree.threshold)),
        tuple(map(int, tree.children_left)),
        tuple(map(int, tree.children_right)),
        tuple(tuple(map(float, v.reshape(-1))) for v in tree.value),
    )


def fit_backend(
    plan: FittedTreePlan,
    inputs: NDArray[np.float64],
    targets: NDArray[np.float64],
    seeds: tuple[int, ...],
) -> tuple[tuple[FittedTreeOutput, ...], tuple[str, ...], str]:
    import sklearn  # pyright: ignore[reportMissingTypeStubs]
    from sklearn.ensemble import (  # pyright: ignore[reportMissingTypeStubs]
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # pyright: ignore[reportMissingTypeStubs]

    classifier = plan.task == "classification"
    constructors: dict[str, Any] = {
        "dt": DecisionTreeClassifier if classifier else DecisionTreeRegressor,
        "extra_trees": ExtraTreesClassifier if classifier else ExtraTreesRegressor,
        "rf": RandomForestClassifier if classifier else RandomForestRegressor,
        "gbdt": GradientBoostingClassifier if classifier else GradientBoostingRegressor,
    }
    outputs: list[FittedTreeOutput] = []
    parameters: list[str] = []
    for channel, seed in enumerate(seeds):
        kwargs = fitted_backend_contract(plan, seed)
        estimator = constructors[plan.family](**kwargs)
        estimator.fit(inputs, targets[:, channel])
        parameters.append(
            json.dumps(estimator.get_params(deep=False), sort_keys=True, allow_nan=False)
        )
        classes = tuple(map(int, estimator.classes_)) if classifier else ()
        if plan.family == "gbdt":
            # Default Dummy initializer is constant. Retain its actual version-specific
            # raw scores rather than assuming class priors use an un-clipped log.
            initial = tuple(map(float, estimator._raw_predict_init(inputs[:1])[0]))
            tables = tuple(_table(tree) for tree in estimator.estimators_.flat)
            combiner = (
                "boost_softmax"
                if len(classes) > 2
                else "boost_sigmoid"
                if classifier
                else "boost_identity"
            )
            outputs.append(FittedTreeOutput(tables, combiner, classes, initial, plan.learning_rate))
        else:
            trees = (estimator,) if plan.family == "dt" else estimator.estimators_
            combiner = plan.classification_rule if classifier else "mean"
            outputs.append(
                FittedTreeOutput(tuple(_table(tree) for tree in trees), combiner, classes)
            )
    return tuple(outputs), tuple(parameters), sklearn.__version__
