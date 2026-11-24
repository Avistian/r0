"""Frozen baseline recipes and context-only preprocessing, independent of training."""

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contracts import validate_view
from .storage import Document

Array = NDArray[np.float64]
FitSpy = Callable[[str, tuple[str, ...]], None]


class Unsupported(ValueError):
    """A declared recipe cannot represent this context."""


def recipe(name: str, task: str) -> Document:
    regression = task == "regression"
    if name in ("simple", "learner_fixture", "tabpfn_fixture"):
        return {
            "name": name,
            "algorithm": "arithmetic_mean" if regression else "unsmoothed_frequencies",
            "preprocessing": "none",
            "probability_conversion": "context_class_frequencies",
            "weights": None,
            "fixture_substitute": name.endswith("_fixture"),
        }
    if name == "linear":
        return {
            "name": name,
            "algorithm": "Ridge" if regression else "LogisticRegression",
            "parameters": {"alpha": 1.0, "fit_intercept": True, "solver": "svd"}
            if regression
            else {
                "C": 1.0,
                "fit_intercept": True,
                "solver": "lbfgs",
                "max_iter": 1000,
                "tol": 1e-4,
                "class_weight": None,
                "l1_ratio": 0.0,
            },
            "preprocessing": "context-median-zero-empty-indicators-standardize-onehot-v1",
            "probability_conversion": "sklearn_predict_proba_declared_class_order",
            "weights": None,
        }
    if name == "boosted_tree":
        return {
            "name": name,
            "algorithm": "HistGradientBoostingRegressor"
            if regression
            else "HistGradientBoostingClassifier",
            "parameters": {
                "max_iter": 100,
                "learning_rate": 0.1,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
                "early_stopping": False,
                "random_state": 0,
            },
            "preprocessing": "native-categories-numeric-nan-missing-unseen-indicators-v1",
            "probability_conversion": "sklearn_predict_proba_declared_class_order",
            "weights": None,
        }
    raise Unsupported(f"unknown predictor {name}")


def _preprocess(view: Document, tree: bool, spy: FitSpy | None) -> tuple[Array, Array, Document]:
    context: list[list[Any]] = view["context_values"]
    query: list[list[Any]] = view["query_values"]
    ids = tuple(view["context_ids"])
    if spy is not None:
        spy("preprocessing", ids)
    context_columns: list[Array] = []
    query_columns: list[Array] = []
    evidence: list[Document] = []
    categorical: list[bool] = []
    for j, feature in enumerate(view["features"]):
        train = [r[j] for r in context]
        test = [r[j] for r in query]
        if feature["kind"] == "categorical":
            vocabulary = sorted({v for v in train if v is not None})
            if tree and len(vocabulary) > 255:
                raise Unsupported("more than 255 context categories")
            mapping = {v: i for i, v in enumerate(vocabulary)}

            def encode(
                values: list[Any],
                mapping: dict[str, int] = mapping,
                vocabulary: list[str] = vocabulary,
            ) -> Array:
                if tree:
                    return np.array(
                        [
                            [
                                mapping.get(v, np.nan),
                                float(v is None),
                                float(v is not None and v not in mapping),
                            ]
                            for v in values
                        ],
                        dtype=np.float64,
                    )
                return np.array(
                    [
                        [float(v == c) for c in vocabulary]
                        + [float(v is None), float(v is not None and v not in mapping)]
                        for v in values
                    ],
                    dtype=np.float64,
                )

            a, b = encode(train), encode(test)
            categorical.extend([True, False, False] if tree else [False] * a.shape[1])
            evidence.append(
                {
                    "feature": feature["name"],
                    "categories": vocabulary,
                    "missing": "separate_indicator",
                    "unseen": "separate_indicator",
                }
            )
        else:
            a = np.array([np.nan if v is None else v for v in train], dtype=np.float64)
            b = np.array([np.nan if v is None else v for v in test], dtype=np.float64)
            missing_a, missing_b = np.isnan(a), np.isnan(b)
            median = float(np.median(a[~missing_a])) if (~missing_a).any() else 0.0
            mean, scale = 0.0, 1.0
            if not tree:
                a, b = np.where(missing_a, median, a), np.where(missing_b, median, b)
                mean, scale = float(a.mean()), float(a.std())
                scale = scale if scale else 1.0
                a, b = (a - mean) / scale, (b - mean) / scale
            a, b = np.column_stack((a, missing_a)), np.column_stack((b, missing_b))
            categorical.extend([False, False])
            evidence.append(
                {
                    "feature": feature["name"],
                    "median": median,
                    "mean": mean,
                    "scale": scale,
                    "numeric_missing_retained": tree,
                }
            )
        context_columns.append(a)
        query_columns.append(b)
    return (
        np.column_stack(context_columns),
        np.column_stack(query_columns),
        {"fit_ids": list(ids), "columns": evidence, "categorical_features": categorical},
    )


def fit_predictor(
    view: Document, name: str, *, spy: FitSpy | None = None
) -> tuple[Callable[[list[int]], Array], Document]:
    """Fit only on context; query labels are forbidden in the input contract."""
    validate_view(view)
    ids = view["context_ids"]
    query_ids = view["query_ids"]
    spec = recipe(name, view["task"])
    regression = view["task"] == "regression"
    classes = view["classes"]
    targets: Array = np.array(
        view["context_targets"]
        if regression
        else [classes.index(t) for t in view["context_targets"]],
        dtype=np.float64,
    )
    if not regression and set(targets) != set(range(len(classes))):
        raise ValueError("context does not cover declared classes")
    if name in ("simple", "learner_fixture", "tabpfn_fixture"):
        if spy is not None:
            spy("estimator", tuple(ids))
        result = (
            np.full(len(query_ids), targets.mean())
            if regression
            else np.tile(
                [float(np.count_nonzero(targets == i)) / len(targets) for i in range(len(classes))],
                (len(query_ids), 1),
            )
        )
        return lambda indices: result[indices], {
            "recipe": spec,
            "fit_ids": ids,
            "resolved_parameters": {},
            "preprocessing": {"fit_ids": [], "columns": []},
        }
    tree = name == "boosted_tree"
    x, q, preprocessing = _preprocess(view, tree, spy)
    library = importlib.import_module("sklearn.ensemble" if tree else "sklearn.linear_model")
    constructor: Any = getattr(library, spec["algorithm"])
    parameters = dict(spec["parameters"])
    if tree:
        parameters["categorical_features"] = preprocessing["categorical_features"]
    model: Any = constructor(**parameters)
    if spy is not None:
        spy("estimator", tuple(ids))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, targets)
        if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
            raise ValueError("convergence failure; no automatic retuning")

    def infer(indices: list[int]) -> Array:
        result = np.asarray(
            model.predict(q[indices]) if regression else model.predict_proba(q[indices]),
            dtype=np.float64,
        )
        if not regression:
            result = result[:, [list(model.classes_).index(i) for i in range(len(classes))]]
        return result

    return infer, {
        "recipe": spec,
        "resolved_parameters": model.get_params(),
        "fit_ids": ids,
        "preprocessing": preprocessing,
        "warnings": [str(w.message) for w in caught],
    }


def predict(view: Document, name: str, *, spy: FitSpy | None = None) -> tuple[Array, Document]:
    """Fit on context and predict the ordered query rows."""
    infer, evidence = fit_predictor(view, name, spy=spy)
    return infer(list(range(len(view["query_ids"])))), evidence
