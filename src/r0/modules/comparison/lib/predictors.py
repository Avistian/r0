"""Frozen baseline recipes and context-only preprocessing, independent of training."""

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .contracts import Array
from .storage import Document

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


def _preprocess(
    context: Document, tree: bool, spy: FitSpy | None
) -> tuple[Array, Callable[[list[list[Any]]], Array], Document]:
    """Fit column transforms on context; queries only apply these frozen transforms."""
    if spy:
        spy("preprocessing", tuple(context["row_ids"]))
    transforms: list[Callable[[list[Any]], Array]] = []
    evidence: list[Document] = []
    categorical: list[bool] = []
    for j, feature in enumerate(context["features"]):
        train = [row[j] for row in context["values"]]
        if feature["kind"] == "categorical":
            vocabulary = sorted({v for v in train if v is not None})
            if tree and len(vocabulary) > 255:
                raise Unsupported("more than 255 context categories")
            mapping = {v: i for i, v in enumerate(vocabulary)}

            def encode_category(
                values: list[Any],
                mapping: dict[str, int] = mapping,
                vocabulary: list[str] = vocabulary,
            ) -> Array:
                return np.array(
                    [
                        ([mapping.get(v, np.nan)] if tree else [float(v == c) for c in vocabulary])
                        + [float(v is None), float(v is not None and v not in mapping)]
                        for v in values
                    ],
                    dtype=np.float64,
                )

            transforms.append(encode_category)
            categorical.extend([True, False, False] if tree else [False] * (len(vocabulary) + 2))
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
            missing = np.isnan(a)
            median = float(np.median(a[~missing])) if (~missing).any() else 0.0
            mean, scale = 0.0, 1.0
            if not tree:
                a = np.where(missing, median, a)
                mean, scale = float(a.mean()), float(a.std()) or 1.0

            def encode_numeric(
                values: list[Any], median: float = median, mean: float = mean, scale: float = scale
            ) -> Array:
                a = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
                missing = np.isnan(a)
                if not tree:
                    a = (np.where(missing, median, a) - mean) / scale
                return np.column_stack((a, missing))

            transforms.append(encode_numeric)
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

    def transform(rows: list[list[Any]]) -> Array:
        return np.column_stack([fn([row[j] for row in rows]) for j, fn in enumerate(transforms)])

    return (
        transform(context["values"]),
        transform,
        {
            "fit_ids": context["row_ids"],
            "columns": evidence,
            "categorical_features": categorical,
        },
    )


class Baseline:
    """The frozen simple, linear and tree recipes implement the same predictor interface."""

    def __init__(self, name: str, spy: FitSpy | None = None) -> None:
        self.name, self.spy = name, spy
        self.context: Document = {}
        self.evidence: Document = {}
        self.model: Any = None
        self.simple: Array = np.empty(0)
        self.transform: Callable[[list[list[Any]]], Array] | None = None

    def fit(self, context: Document) -> None:
        self.context = context
        spec = recipe(self.name, context["task"])
        regression = context["task"] == "regression"
        classes = context["classes"]
        targets: Array = np.array(
            context["targets"] if regression else [classes.index(t) for t in context["targets"]],
            dtype=np.float64,
        )
        if not regression and set(targets) != set(range(len(classes))):
            raise ValueError("context does not cover declared classes")
        self.evidence = {
            "recipe": spec,
            "fit_ids": context["row_ids"],
            "resolved_parameters": {},
            "preprocessing": {"fit_ids": [], "columns": []},
        }
        if self.name in ("simple", "learner_fixture", "tabpfn_fixture"):
            self.simple = np.array(
                [targets.mean()]
                if regression
                else [np.count_nonzero(targets == i) / len(targets) for i in range(len(classes))],
                dtype=np.float64,
            )
            if self.spy:
                self.spy("estimator", tuple(context["row_ids"]))
            return
        tree = self.name == "boosted_tree"
        x, self.transform, preprocessing = _preprocess(context, tree, self.spy)
        library = importlib.import_module("sklearn.ensemble" if tree else "sklearn.linear_model")
        constructor: Any = getattr(library, spec["algorithm"])
        parameters = dict(spec["parameters"])
        if tree:
            parameters["categorical_features"] = preprocessing["categorical_features"]
        self.model = constructor(**parameters)
        if self.spy:
            self.spy("estimator", tuple(context["row_ids"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model.fit(x, targets)
            if any(w.category.__name__ == "ConvergenceWarning" for w in caught):
                raise ValueError("convergence failure; no automatic retuning")
        self.evidence.update(
            resolved_parameters=self.model.get_params(),
            preprocessing=preprocessing,
            warnings=[str(w.message) for w in caught],
        )

    def predict(self, queries: Document) -> tuple[list[str], Array]:
        regression = self.context["task"] == "regression"
        ids = queries["row_ids"]
        if self.model is None:
            values = (
                np.full(len(ids), self.simple[0])
                if regression
                else np.tile(self.simple, (len(ids), 1))
            )
        else:
            if self.transform is None:
                raise ValueError("baseline not fitted")
            q = self.transform(queries["values"])
            values = np.asarray(
                self.model.predict(q) if regression else self.model.predict_proba(q),
                dtype=np.float64,
            )
            if not regression:
                values = values[
                    :,
                    [
                        list(self.model.classes_).index(i)
                        for i in range(len(self.context["classes"]))
                    ],
                ]
        return ids, values
