"""Export the pinned backend's numeric vector trees without estimator retention."""

import json
from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.xgboost_vector import FrozenXGBoostVector, XGBoostVectorTree


def export_xgboost_vector(raw: bytes | bytearray) -> FrozenXGBoostVector:
    """Read bounded native save_raw(JSON), never unpickle or load a backend model.

    This boundary supports numeric CPU hist vector leaves from 3.4.1. Categorical
    splits and scalar-leaf layouts fail explicitly instead of silently changing
    their routing or treating scalar forests as joint vector trees.
    """
    if type(raw) not in (bytes, bytearray) or len(raw) > 16_000_000:
        raise ValueError("XGBoost export must be bounded native JSON bytes")
    try:
        data: Any = json.loads(raw)
        if data["version"] != [3, 4, 1]:
            raise ValueError("XGBoost export requires pinned backend 3.4.1")
        learner = data["learner"]
        params = learner["learner_model_param"]
        booster = learner["gradient_booster"]
        if booster["name"] != "gbtree":
            raise ValueError("only additive gbtree export is supported")
        model = booster["model"]
        native_trees = _list(model["trees"])
        if (
            not 1 <= len(native_trees) <= 32
            or model["tree_info"] != [0] * len(native_trees)
            or model["iteration_indptr"] != list(range(len(native_trees) + 1))
            or model["gbtree_model_param"]["num_parallel_tree"] != "1"
        ):
            raise ValueError("expected one joint vector tree per boosting round")
        width_in = int(params["num_feature"])
        objective = learner["objective"]["name"]
        base = _list(json.loads(params["base_score"]))
        if not 1 <= len(base) <= 100:
            raise ValueError("invalid vector base score width")
        initial = np.asarray(base, dtype=np.float32)
        if initial.ndim != 1 or not 1 <= len(initial) <= 100:
            raise ValueError("invalid vector base score")
        if objective == "binary:logistic":
            if not ((initial > 0) & (initial < 1)).all():
                raise ValueError("logistic initial probabilities must be interior")
            initial = -np.log(1 / initial - 1)
        trees = tuple(_tree(t, width_in, len(initial)) for t in native_trees)
        return FrozenXGBoostVector(width_in, tuple(map(float, initial)), trees, objective)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("malformed native XGBoost vector export") from exc


def _tree(t: Any, input_width: int, width: int) -> XGBoostVectorTree:
    params = t["tree_param"]
    n = int(params["num_nodes"])
    if (
        not 1 <= n <= 511
        or int(params["size_leaf_vector"]) != width
        or width < 2
        or int(params["num_feature"]) != input_width
        or params["num_deleted"] != "0"
    ):
        raise ValueError("expected bounded vector-leaf tree geometry")
    names = (
        "left_children",
        "right_children",
        "split_indices",
        "split_conditions",
        "default_left",
        "split_type",
    )
    if any(type(t[k]) is not list or len(t[k]) != n for k in names):
        raise ValueError("native tree arrays differ from node count")
    if any(
        t[k] for k in ("categories", "categories_nodes", "categories_segments", "categories_sizes")
    ):
        raise ValueError("categorical backend routing requires a separate supported exporter")
    if any(type(v) is not int or v != 0 for v in t["split_type"]):
        raise ValueError("only numerical splits are supported")
    leaves = _list(t["leaf_weights"])
    if len(leaves) > n * width:
        raise ValueError("invalid vector leaf storage")
    features: list[int] = []
    thresholds: list[float] = []
    left: list[int] = []
    right: list[int] = []
    defaults: list[bool] = []
    values: list[tuple[float, ...]] = []
    for i in range(n):
        child_left, r = t["left_children"][i], t["right_children"][i]
        if type(child_left) is not int or type(r) is not int:
            raise ValueError("native child indices must be integers")
        leaf = child_left == -1
        f = t["split_indices"][i]
        default = t["default_left"][i]
        if type(f) is not int or type(default) is not int or default not in (0, 1):
            raise ValueError("invalid native feature or missing direction")
        if leaf and not 0 <= r < len(leaves) // width:
            raise ValueError("invalid vector leaf offset")
        features.append(-1 if leaf else f)
        thresholds.append(0.0 if leaf else float(np.float32(t["split_conditions"][i])))
        left.append(-1 if leaf else child_left)
        right.append(-1 if leaf else r)
        defaults.append(bool(default))
        values.append(
            tuple(float(np.float32(v)) for v in leaves[r * width : (r + 1) * width])
            if leaf
            else (0.0,) * width
        )
    return XGBoostVectorTree(
        tuple(features),
        tuple(thresholds),
        tuple(left),
        tuple(right),
        tuple(defaults),
        tuple(values),
    )


def _list(value: object) -> list[Any]:
    if type(value) is not list:
        raise ValueError("native array must be a JSON list")
    return cast(list[Any], value)
