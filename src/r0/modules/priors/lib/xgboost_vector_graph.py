"""Bounded frozen archive and device-native vector-tree graph evaluation."""

from dataclasses import fields
from typing import Any

import torch

from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.xgboost_vector import FrozenXGBoostVector, XGBoostVectorTree


def decode_xgboost_vector(data: dict[str, Any]) -> FrozenXGBoostVector:
    raw = fitted_object(data, FrozenXGBoostVector)
    if set(raw) != {f.name for f in fields(FrozenXGBoostVector)}:
        raise ValueError("XGBoost frozen archive fields must be complete")
    raw["initial"] = fitted_array(raw["initial"], 100)
    trees: list[XGBoostVectorTree] = []
    for value in fitted_array(raw["trees"], 32):
        tree = fitted_object(value, XGBoostVectorTree)
        if set(tree) != {f.name for f in fields(XGBoostVectorTree)}:
            raise ValueError("XGBoost vector tree fields must be complete")
        for name in ("features", "thresholds", "left", "right", "default_left"):
            tree[name] = fitted_array(tree[name], 511)
        tree["values"] = tuple(fitted_array(row, 100) for row in fitted_array(tree["values"], 511))
        trees.append(XGBoostVectorTree(**tree))
    raw["trees"] = tuple(trees)
    return FrozenXGBoostVector(**raw)


def evaluate_xgboost_tensor(function: FrozenXGBoostVector, inputs: torch.Tensor) -> torch.Tensor:
    """Called only after shared function admission and finite-input validation."""
    values = inputs.to(torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("XGBoost graph input overflow during float32 routing")
    device = inputs.device
    scores = (
        torch.tensor(function.initial, dtype=torch.float32, device=device)
        .expand(len(inputs), -1)
        .clone()
    )
    rows = torch.arange(len(inputs), device=device)
    for tree in function.trees:
        indices = torch.zeros(len(inputs), dtype=torch.long, device=device)
        features = torch.tensor(tree.features, dtype=torch.long, device=device)
        left = torch.tensor(tree.left, dtype=torch.long, device=device)
        right = torch.tensor(tree.right, dtype=torch.long, device=device)
        thresholds = torch.tensor(tree.thresholds, dtype=torch.float32, device=device)
        for _ in range(8):
            selected = features[indices]
            go_left = values[rows, selected.clamp_min(0)] < thresholds[indices]
            indices = torch.where(
                selected >= 0, torch.where(go_left, left[indices], right[indices]), indices
            )
        scores += torch.tensor(tree.values, dtype=torch.float32, device=device)[indices]
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite XGBoost graph margin; no retry")
    if function.objective == "binary:logistic":
        scores = scores.sigmoid()
    elif function.objective == "multi:softprob":
        scores = scores.softmax(dim=1)
    return scores.to(torch.float64)
