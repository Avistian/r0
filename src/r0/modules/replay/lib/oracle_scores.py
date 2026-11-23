"""Query-only diagnostic scores; undefined metrics retain an explicit reason."""

import math
from statistics import mean
from typing import cast

from r0.modules.episodes import Episode
from r0.modules.oracles import ClassificationPrediction, GPPrediction


def _auc(labels: tuple[bool, ...], scores: tuple[float, ...]) -> dict[str, object]:
    positives = [s for y, s in zip(labels, scores, strict=True) if y]
    negatives = [s for y, s in zip(labels, scores, strict=True) if not y]
    if not positives or not negatives:
        return {"value": None, "reason": "requires positive and negative query targets"}
    wins = sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives)
    return {"value": wins / (len(positives) * len(negatives)), "reason": None}


def oracle_scores(
    episode: Episode, prediction: ClassificationPrediction | GPPrediction
) -> dict[str, object]:
    targets = episode.query_targets
    if isinstance(prediction, ClassificationPrediction):
        if prediction.classes != episode.classes or len(prediction.probabilities) != len(targets):
            raise ValueError("classification prediction must align with scored episode")
        brier: list[float] = []
        losses: list[float] = []
        for target, probabilities in zip(targets, prediction.probabilities, strict=True):
            if (
                len(probabilities) != len(episode.classes)
                or any(not math.isfinite(p) or p <= 0 or p > 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12)
            ):
                raise ValueError("oracle probabilities must be positive and sum to one")
            brier.append(
                sum(
                    (p - (label == target)) ** 2
                    for label, p in zip(episode.classes, probabilities, strict=True)
                )
            )
            losses.append(-math.log(probabilities[episode.classes.index(str(target))]))
        components = {
            label: _auc(
                tuple(y == label for y in targets), tuple(p[i] for p in prediction.probabilities)
            )
            for i, label in enumerate(episode.classes)
        }
        if prediction.positive_class is not None:
            auc = components[prediction.positive_class]
        elif any(component["value"] is None for component in components.values()):
            auc = {"value": None, "reason": "macro AUC requires every declared class component"}
        else:
            auc = {
                "value": mean(cast(float, c["value"]) for c in components.values()),
                "reason": None,
            }
        return {
            "query_ids": tuple(r.id for r in episode.queries),
            "targets": targets,
            "brier": mean(brier),
            "brier_per_query": brier,
            "negative_log_probability": mean(losses),
            "negative_log_probability_per_query": losses,
            "roc_auc": auc,
            "roc_auc_per_class": components,
            "auc_convention": "binary declared positive class; multiclass macro one-versus-rest",
        }
    if episode.task_type != "regression" or len(prediction.mean) != len(targets):
        raise ValueError("GP prediction must align with regression episode")
    squared: list[float] = []
    densities: list[float | None] = []
    for y, m, variance in zip(
        targets, prediction.mean, prediction.observation_variance, strict=True
    ):
        if not math.isfinite(m) or not math.isfinite(variance) or variance < 0:
            raise ValueError("GP scoring requires finite means and nonnegative variances")
        residual = float(y) - m
        error = residual * residual
        squared.append(error)
        nll = (
            None
            if variance == 0
            else 0.5 * (math.log(2 * math.pi) + math.log(variance) + error / variance)
        )
        densities.append(nll)
    if any(not math.isfinite(v) for v in squared + [v for v in densities if v is not None]):
        raise ValueError("diagnostic score exceeds finite numeric range")
    density = (
        {"value": None, "reason": "Gaussian density undefined for zero observation variance"}
        if any(v is None for v in densities)
        else {"value": mean(v for v in densities if v is not None), "reason": None}
    )
    return {
        "query_ids": tuple(r.id for r in episode.queries),
        "targets": targets,
        "mse": mean(squared),
        "squared_error_per_query": squared,
        "gaussian_negative_log_density": density,
        "negative_log_density_per_query": densities,
    }
