"""Known-answer checks, separate from any benchmark or learned-model ranking."""

import math
from dataclasses import asdict
from typing import TypedDict, cast

from r0.modules.challenges.lib.cases import ChallengePair
from r0.modules.challenges.lib.predictors import reference_predictions, shortcut_predictions
from r0.modules.episodes import Episode, Target


class PredictionCheck(TypedDict):
    predictions: list[Target]
    matches: list[bool]
    absolute_differences: list[float | None]


class EpisodeCheck(TypedDict):
    episode: dict[str, object]
    predictor_view: dict[str, object]
    expected_answers: list[Target]
    reference: PredictionCheck
    shortcut: PredictionCheck


class PairReport(TypedDict):
    name: str
    family: str
    shortcut: str
    transformation: str
    derivation: str
    before: EpisodeCheck
    after: EpisodeCheck
    transformation_valid: bool
    status: str


def _check(episode: Episode, predictions: tuple[Target, ...]) -> PredictionCheck:
    if len(predictions) != len(episode.query_targets):
        raise ValueError("prediction count differs from query count")
    matches: list[bool] = []
    differences: list[float | None] = []
    for predicted, expected in zip(predictions, episode.query_targets, strict=True):
        if episode.task_type == "regression":
            difference = abs(float(predicted) - float(expected))
            matches.append(
                math.isclose(float(predicted), float(expected), abs_tol=1e-10, rel_tol=1e-10)
            )
            differences.append(difference)
        else:
            matches.append(predicted == expected)
            differences.append(None)
    return {
        "predictions": list(predictions),
        "matches": matches,
        "absolute_differences": differences,
    }


def _episode_check(episode: Episode, pair: ChallengePair) -> EpisodeCheck:
    if len(episode.context) > 12 or len(episode.queries) > 6 or len(episode.features) > 4:
        raise ValueError("challenge exceeds 12 context rows, 6 queries or 4 features")
    view = episode.predictor_view()
    return {
        "episode": asdict(episode),
        "predictor_view": asdict(view),
        "expected_answers": list(episode.query_targets),
        "reference": _check(episode, reference_predictions(view, pair.family)),
        "shortcut": _check(episode, shortcut_predictions(view, pair.shortcut)),
    }


def _transformation_valid(pair: ChallengePair) -> bool:
    a, b = pair.before, pair.after
    if (
        a.source_task_id != b.source_task_id
        or a.episode_id == b.episode_id
        or a.task_type != b.task_type
        or a.classes != b.classes
        or a.class_provenance != b.class_provenance
    ):
        return False
    if pair.name == "column-permutation":
        return (
            b.features == a.features[::-1]
            and b.context_targets == a.context_targets
            and b.query_targets == a.query_targets
            and tuple((r.id, r.values) for r in b.context + b.queries)
            == tuple((r.id, r.values[::-1]) for r in a.context + a.queries)
        )
    if pair.name == "context-remapping":
        mapping = dict(zip(a.context_targets, b.context_targets, strict=True))
        return (
            a.features == b.features
            and a.context == b.context
            and a.queries == b.queries
            and set(mapping) == set(a.classes)
            and set(mapping.values()) == set(a.classes)
            and all(
                mapping[x] == y for x, y in zip(a.context_targets, b.context_targets, strict=True)
            )
            and tuple(mapping[y] for y in a.query_targets) == b.query_targets
            and a.query_targets != b.query_targets
        )
    if pair.name == "affine-extrapolation":
        low = min(
            float(cast(float, r.values[0])) for r in a.context
        )  # validated numeric by reference
        high = max(float(cast(float, r.values[0])) for r in a.context)
        return (
            a.features == b.features
            and a.context == b.context
            and a.context_targets == b.context_targets
            and tuple(r.id for r in a.queries) == tuple(r.id for r in b.queries)
            and all(low <= float(cast(float, r.values[0])) <= high for r in a.queries)
            and all(
                float(cast(float, r.values[0])) < low or float(cast(float, r.values[0])) > high
                for r in b.queries
            )
        )
    raise ValueError("unknown public challenge transformation")


def evaluate_pair(pair: ChallengePair) -> PairReport:
    """Retain failed comparisons; reference errors remain explicit invalid-case errors."""
    before, after = _episode_check(pair.before, pair), _episode_check(pair.after, pair)
    transformation = _transformation_valid(pair)
    passed = (
        transformation
        and all(before["reference"]["matches"])
        and all(after["reference"]["matches"])
        and all(before["shortcut"]["matches"])
        and not all(after["shortcut"]["matches"])
    )
    return {
        "name": pair.name,
        "family": pair.family,
        "shortcut": pair.shortcut,
        "transformation": pair.transformation,
        "derivation": pair.derivation,
        "before": before,
        "after": after,
        "transformation_valid": transformation,
        "status": "PASS" if passed else "FAIL",
    }
