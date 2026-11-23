"""Deterministic fixture replay with inspectable inputs, predictions and scores."""

import json
import math
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import TypedDict

import numpy as np

from r0.modules.episodes import Episode
from r0.modules.featurize import TransformedView, transform
from r0.modules.priors import SourceTask, sample_fixture
from r0.modules.replay.lib.config import Config as Config
from r0.modules.replay.lib.config import parse_config as parse_config
from r0.modules.replay.lib.identity import fingerprint as fingerprint
from r0.modules.replay.lib.oracle import oracle_replay as oracle_replay
from r0.modules.replay.lib.oracle import write_oracle_report as write_oracle_report
from r0.modules.replay.lib.oracle_compare import compare_oracle_report as compare_oracle_report
from r0.modules.replay.lib.oracle_config import OracleConfig as OracleConfig
from r0.modules.replay.lib.oracle_config import parse_oracle_config as parse_oracle_config
from r0.modules.replay.lib.oracle_scores import oracle_scores as oracle_scores


class Report(TypedDict):
    config: dict[str, object]
    config_fingerprint: str
    content_fingerprint: str
    environment: dict[str, str]
    source_task: dict[str, object]
    episode: dict[str, object]
    predictor_view: dict[str, object]
    transformed_view: dict[str, object]
    scoring: dict[str, object]


def _task(config: Config) -> tuple[SourceTask, str]:
    key = (config.generator_version, config.fixture, config.root_seed, config.task_key)
    rng = np.random.default_rng(int(fingerprint(key), 16))
    task = sample_fixture(config.fixture, rng=rng)
    return task, fingerprint((key, asdict(task)))


def _episode(config: Config, task: SourceTask, source_id: str) -> Episode:
    key = (source_id, config.split_seed, config.episode_key)
    rng = np.random.default_rng(int(fingerprint(key), 16))
    indices = [int(i) for i in rng.permutation(len(task.rows))]
    context, queries = indices[: config.context_rows], indices[config.context_rows :]
    return Episode(
        source_id,
        fingerprint((key, config.context_rows)),
        config.fixture,
        task.features,
        tuple(task.rows[i] for i in context),
        tuple(task.targets[i] for i in context),
        tuple(task.rows[i] for i in queries),
        tuple(task.targets[i] for i in queries),
        task.classes,
        task.class_provenance,
    )


def build_episode(config: Config) -> Episode:
    task, source_id = _task(config)
    return _episode(config, task, source_id)


def predict(view: TransformedView) -> tuple[tuple[float, ...], ...]:
    """Fixture-only predictors consume the restricted transformed view."""
    if view.task_type == "regression":
        prediction = (mean(float(y) for y in view.context_targets),)
    else:
        prediction = tuple(
            view.context_targets.count(label) / len(view.context_targets) for label in view.classes
        )
    return tuple(prediction for _ in view.queries)


def score(episode: Episode, predictions: tuple[tuple[float, ...], ...]) -> dict[str, object]:
    """Query-only fixture scoring after prediction has completed."""
    if len(predictions) != len(episode.queries):
        raise ValueError("prediction count must match query rows")
    width = 1 if episode.task_type == "regression" else len(episode.classes)
    for prediction in predictions:
        if len(prediction) != width:
            raise ValueError("prediction width does not match task outputs")
        if any(not math.isfinite(value) for value in prediction):
            raise ValueError("predictions must be finite")
        if episode.task_type != "regression" and (
            any(value < 0 or value > 1 for value in prediction)
            or not math.isclose(sum(prediction), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("class probabilities must be in [0,1] and sum to one")
    if episode.task_type == "regression":
        terms = tuple(
            (p[0] - float(y)) * (p[0] - float(y))
            for p, y in zip(
                predictions,
                episode.query_targets,
                strict=True,
            )
        )
        metric = "mean_squared_error"
    else:
        terms = tuple(
            sum(
                (v - float(label == y)) ** 2
                for label, v in zip(
                    episode.classes,
                    p,
                    strict=True,
                )
            )
            for p, y in zip(predictions, episode.query_targets, strict=True)
        )
        metric = "multiclass_brier_sum_classes_mean_queries"
    if any(not math.isfinite(term) for term in terms):
        raise ValueError("score exceeds finite numeric range")
    return {
        "query_ids": [r.id for r in episode.queries],
        "targets": episode.query_targets,
        "predictions": predictions,
        "per_query_terms": terms,
        "metric": metric,
        "value": mean(terms),
    }


def replay(config: Config) -> Report:
    task, source_id = _task(config)
    episode = _episode(config, task, source_id)
    view = episode.predictor_view()
    transformed = transform(view)
    predictions = predict(transformed)
    scoring = score(episode, predictions)
    payload = (asdict(episode), asdict(transformed), predictions)
    return {
        "config": asdict(config),
        "config_fingerprint": fingerprint(asdict(config)),
        "content_fingerprint": fingerprint(payload),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "r0": version("r0"),
        },
        "source_task": {"id": source_id, **asdict(task)},
        "episode": asdict(episode),
        "predictor_view": asdict(view),
        "transformed_view": asdict(transformed),
        "scoring": scoring,
    }


def write_report(config_path: "Path", output_path: "Path") -> Report:
    """Read configuration and atomically replace a report after complete serialization."""
    import os
    import tempfile

    report = replay(parse_config(config_path.read_text()))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return report
