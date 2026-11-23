"""Bounded authored reference replay; prediction finishes before scoring begins."""

import json
import os
import platform
import tempfile
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np

from r0.modules.episodes import TaskType
from r0.modules.oracles import (
    ClassificationSpec,
    classification,
    classification_prior,
    gp,
    gp_prior,
)
from r0.modules.priors import authored_episode
from r0.modules.replay.lib.identity import fingerprint
from r0.modules.replay.lib.oracle_compare import compare_oracle_report
from r0.modules.replay.lib.oracle_config import OracleConfig, parse_oracle_config
from r0.modules.replay.lib.oracle_scores import oracle_scores


def oracle_replay(config: OracleConfig) -> dict[str, object]:
    reference = config.reference
    source = {
        "authorship": "explicit observations; not randomly sampled",
        "context_x": config.context_x,
        "context_y": config.context_y,
        "query_x": config.query_x,
        "query_targets": config.query_targets,
    }
    source_id = fingerprint(source)
    reference_id = fingerprint(asdict(reference))
    episode = None
    if config.mode == "posterior":
        task_type: TaskType = "regression"
        classes: tuple[str, ...] = ()
        if isinstance(reference, ClassificationSpec):
            task_type = "binary" if reference.family == "beta_bernoulli" else "multiclass"
            classes = reference.classes
        episode = authored_episode(
            source_id=source_id,
            episode_id=fingerprint((source_id, reference_id)),
            task_type=task_type,
            context_x=config.context_x,
            context_y=config.context_y,
            query_x=config.query_x,
            query_targets=config.query_targets,
            classes=classes,
        )
        view = episode.predictor_view()
        prediction = (
            classification(view, reference)
            if isinstance(reference, ClassificationSpec)
            else gp(view, reference)
        )
        predictor_input: dict[str, object] = asdict(view)
    else:
        prediction = (
            classification_prior(reference, len(config.query_x))
            if isinstance(reference, ClassificationSpec)
            else gp_prior(config.query_x, reference)
        )
        predictor_input = {"query_x": config.query_x, "context": (), "mode": "prior"}
    payload: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "config_fingerprint": fingerprint(asdict(config)),
        "source_task": {"id": source_id, **source},
        "reference_id": reference_id,
        "reference": asdict(reference),
        "episode": asdict(episode) if episode else None,
        "predictor_view": predictor_input,
        "prediction": asdict(prediction),
        "units": {"feature": config.feature_unit, "target": config.target_unit},
        "preprocessing": "raw coordinates and original target units",
        "scoring": oracle_scores(episode, prediction) if episode else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "r0": version("r0"),
        },
        "numerical_policy": {
            "atol": config.atol,
            "rtol": config.rtol,
            "reason": config.tolerance_reason,
            "dtype": "float64",
        },
    }
    payload["verification"] = {
        "status": "NOT_RUN",
        "reason": "no independent reference card supplied",
    }
    payload["content_fingerprint"] = fingerprint(payload)
    return payload


def write_oracle_report(
    config_path: Path, output_path: Path, expected_path: Path | None = None
) -> dict[str, object]:
    """Atomically create an immutable report; retain comparison failures as evidence."""
    config = parse_oracle_config(config_path.read_text())
    report = oracle_replay(config)
    if expected_path is not None:
        expected: object = json.loads(expected_path.read_text())
        if not isinstance(expected, dict):
            raise ValueError("reference card must be a JSON object of expected numeric values")
        report["verification"] = compare_oracle_report(
            report, cast(dict[str, float], expected), atol=config.atol, rtol=config.rtol
        )
        report["reference_card_fingerprint"] = fingerprint(cast(dict[str, object], expected))
        del report["content_fingerprint"]
        report["content_fingerprint"] = fingerprint(report)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_path.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if output_path.read_text() != encoded:
                raise ValueError(
                    "output already contains a different run record; choose a new path"
                ) from None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return report
