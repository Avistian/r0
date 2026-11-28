"""Typed conversion composes with measurements, masks and saved source restoration."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def encodings() -> tuple[p.ColumnEncoding, ...]:
    return (
        p.ColumnEncoding(0, "ordinal", (-1.0, 1.0), ("low", "mid", "high")),
        p.ColumnEncoding(1, "count", count_rounding="ceil"),
    )


@pytest.mark.parametrize("measure", [False, True])
def test_typed_observation_archive_and_source_restore(tmp_path: Path, measure: bool) -> None:
    source = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=30,
        task_id=7,
    )
    stages = (p.MeasurementStage("affine", (0,), (("scale", 2.0),)),) if measure else ()
    task = p.observe_missingness(
        source,
        p.MissingnessPlan("mcar", (0, 1), probability=0.25),
        seed=42,
        measurements=stages,
        encodings=encodings(),
    )
    assert task.missing_mask is not None
    expected = source.inputs.numpy().copy()
    if measure:
        expected[:, 0] *= 2
    expected[:, 0] = [int(x > -1) + int(x > 1) for x in expected[:, 0]]
    expected[:, 1] = np.ceil(np.maximum(0, expected[:, 1]))
    present = ~task.missing_mask.numpy()
    np.testing.assert_array_equal(task.inputs.numpy()[present], expected[present])
    assert task.feature_schema == (
        p.TrainingFeature("ordinal", ("low", "mid", "high")),
        p.TrainingFeature("count"),
    )
    path = tmp_path / "typed.zip"
    g.save_group(path, [(7, task)], max_bytes=8000000)
    saved = next(g.load_group(path))
    assert saved.episode() == task.episode()
    restored = p.restore_observation_source(saved)
    assert restored.episode() == source.episode()
    assert restored.feature_schema == source.feature_schema
    with pytest.raises(ValueError, match="schema"):
        replace(
            task,
            feature_schema=(
                p.TrainingFeature("ordinal", ("HIGH", "MID", "LOW")),
                p.TrainingFeature("count"),
            ),
        )
    with pytest.raises(ValueError, match="truth"):
        replace(task, observation_truth=None)
    assert task.observation_truth is not None
    raw = task.observation_truth.raw_inputs.clone()
    raw[0, 0] -= 100
    with pytest.raises(ValueError, match="receipt"):
        replace(task, observation_truth=replace(task.observation_truth, raw_inputs=raw))


def test_typed_guarded_request_and_resume(tmp_path: Path) -> None:
    observation = g.ObservationRequest(
        p.MissingnessPlan("mcar", (0, 1), probability=0.2), encodings=encodings()
    )
    request = g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=30,
        count=2,
        max_seconds=30,
        max_bytes=8000000,
        observation=observation,
    )
    assert g.request_from_dict(json.loads(json.dumps(asdict(request)))) == request
    run = g.run_guarded(request, tmp_path, workers=2)
    assert run.returncode == 0, run.stderr
    assert run.summary is not None
    assert run.summary["completed"] == 2
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
    changed = replace(
        request,
        observation=replace(
            observation, encodings=(replace(encodings()[0], equality="upper"), encodings()[1])
        ),
    )
    rejected = g.run_guarded(changed, tmp_path)
    assert rejected.returncode != 0
    assert "configuration changed" in rejected.stderr
